import os
import numpy as np
import time
import datetime
import torch
from torch.amp import GradScaler
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from boilerplate import boilerplate
import wandb
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def train_network(
    model,
    lr,
    max_epochs,
    train_loader,
    val_loader,
    gaussian_noise_std,
    model_name,
    directory_path="./",
    batch_size=8,
    alpha=1,
    beta=1,
    gamma=1,
    max_grad_norm=None,
    amp=True,
    gradient_scale=8192,
    use_wandb=True,
    initial_label_size=1,
    final_label_size=10,
    initial_mask_size=1,
    final_mask_size=10,
    step_interval=20,
):
    """Train Hierarchical DivNoising network.
    Parameters
    ----------
    model: Ladder VAE object
        Hierarchical DivNoising model.
    lr: float
        Learning rate
    max_epochs: int
        Number of epochs to train the model for.
    train_loader: PyTorch data loader
        Data loader for training set.
    val_loader: PyTorch data loader
        Data loader for validation set.
    test_loader: PyTorch data loader
        Data loader for test set.
    gaussian_noise_std: float
        standard deviation of gaussian noise (required when 'noiseModel' is None).
    model_name: String
        Name of Hierarchical DivNoising model with which to save weights.
    directory_path: String
        Path where the DivNoising weights to be saved.
    max_grad_norm: float
        Value to limit/clamp the gradients at.
    """
    cudnn.benchmark = True
    cudnn.fastest = True
    change_mask_size = False
    change_label_size = False
    model_folder = directory_path + "model_" + model.training_mode + "/"
    device = model.device
    #optimizer, scheduler = boilerplate._make_optimizer_and_scheduler(model, lr, 0.0)
    # if initial_mask_size != final_mask_size:
    #     mask_size_scheduler = boilerplate.LabelSizeScheduler(
    #         initial_size=initial_mask_size,
    #         final_size=final_mask_size,
    #         step_interval=step_interval,
    #     )
    #     change_mask_size = True
    # if initial_label_size != final_label_size:
    #     label_size_scheduler = boilerplate.LabelSizeScheduler(
    #         initial_size=initial_label_size,
    #         final_size=final_label_size,
    #         step_interval=step_interval,
    #     )
    #     change_label_size = True

    loss_val_history = []

    patience_ = 0

    try:
        os.makedirs(model_folder)
    except FileExistsError:
        # directory already exists
        pass

    seconds_last = time.time()

    os.environ["WANDB_START_TIMEOUT"] = "600"

    # AMP gradscaler (This is managed by lightning in the refactored code, even though it does not expose a gradscaler attribute)
    #scaler = GradScaler(init_scale=gradient_scale, enabled=amp)

    if use_wandb:
        wandb.login()
        run = wandb.init(
            project=model_name,
            config={
                "learning rate": lr,
                "epochs": max_epochs,
                "batch size": batch_size,
                "inpainting loss weight": alpha,
                "KLD weight": beta,
                "contrastive learning weight": gamma,
                "lambda (cl)": model.lambda_contrastive,
                "margin": model.margin,
            },
        )
        run.config.update(dict(epochs=max_epochs))
        wandb.run.log_code(
            ("/home/sheida.rahnamai/GIT/My_Plugin/epsSeg/"),
            include_fn=lambda path: path.endswith(".py")
            or path.endswith(".ipynb" or path.endswith(".sbatch")),
        )
    #threshold = 0.50
    for epoch in range(max_epochs):

        print(f"Starting epoch {epoch}")
        log_interval = 5  # Log every 5 batches
        running_metrics = {
            "IP": 0,
            "KL": 0,
            "CL": 0,
            "CE": 0,
            "EL": 0,
            "Total": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }
        
        for idx, (x, y, z, _) in tqdm(enumerate(train_loader), desc="Training"):
            # if not use_wandb:
            #     if idx == 5:
            #         break
            # if change_label_size:
            #     train_loader.dataset.update_patches(
            #         label_size_scheduler.get_label_size(patience_)
            #     )
            # if change_mask_size:
            #     model.mask_size = mask_size_scheduler.get_label_size(patience_)
            x = x.squeeze(0)
            y = y.squeeze(0)
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)

            optimizer.zero_grad()

            if torch.isnan(x).any() or torch.isinf(x).any():
                print("x has nan or inf")
                continue

            outputs = boilerplate.forward_pass(
                x, y, device, model, gaussian_noise_std, amp=amp, threshold=threshold
            )
            
            ################################################################
            if model.training_mode == "unsupervised":
                pairs = [
                    (i, j) for i in range(batch_size) for j in range(i + 1, batch_size)
                ]
                quadrants = outputs["q"]
                z = z.squeeze()
                center_y, center_x = 31, 31
                patch_labels = z[:, center_y, center_x]

                quadrant_pair_labels = {}

                for quadrant, pair_indices in quadrants.items():
                    # Extract relevant pairs
                    selected_pairs = [pairs[i] for i in pair_indices.tolist()]

                    labels = []
                    for i, j in selected_pairs:
                        li = patch_labels[i].item()
                        lj = patch_labels[j].item()
                        labels.append((li, lj))

                    quadrant_pair_labels[quadrant] = labels

                quadrant_expectation = {
                    "top_left": 0,  # expect dissimilar
                    "top_right": 0,  # expect dissimilar
                    "bottom_left": 1,  # expect similar
                    "bottom_right": 1,  # expect similar
                }

                y_true = []  # expected similarity: 1 for similar, 0 for dissimilar
                y_pred = []  # predicted similarity: based on label equality

                for quadrant, pairs in quadrant_pair_labels.items():
                    expected = quadrant_expectation[quadrant]
                    for label_i, label_j in pairs:
                        pred = int(label_i == label_j)
                        y_true.append(expected)
                        y_pred.append(pred)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                running_metrics["tp"] += tp
                running_metrics["tn"] += tn
                running_metrics["fp"] += fp
                running_metrics["fn"] += fn
                running_metrics["precision"] += precision
                running_metrics["recall"] += recall
                running_metrics["f1"] += f1

            ################################################################

            # inpainting_loss = outputs["inpainting_loss"]
            # kl_loss = outputs["kl_loss"]
            # cl_loss = (
            #     outputs["cl_loss"]
            #     if not torch.isnan(outputs["cl_loss"])
            #     else torch.tensor(0.0, dtype=torch.float32, device=device)
            # )
            # ce = outputs["ce"]
            # entropy = outputs["entropy"]

            # loss = (
            #     alpha * inpainting_loss
            #     + beta * kl_loss
            #     + gamma * cl_loss
            #     + ce 
            #     + entropy
            # )

            # with torch.autograd.set_detect_anomaly(mode=True):
            #     scaler.scale(loss).backward()

            # if max_grad_norm is not None:
            #     torch.nn.utils.clip_grad_norm_(
            #         model.parameters(), max_norm=max_grad_norm
            #     )

            # # Optimization step

            # scaler.step(optimizer)
            # scaler.update()
            # model.increment_global_step()

            # # scaled_loss = scaler.scale(loss)
            # # scaled_loss.backward()
            # # scaler.unscale_(optimizer)
            # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # # scaler.step(optimizer)
            # # scaler.update()
            # # model.increment_global_step()

            # # Accumulate loss metrics
            # running_metrics["IP"] += inpainting_loss.item() * alpha
            # running_metrics["KL"] += kl_loss.item() * beta
            # running_metrics["CL"] += cl_loss.item() * gamma
            # running_metrics["CE"] += ce.item() 
            # running_metrics["EL"] += entropy.item()
            # running_metrics["Total"] += loss.item()

            # Log every `log_interval` batches
            if (idx + 1) % log_interval == 0:
                avg_metrics = {
                    key: value / log_interval for key, value in running_metrics.items()
                }

                if use_wandb:
                    run.log(avg_metrics)
                    # Reset accumulated metrics
                    running_metrics = {key: 0 for key in running_metrics}

        print("saving", model_folder + "last.net")
        torch.save(model, model_folder + "last.net")
        # print(f"Threshold = {threshold}")
        # if model.training_mode == "semisupervised" and threshold < 0.99:
        #     threshold += 0.005
        
        ### Validation step
        running_validation_loss = []

        model.eval()
        # Before validation loop
        val_metrics = {
            "val_IP": 0,
            "val_KL": 0,
            "val_CE": 0,
            "val_EL": 0,
            "val_CL": 0,
            "val_total": 0,
        }
        num_val_batches = len(val_loader)

        with torch.no_grad():
            for idx, (x, y, z, _) in tqdm(enumerate(val_loader), desc="Validation"):
                if not use_wandb:
                    if idx == 5:
                        break
                if change_label_size:
                    val_loader.dataset.update_patches(
                        label_size_scheduler.get_label_size(patience_)
                    )
                if change_mask_size:
                    model.mask_size = mask_size_scheduler.get_label_size(patience_)
                x = x.squeeze(0)
                y = y.squeeze(0)
                z = z.squeeze(0)
                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                val_outputs = boilerplate.forward_pass(
                    x, y, device, model, gaussian_noise_std
                )

                val_inpainting_loss = val_outputs["inpainting_loss"]
                val_kl_loss = val_outputs["kl_loss"]
                val_ce = val_outputs["ce"]
                val_entropy = val_outputs["entropy"]
                val_cl_loss = (
                    val_outputs["cl_loss"]
                    if not torch.isnan(val_outputs["cl_loss"])
                    else torch.tensor(0.0, dtype=torch.float32, device=device)
                )

                val_loss = (
                    alpha * val_inpainting_loss
                    + beta * val_kl_loss
                    + gamma * val_cl_loss
                    + val_ce 
                    + val_entropy
                )
                running_validation_loss.append(val_loss)
                # Accumulate batch-wise metrics
                val_metrics["val_IP"] += alpha * val_inpainting_loss
                val_metrics["val_KL"] += beta * val_kl_loss
                val_metrics["val_CE"] += val_ce
                val_metrics["val_EL"] += val_entropy
                val_metrics["val_CL"] += gamma * val_cl_loss
                val_metrics["val_total"] += val_loss

        # Compute the mean
        for key in val_metrics:
            if val_metrics[key] is not None:
                val_metrics[key] /= num_val_batches
        # Log once per validation cycle
        if use_wandb:
            run.log(val_metrics)

        model.train()

        total_epoch_loss_val = torch.mean(torch.stack(running_validation_loss))
        scheduler.step(total_epoch_loss_val)

        ### Save validation losses
        loss_val_history.append(total_epoch_loss_val.item())
        np.save(model_folder + "val_loss.npy", np.array(loss_val_history))

        if total_epoch_loss_val.item() < 1e-6 + np.min(loss_val_history):
            patience_ = 0
            print("saving", model_folder + "best.net")
            torch.save(model, model_folder + "best.net")
        else:
            patience_ += 1

        print(
            "Patience:",
            patience_,
            "Validation Loss:",
            total_epoch_loss_val.item(),
            "Min validation loss:",
            np.min(loss_val_history),
        )

        seconds = time.time()
        secondsElapsed = float(seconds - seconds_last)
        seconds_last = seconds
        remainingEps = (max_epochs + 1) - (epoch + 1)
        estRemainSecondsInt = int(secondsElapsed) * (remainingEps)
        print("Time for epoch: " + str(int(secondsElapsed)) + "seconds")

        print(
            "Est remaining time: "
            + str(datetime.timedelta(seconds=estRemainSecondsInt))
            + " or "
            + str(estRemainSecondsInt)
            + " seconds"
        )

        print("----------------------------------------", flush=True)

        if patience_ == 50 and train_loader.dataset.mode == "supervised":

            print("--------------------------------------")
            print("Switching to semi-supervised mode")
            print("--------------------------------------")
            train_loader.dataset.set_mode('semisupervised')
            checkpoint = torch.load(model_folder + "best.net", weights_only=False)
            model.load_state_dict(checkpoint.state_dict())
            
            model.update_mode("semisupervised")
            patience_ = 0
            shutil.copy(
                model_folder + "best.net",
                model_folder + "best_supervised.net",
            )
            

        if patience_ == 50 and train_loader.dataset.radius < 10 and train_loader.dataset.mode == "semisupervised":
            print("--------------------------------------")
            print(
                f"increasing radius from {train_loader.dataset.radius} to {train_loader.dataset.radius + 1}"
            )
            print("--------------------------------------")

            train_loader.dataset.increase_radius()
            patience_ = 0
            checkpoint = torch.load(model_folder + "best.net", weights_only=False)
            model.load_state_dict(checkpoint.state_dict())
            

        if patience_ == 51:
            print("Early stopping")
            break