import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction import image
from tqdm import tqdm
from matplotlib import pyplot as plt

class Interpolate(nn.Module):
    """Wrapper for torch.nn.functional.interpolate."""

    def __init__(self, size=None, scale=None, mode="bilinear", align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return out


class CropImage(nn.Module):
    """Crops image to given size.
    Args:
        size
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return crop_img_tensor(x, self.size) # tODO: Fix this assertion error assert len(size) in [2, 3], "Invalid input depth dimension"
        # return x


def normalize(img, mean, std):
    """Normalize an array of images with mean and standard deviation.
    Parameters
    ----------
    img: array
        An array of images.
    mean: float
        Mean of img array.
    std: float
        Standard deviation of img array.
    """
    return (img - mean) / std


def denormalize(img, mean, std):
    """Denormalize an array of images with mean and standard deviation.
    Parameters
    ----------
    img: array
        An array of images.
    mean: float
        Mean of img array.
    std: float
        Standard deviation of img array.
    """
    return (img * std) + mean


def convertToFloat32(train_images, val_images):
    """Converts the data to float 32 bit type.
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
    """
    x_train = train_images.astype("float32")
    x_val = val_images.astype("float32")
    return x_train, x_val


def getMeanStdData(train_images, val_images):
    """Compute mean and standrad deviation of data.
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
    """
    x_train_ = train_images.astype("float32")
    x_val_ = val_images.astype("float32")
    data = np.concatenate((x_train_, x_val_), axis=0)
    mean, std = np.mean(data), np.std(data)
    return mean, std


def convertNumpyToTensor(numpy_array):
    """Convert numpy array to PyTorch tensor.
    Parameters
    ----------
    numpy_array: numpy array
        Numpy array.
    """
    return torch.from_numpy(numpy_array)


def augment_data(patches):
    if len(patches.shape[1:]) == 2:
        augmented = np.concatenate(
            (
                patches,
                np.rot90(patches, k=1, axes=(1, 2)),
                np.rot90(patches, k=2, axes=(1, 2)),
                np.rot90(patches, k=3, axes=(1, 2)),
            )
        )
    elif len(patches.shape[1:]) == 3:
        augmented = np.concatenate(
            (
                patches,
                np.rot90(patches, k=1, axes=(2, 3)),
                np.rot90(patches, k=2, axes=(2, 3)),
                np.rot90(patches, k=3, axes=(2, 3)),
            )
        )

    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented


def extract_patches(x, patch_size, num_patches):
    """Deterministically extract patches from array of images.
    Parameters
    ----------
    x: numpy array
        Array of images.
    patch_size: int
        Size of patches to be extracted from each image.
    num_patches: int
        Number of patches to be extracted from each image.
    """
    patches = np.zeros(shape=(x.shape[0] * num_patches, patch_size, patch_size))

    for i in tqdm(range(x.shape[0])):
        patches[i * num_patches : (i + 1) * num_patches] = image.extract_patches_2d(
            image=x[i],
            patch_size=(patch_size, patch_size),
            max_patches=num_patches,
            random_state=i,
        )

    return patches


def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) or (batch, channels, d, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The cropped tensor
    """
    return _pad_crop_img(x, size, "crop")


def _pad_crop_img(x, size, mode) -> torch.Tensor:
    """Pads or crops a tensor.
    Pads or crops a tensor of shape (batch, channels, h, w) or (batch, channels, d, h, w) to new height
    and width given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (depth: opt, height, width)
        mode (str): Mode, either 'pad' or 'crop'
    Returns:
        The padded or cropped tensor
    """
    assert x.dim() in [4, 5], "Invalid input array dimension"
    assert len(size) in [2, 3], "Invalid input depth dimension"
    # assert
    size = tuple(size)
    x_size = x.size()[2:]

    if mode == "pad":
        cond = any(x_size) > any(size)
    elif mode == "crop":
        cond = any(x_size) < any(size)
    else:
        raise ValueError(f"invalid mode {mode}")

    if cond:
        raise ValueError(f"trying to {mode} from size {x_size} to size {size}")

    padding = []
    for d in reversed(range(len(x_size))):
        pad_val = abs(x_size[d] - size[d])
        padding.append(pad_val // 2)
        padding.append(pad_val - (pad_val // 2))

    if mode == "pad":
        return nn.functional.pad(x, padding)
    elif mode == "crop":
        if len(x_size) == 2:
            return x[
                :,
                :,
                padding[2] : x_size[0] - padding[3],
                padding[0] : x_size[1] - padding[1],
            ]
        elif len(x_size) == 3:
            return x[
                :,
                :,
                padding[4] : x_size[0] - padding[5],
                padding[2] : x_size[1] - padding[3],
                padding[0] : x_size[2] - padding[1],
            ]


def free_bits_kl(kl, free_bits, batch_average=False, eps=1e-6) -> torch.Tensor:
    """Computes free-bits version of KL divergence.
    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.
    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).
    Args:
        kl (torch.Tensor)
        free_bits (float)
        batch_average (bool, optional))
        eps (float, optional)
    Returns:
        The KL with free bits
    """

    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)


def pad_img_tensor(x, size) -> torch.Tensor:
    """Pads a tensor.
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The padded tensor
    """

    return _pad_crop_img(x, size, "pad")


def plotProbabilityDistribution(
    signalBinIndex,
    histogram,
    gaussianMixtureNoiseModel,
    min_signal,
    max_signal,
    n_bin,
    device,
):
    """Plots probability distribution P(x|s) for a certain ground truth signal.
    Predictions from both Histogram and GMM-based Noise models are displayed for comparison.
     Parameters
     ----------
     signalBinIndex: int
         index of signal bin. Values go from 0 to number of bins (`n_bin`).
     histogram: numpy array
         A square numpy array of size `nbin` times `n_bin`.
     gaussianMixtureNoiseModel: GaussianMixtureNoiseModel
         Object containing trained parameters.
     min_signal: float
         Lowest pixel intensity present in the actual sample which needs to be denoised.
     max_signal: float
         Highest pixel intensity present in the actual sample which needs to be denoised.
     n_bin: int
         Number of Bins.
     device: GPU device
    """
    histBinSize = (max_signal - min_signal) / n_bin
    querySignal_numpy = (
        signalBinIndex / float(n_bin) * (max_signal - min_signal) + min_signal
    )
    querySignal_numpy += histBinSize / 2
    querySignal_torch = torch.from_numpy(np.array(querySignal_numpy)).float().to(device)

    queryObservations_numpy = np.arange(min_signal, max_signal, histBinSize)
    queryObservations_numpy += histBinSize / 2
    queryObservations = torch.from_numpy(queryObservations_numpy).float().to(device)
    pTorch = gaussianMixtureNoiseModel.likelihood(queryObservations, querySignal_torch)
    pNumpy = pTorch.cpu().detach().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel("Observation Bin")
    plt.ylabel("Signal Bin")
    plt.imshow(histogram**0.25, cmap="gray")
    plt.axhline(y=signalBinIndex + 0.5, linewidth=5, color="blue", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(
        queryObservations_numpy,
        histogram[signalBinIndex, :] / histBinSize,
        label="GT Hist: bin =" + str(signalBinIndex),
        color="blue",
        linewidth=2,
    )
    plt.plot(
        queryObservations_numpy,
        pNumpy,
        label="GMM : " + " signal = " + str(np.round(querySignal_numpy, 2)),
        color="red",
        linewidth=2,
    )
    plt.xlabel("Observations (x) for signal s = " + str(querySignal_numpy))
    plt.ylabel("Probability Density")
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal_numpy))
    plt.legend()


def PSNR(gt, img, psnrRange):
    """
    Compute PSNR.
    Parameters
    ----------
    gt: array
        Ground truth image.
    img: array
        Predicted image.
    psnrRange: float
        Range PSNR
    """
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(psnrRange) - 10 * np.log10(mse)


def get_normalized_tensor(img, model, device):
    """
    Normalizes tensor with mean and std.
    Parameters
    ----------
    img: array
        Image.
    model: Hierarchical DivNoising model
    device: GPU device.
    """
    test_images = torch.from_numpy(img).to(device)
    data_mean = model.data_mean
    data_std = model.data_std
    test_images = (test_images - data_mean) / data_std
    return test_images


def compute_cl_loss(
    mus,
    logvars,
    pis,
    labels,
    margin=50,
    lambda_contrastive=0.5,
    training_mode='supervised',
    prior="normal",
):
    return multiscale_supervised_cl_sup(mus, labels)
    # if training_mode == 'supervised':
    #     return multiscale_supervised_cl_sup(mus, labels)
    # elif training_mode == 'semisupervised':
    #     return multiscale_supervised_cl_semi(mus, labels)
    
    # if training_mode == 'supervised':
    #     labeled_ratio = 1
    #     return multiscale_supervised_cl(mus, labels)
    # elif training_mode == 'semisupervised':
    #     labeled_ratio = 0.25
    #     return compute_semisupervised_cl_loss(mus, labels)
    # elif training_mode == 'unsupervised':
    #     labeled_ratio = 0
    #     return compute_unsupervised_cl_loss(mus, labels)
        
        
    # if prior == "mixture":
    #     pos_pair_loss, neg_pair_loss_terms = pos_neg_loss(
    #         mus, labels, margin=margin, labeled_ratio=labeled_ratio
    #     )
    # elif prior == "normal":
    #     pos_pair_loss, neg_pair_loss_terms = pos_neg_loss_normal(
    #         mus, labels, margin=margin, labeled_ratio=labeled_ratio
    #     )

    # neg_thetas = get_thetas(neg_pair_loss_terms)
    # weighted_neg = compute_weighted_neg(neg_pair_loss_terms, neg_thetas)
    # contrastive_loss = (
    #     lambda_contrastive * pos_pair_loss + (1 - lambda_contrastive) * weighted_neg
    # )
    # return contrastive_loss, None

def pct_equal_blocks(x: torch.Tensor) -> float:
    assert x.numel() % 4 == 0, "Length must be multiple of 4"
    blocks = x.view(-1, 4)                         # [B/4, 4]
    row_ok = (blocks == blocks[:, :1]).all(dim=1)  # [B/4] True if all 4 equal
    return (row_ok.float().mean().item() * 100.0)  # percentage


def compute_unsupervised_cl_loss(mus, coords):
    """
    Computes unsupervised contrastive loss.
    This function computes the contrastive loss based on the latent representation distances
    and the distance of the patches in pixel space.
    It uses the coordinates of the patches to calculate the distances in pixel space.
    Rank and extract specific patch pairs:
    16 closest in both (pixel + latent) → positive
    16 farthest in both (pixel + latent) → negative
    16 close in pixel but far in latent → negative
    16 far in pixel but close in latent → positive


    Args:
        mus (list): List of latent representations.
        coords (torch.Tensor): Coordinates of the patches.
    """
    B = coords.size(0)
    flat = [m.view(m.size(0), -1) for m in mus]
    z = torch.cat(flat, dim=1)
    
    latent_dist = torch.cdist(z, z, p=2)
    pixel_dist = torch.cdist(coords.float(), coords.float(), p=2)
    top_k = int(B / 128)

    latent_high_pixel_low, both_high, both_low, latent_low_pixel_high, q = get_contrastive_pairs(pixel_dist, latent_dist, top_k=top_k)
    positives = torch.stack(both_low + latent_low_pixel_high)  # These are semantically and spatially similar
    negatives = torch.stack(both_high + latent_high_pixel_low)  # These are dissimilar in either space
    m = 150
    target = torch.ones_like(positives)
    
    return F.margin_ranking_loss(negatives, positives, target, margin=m), q
    # return compute(positives, positive=True) + compute(negatives, positive=False), q

def compute(d, positive=True):
    return (d.pow(2).mean() if positive else F.relu(100 - d).pow(2).mean())

def get_contrastive_pairs(pixel_dist, latent_dist, top_k):
    N = pixel_dist.shape[0]
    # Extract upper triangle (i < j)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pixel_vals = torch.tensor([pixel_dist[i, j] for i, j in pairs], device=pixel_dist.device)
    latent_vals = torch.tensor([latent_dist[i, j] for i, j in pairs], device=latent_dist.device)
    q = get_percentile(pixel_vals, latent_vals, k=top_k)
    
    latent_high_pixel_low = [latent_dist[j] for j in [pairs[i] for i in q['top_left']]]  # High latent, low pixel
    both_high = [latent_dist[j] for j in [pairs[i] for i in q['top_right']]]  # High in both
    both_low = [latent_dist[j] for j in [pairs[i] for i in q['bottom_left']]]  # Low in both
    latent_low_pixel_high = [latent_dist[j] for j in [pairs[i] for i in q['bottom_right']]]  # Low latent, high pixel
    
    return both_low, both_high, latent_high_pixel_low, latent_low_pixel_high, q

def get_percentile(pixel_vals, latent_vals, k=4):
    
    x_10 = pixel_vals.kthvalue(int(0.10 * len(pixel_vals)))[0]
    x_25 = pixel_vals.kthvalue(int(0.25 * len(pixel_vals)))[0]
    x_75 = pixel_vals.kthvalue(int(0.75 * len(pixel_vals)))[0]
    x_90 = pixel_vals.kthvalue(int(0.90 * len(pixel_vals)))[0]

    y_05 = latent_vals.kthvalue(int(0.05 * len(latent_vals)))[0]
    y_15 = latent_vals.kthvalue(int(0.15 * len(latent_vals)))[0]
    y_85 = latent_vals.kthvalue(int(0.85 * len(latent_vals)))[0]
    y_95 = latent_vals.kthvalue(int(0.95 * len(latent_vals)))[0]

    quadrants = {}

    # Define inter-percentile masks
    masks = {
        "top_left":     (pixel_vals >= x_10) & (pixel_vals <= x_25) & (latent_vals >= y_85) & (latent_vals <= y_95),
        "top_right":    (pixel_vals >= x_75) & (pixel_vals <= x_90) & (latent_vals >= y_85) & (latent_vals <= y_95),
        "bottom_left":  (pixel_vals >= x_10) & (pixel_vals <= x_25) & (latent_vals >= y_05) & (latent_vals <= y_15),
        "bottom_right": (pixel_vals >= x_75) & (pixel_vals <= x_90) & (latent_vals >= y_05) & (latent_vals <= y_15),
    }

    for name, mask in masks.items():
        x = pixel_vals[mask]
        y = latent_vals[mask]
        indices = torch.arange(len(pixel_vals), device=pixel_vals.device)[mask]

        if x.numel() == 0:
            print(f"Warning: No points found in {name} quadrant.")
            quadrants[name] = []
            continue

        # Sort logic: for top -> highest y, for bottom -> lowest y
        if name.startswith("top"):
            sort_key = torch.stack([-y, x], dim=1)
        else:
            sort_key = torch.stack([y, x], dim=1)

        x_sort = torch.argsort(sort_key[:, 1], stable=True)
        yx_sort = torch.argsort(sort_key[x_sort, 0], stable=True)
        selected = indices[x_sort[yx_sort]][:k]
        quadrants[name] = selected

    return quadrants

def multiscale_supervised_cl_sup(mus, labels, margin=1.5):
    B = len(mus[0])
    device = mus[0].device
    # num_classes = torch.unique(labels).size(0)
    if labels is not None:
        labels = labels[2].view(-1)
    # print("unique percentage:", pct_equal_blocks(labels))
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))            # [B,B]
    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = same & ~eye                                        # same class, not self
    neg_mask = ~same 
    tri = torch.triu(torch.ones(B, B, dtype=torch.bool, device=device), diagonal=1)
    pos_mask = pos_mask & tri
    neg_mask = neg_mask & tri
    
    valid = labels != -1
    valid_mask = valid.unsqueeze(0) & valid.unsqueeze(1)   # both labels must be valid
    pos_mask = pos_mask & valid_mask
    neg_mask = neg_mask & valid_mask
    
    descriptors = torch.cat([F.adaptive_avg_pool2d(mus[i], (1,1)).squeeze(-1).squeeze(-1) for i in range(len(mus))], dim=1)
    descriptors = F.normalize(descriptors, dim=1)
    dist = torch.cdist(descriptors, descriptors, p=2)
    dist = torch.clamp(dist, min=0, max=1e6)
    pos_d = dist[pos_mask]
    neg_d = dist[neg_mask]
    pos_loss = (pos_d ** 2).mean() if pos_d.numel() > 0 else dist.new_tensor(0.)
    neg_loss = (F.relu(margin - neg_d) ** 2).mean() if neg_d.numel() > 0 else dist.new_tensor(0.)
    return pos_loss + neg_loss, None

def multiscale_supervised_cl_semi(
    mus,
    labels,
    margin=1.0,
):
    """
    Contrastive (Siamese hinge) with masks that ignore noisy positives *and* negatives.
    Batch layout: groups of 4, anchor at i%4==0 and neighbors at i+1..i+3.
    labels: [B], anchor labels are gold; neighbors inherit anchor label but may be noisy.
    """
    B = mus[0].size(0)
    device = mus[0].device
    idx = torch.arange(B, device=device)
    
    is_anchor  = (idx % 8 == 0)                 # [B]
    group_id   = idx // 8                      # [B]
    same_label = labels[:, None].eq(labels[None, :])     # [B,B]
    same_group = group_id[:, None].eq(group_id[None, :]) # [B,B]
    eye        = torch.eye(B, dtype=torch.bool, device=device)

    # helpers
    both_anchors   = (is_anchor[:, None]  &  is_anchor[None, :])     # both anchors
    one_anchor     = is_anchor[:, None] ^  is_anchor[None, :]        # exactly one is anchor
    neither_anchor = (~is_anchor[:, None] & ~is_anchor[None, :])  # neither is anchor

    # strong positives
    pos_mask = (both_anchors & same_label)# | (one_anchor & same_group)

    # strong negatives
    neg_mask = both_anchors & ~same_label
    
    # weak positives and negatives
    # weak_pos_mask = one_anchor & ~same_group & same_label
    # weak_neg_mask = one_anchor & ~same_group & ~same_label
    
    # weakest positives and negatives
    # weakest_pos_mask = neither_anchor & ~same_group & same_label
    # weakest_neg_mask = neither_anchor & ~same_group & ~same_label
    
    # pos_mask = pos_mask | weak_pos_mask | weakest_pos_mask
    # neg_mask = neg_mask | weak_neg_mask | weakest_neg_mask
    
    # --- descriptors (pooled + L2-normalized) ---
    z = torch.cat([F.adaptive_avg_pool2d(x, (1,1)).flatten(1) for x in mus], dim=1)
    z = F.normalize(z, dim=1)
    dist = torch.cdist(z, z, p=2)
    dist = torch.clamp(dist, min=0, max=1e6)

    # ===== compute loss =====
    tri = torch.triu(torch.ones(B, B, dtype=torch.bool, device=device), diagonal=1)
    pos_d = dist[pos_mask & tri & ~eye]
    neg_d = dist[neg_mask & tri & ~eye]

    pos_loss = (pos_d ** 2).mean() if pos_d.numel() else dist.new_tensor(0.)
    neg_loss = (F.relu(margin - neg_d) ** 2).mean() if neg_d.numel() else dist.new_tensor(0.)
    return pos_loss + neg_loss, None

def pos_neg_loss(mus, labels, margin=50.0, labeled_ratio=1):

    batch_size = len(mus[0])
    device = mus[0].device
    small_batch_size = int(batch_size * labeled_ratio)

    labels = labels[:small_batch_size]
    num_classes = torch.unique(labels).size(0)
    labels = labels.unsqueeze(0)
    boolean_matrix = (labels == labels.T).to(device=device)
    mask = torch.eye(small_batch_size, dtype=torch.bool).to(device)
    boolean_matrix = boolean_matrix.masked_fill(mask, 0)

    top_mus = mus[-1][:small_batch_size].view(small_batch_size, -1)
    top_mus = top_mus.unsqueeze(0)
    dist = torch.cdist(top_mus, top_mus, p=2).squeeze(0)
    dist = torch.clamp(dist, min=0, max=1e6)

    pos_pair_loss = torch.sum(boolean_matrix * dist) / torch.sum(boolean_matrix)

    neg_pair_loss_terms = {}
    for i in range(num_classes - 1):
        for j in range(i + 1, num_classes):
            mask_i = labels == i
            mask_j = labels == j
            mask_ij = mask_i & mask_j.T

            neg_bool_matrix = mask_ij.to(device=device)
            neg_loss = custom_distance_loss_masked(dist, neg_bool_matrix, margin=margin)

            num_neg_pairs = torch.sum(neg_bool_matrix)
            if num_neg_pairs == 0:
                neg_loss = torch.tensor(0.0, device=device)
            else:
                neg_loss /= num_neg_pairs

            neg_pair_loss_terms[f"{i}{j}"] = neg_loss

    for index, mu in enumerate(mus[:-1]):
        mus = mu.view(1, batch_size, -1)
        mus = mus[:, :small_batch_size]

        dist = torch.cdist(mus, mus, p=2).squeeze(0)

        dist = torch.clamp(dist, min=0, max=1e6)

        pos_pair_loss += torch.sum(boolean_matrix * dist) / (
            torch.sum(boolean_matrix) * (2 ** (2 - index))
        )

        for i in range(num_classes - 1):
            for j in range(i + 1, num_classes):
                mask_i = labels == i
                mask_j = labels == j
                mask_ij = mask_i & mask_j.T

                neg_bool_matrix = mask_ij.to(device=device)
                neg_loss = custom_distance_loss_masked(
                    dist, neg_bool_matrix, margin=margin
                )

                num_neg_pairs = torch.sum(neg_bool_matrix)
                if num_neg_pairs == 0:
                    neg_loss = torch.tensor(0.0, device=device)
                else:
                    neg_loss /= num_neg_pairs * (2 ** (2 - index))

                neg_pair_loss_terms[f"{i}{j}"] += neg_loss

    return pos_pair_loss, neg_pair_loss_terms


def pos_neg_loss_normal(mus, labels, margin=50.0, labeled_ratio=1):

    batch_size = len(mus[0])
    device = mus[0].device
    small_batch_size = int(batch_size * labeled_ratio)

    labels = labels[:small_batch_size]
    num_classes = torch.unique(labels).size(0)
    labels = labels.unsqueeze(0)
    boolean_matrix = (labels == labels.T).to(device=device)
    mask = torch.eye(small_batch_size, dtype=torch.bool).to(device)
    boolean_matrix = boolean_matrix.masked_fill(mask, 0)

    pos_pair_loss = 0
    neg_pair_loss_terms = {}

    for index, mu in enumerate(mus):
        mus = mu.view(1, batch_size, -1)
        mus = mus[:, :small_batch_size]

        dist = torch.cdist(mus, mus, p=2).squeeze(0)

        dist = torch.clamp(dist, min=1e-6, max=1e6)

        pos_pair_loss += torch.sum(boolean_matrix * dist) / (
            torch.sum(boolean_matrix) * (2 ** (2 - index))
        )

        for i in range(num_classes - 1):
            for j in range(i + 1, num_classes):
                mask_i = labels == i
                mask_j = labels == j
                mask_ij = mask_i & mask_j.T

                neg_bool_matrix = mask_ij.to(device=device)
                neg_loss = custom_distance_loss_masked(
                    dist, neg_bool_matrix, margin=margin
                )

                num_neg_pairs = torch.sum(neg_bool_matrix)
                if num_neg_pairs == 0:
                    neg_loss = torch.tensor(0.0, device=device)
                else:
                    neg_loss /= num_neg_pairs * (2 ** (2 - index))

                if f"{i}{j}" in neg_pair_loss_terms.keys():
                    neg_pair_loss_terms[f"{i}{j}"] += neg_loss
                else:
                    neg_pair_loss_terms[f"{i}{j}"] = neg_loss

    return pos_pair_loss, neg_pair_loss_terms


def custom_distance_loss_masked(distances, mask, margin=16.0, epsilon=1e-6, alpha=1.0):
    """
    Custom loss function to compute penalties only for selected elements based on a mask.

    Args:
        distances (torch.Tensor): Pairwise distances.
        mask (torch.Tensor): Boolean mask to select elements for loss computation.
        margin (float): The desired distance (e.g., 16.0).
        epsilon (float): Small constant to avoid division by zero.
        alpha (float): Scaling factor for the penalty term.

    Returns:
        torch.Tensor: Loss value.
    """
    # Select only the distances where the mask is True
    masked_distances = distances[mask]

    # Loss initialization
    loss = torch.zeros_like(masked_distances)

    # Penalize distances less than margin
    mask_small = masked_distances < margin
    penalty_small = (1 / (masked_distances[mask_small] + epsilon)) + alpha * (
        (margin - masked_distances[mask_small]) ** 2
    )
    loss[mask_small] = penalty_small

    # Leave distances greater than or equal to margin untouched or reward
    # mask_large = masked_distances >= margin

    # Sum up the loss
    return loss.sum()


def get_thetas(neg_pair_loss_terms):

    losses = torch.tensor(list(neg_pair_loss_terms.values()))
    if torch.all(losses == 0):
        normalized_losses = torch.ones_like(losses) / len(
            losses
        )  # Distribute uniformly
    else:
        normalized_losses = F.softmax(losses, dim=0)
    thetas = {
        key: normalized_losses[i].item()
        for i, key in enumerate(neg_pair_loss_terms.keys())
    }
    return thetas


def compute_weighted_neg(neg_pair_loss_terms, neg_thetas):

    weighted_neg = 0
    for pair, loss in neg_pair_loss_terms.items():
        weight = neg_thetas.get(pair, 1.0)
        if math.isnan(weight) or weight == 0:
            weight = 1e-6
        weighted_neg += weight * loss

    return weighted_neg
