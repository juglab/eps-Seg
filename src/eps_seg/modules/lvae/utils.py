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
        return crop_img_tensor(
            x, self.size
        )  # tODO: Fix this assertion error assert len(size) in [2, 3], "Invalid input depth dimension"
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
    labels,
    nips=True,
    margin_norm=1.4,  # for normalized space 
    margin_raw=5.0,  # for raw per-level features
    w_ms_sup=0.25,  # add a bit of multiscale in supervised
    w_nips_semi=0.2,  # add a bit of reweighted negs in semi
    learnable_thetas=True,
):
    if nips:
        pos_pair_loss, neg_terms = pos_neg_loss(mus, labels, margin=margin_raw)
        thetas = get_thetas(neg_terms, learnable=learnable_thetas)
        weighted_neg = compute_weighted_neg(neg_terms, thetas)
        nips_loss = 0.5 * pos_pair_loss + 0.5 * weighted_neg
        ms_loss = multiscale_cl(mus, labels, margin=margin_norm)
        return nips_loss + w_ms_sup * ms_loss
    else:
        ms_loss = multiscale_cl(mus, labels, margin=margin_norm)
        hard_neg = adaptive_neg_in_normalized_space(
            mus, labels, margin=margin_norm, learnable_thetas=learnable_thetas
        )
        return ms_loss + w_nips_semi * hard_neg


def multiscale_cl(mus, labels, margin=1.5):
    B = len(mus[0])
    device = mus[0].device
    if labels is not None:
        labels = labels[-1].view(-1)
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))  # [B,B]
    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = same & ~eye  # same class, not self
    neg_mask = ~same
    tri = torch.triu(torch.ones(B, B, dtype=torch.bool, device=device), diagonal=1)
    pos_mask = pos_mask & tri
    neg_mask = neg_mask & tri

    valid = labels != -1
    valid_mask = valid.unsqueeze(0) & valid.unsqueeze(1)  # both labels must be valid
    pos_mask = pos_mask & valid_mask
    neg_mask = neg_mask & valid_mask

    descriptors = torch.cat(
        [
            F.adaptive_avg_pool2d(mus[i], (1, 1)).squeeze(-1).squeeze(-1)
            for i in range(len(mus))
        ],
        dim=1,
    )
    descriptors = F.normalize(descriptors, dim=1)
    dist = torch.cdist(descriptors, descriptors, p=2)
    dist = torch.clamp(dist, min=0, max=1e6)
    pos_d = dist[pos_mask]
    neg_d = dist[neg_mask]
    pos_loss = (pos_d**2).mean() if pos_d.numel() > 0 else dist.new_tensor(0.0)
    neg_loss = (
        (F.relu(margin - neg_d) ** 2).mean()
        if neg_d.numel() > 0
        else dist.new_tensor(0.0)
    )
    return pos_loss + neg_loss


def adaptive_neg_in_normalized_space(mus, labels, margin=1.2, learnable_thetas=True):
    device = mus[0].device
    labels = labels[-1].view(-1)
    valid = (labels != -1)
    idxs = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if idxs.numel() < 2:
        return torch.zeros((), device=device)

    y = labels[idxs]
    D = torch.cat([F.adaptive_avg_pool2d(m[idxs], (1,1)).squeeze(-1).squeeze(-1) for m in mus], dim=1)
    D = F.normalize(D, dim=1)
    dist = torch.cdist(D.unsqueeze(0), D.unsqueeze(0), p=2).squeeze(0)

    classes = torch.unique(y).tolist()
    classes.sort()
    tri = torch.triu(torch.ones_like(dist, dtype=torch.bool), 1)

    neg_terms = {}
    for a in range(len(classes)-1):
        for b in range(a+1, len(classes)):
            ca, cb = classes[a], classes[b]
            mi = (y == ca).unsqueeze(1)
            mj = (y == cb).unsqueeze(0)
            mask = ((mi & mj) | (mj & mi)) & tri
            if mask.any():
                d = dist[mask]
                neg_terms[f"{ca}-{cb}"] = F.relu(margin - d).pow(2).mean()
            else:
                neg_terms[f"{ca}-{cb}"] = torch.zeros((), device=device)

    if not neg_terms:
        return torch.zeros((), device=device)

    thetas = get_thetas(neg_terms, learnable=learnable_thetas)
    return compute_weighted_neg(neg_terms, thetas)

def pos_neg_loss(mus, labels, margin=5.0):
    device = mus[0].device
    labeled_mask = labels[-1] != -1
    labeled_indices = torch.nonzero(labeled_mask, as_tuple=False).squeeze(-1)

    # If fewer than 2 labeled samples, there are no pairs to compare
    if labeled_indices.numel() < 2:
        return torch.tensor(0.0, device=device, dtype=torch.float32), {}

    labels_l = labels[-1][labeled_indices]
    classes = torch.unique(labels_l)

    # 2) Build positive-pair boolean matrix (diag excluded)
    n = labels_l.size(0)
    labels_row = labels_l.unsqueeze(0)  # [1, n]
    pos_mask = labels_row == labels_row.T  # [n, n]
    pos_mask.fill_diagonal_(False)
    pos_count = pos_mask.sum().clamp(min=1)  # avoid div-by-zero later

    def pairwise_dist(x):
        # x: [n, ...] -> flatten features per sample
        x = x.view(n, -1).unsqueeze(0)  # [1, n, d]
        d = torch.cdist(x, x, p=2).squeeze(0)  # [n, n]
        return torch.clamp(d, min=0, max=1e6)

    # 3) Top level distances and positive loss
    top = mus[-1][labeled_indices]
    dist_top = pairwise_dist(top)

    pos_pair_loss = (pos_mask.to(device) * dist_top).sum() / pos_count

    # 4) Negative pair losses per class pair
    neg_pair_loss_terms = {}
    # Iterate over *actual* class ids (not reindexed), excluding -1 already
    class_list = classes.tolist()
    class_list.sort()
    for idx_a in range(len(class_list) - 1):
        for idx_b in range(idx_a + 1, len(class_list)):
            ca, cb = class_list[idx_a], class_list[idx_b]

            mask_i = (labels_l == ca).unsqueeze(0)  # [1, n]
            mask_j = (labels_l == cb).unsqueeze(0)  # [1, n]
            neg_mask = (mask_i.T & mask_j) | (mask_j.T & mask_i)  # symmetrical [n, n]

            num_neg_pairs = neg_mask.sum()
            if num_neg_pairs.item() == 0:
                neg_loss = torch.tensor(0.0, device=device)
            else:
                neg_loss = custom_distance_loss_masked(
                    dist_top, neg_mask.to(device), margin=margin
                )
                neg_loss = neg_loss / num_neg_pairs

            neg_pair_loss_terms[f"{ca}-{cb}"] = neg_loss

    for level_idx, mu in enumerate(mus[:-1]):
        x = mu[labeled_indices]
        dist = pairwise_dist(x)

        # weight factor matches your original (top level handled separately)
        weight_div = 2 ** (2 - level_idx)
        pos_pair_loss += (pos_mask.to(device) * dist).sum() / (pos_count * weight_div)

        for idx_a in range(len(class_list) - 1):
            for idx_b in range(idx_a + 1, len(class_list)):
                ca, cb = class_list[idx_a], class_list[idx_b]

                mask_i = (labels_l == ca).unsqueeze(0)
                mask_j = (labels_l == cb).unsqueeze(0)
                neg_mask = (mask_i.T & mask_j) | (mask_j.T & mask_i)

                num_neg_pairs = neg_mask.sum()
                if num_neg_pairs.item() == 0:
                    add_loss = torch.tensor(0.0, device=device)
                else:
                    add_loss = custom_distance_loss_masked(
                        dist, neg_mask.to(device), margin=margin
                    )
                    add_loss = add_loss / (num_neg_pairs * weight_div)

                neg_pair_loss_terms[f"{ca}-{cb}"] += add_loss

    return pos_pair_loss, neg_pair_loss_terms


def custom_distance_loss_masked(distances, mask, margin=5.0, epsilon=1e-6, alpha=1.0):
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


def get_thetas(neg_pair_loss_terms, learnable=True):
    keys = list(neg_pair_loss_terms.keys())
    vals = torch.stack([neg_pair_loss_terms[k] for k in keys])
    if not learnable:
        vals = vals.detach()
    if (vals != 0).any():
        w = F.softmax(vals, dim=0)
    else:
        w = torch.full_like(vals, 1.0 / max(len(vals), 1))
    return {k: w[i].item() if not learnable else w[i] for i, k in enumerate(keys)}


def compute_weighted_neg(neg_pair_loss_terms, neg_thetas):
    device = (
        next(iter(neg_pair_loss_terms.values())).device
        if len(neg_pair_loss_terms)
        else "cpu"
    )
    weighted_neg = torch.zeros((), device=device)
    for pair, loss in neg_pair_loss_terms.items():
        weight = neg_thetas[pair]
        if isinstance(weight, float):
            if math.isnan(weight) or weight == 0:
                weight = 1e-6
            weighted_neg = weighted_neg + weight * loss
        else:
            weighted_neg = weighted_neg + weight * loss

    return weighted_neg
