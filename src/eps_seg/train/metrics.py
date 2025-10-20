from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def compute_unsupervised_metrics(batch_size, z, outputs):
    """
    Compute similarity metrics for unsupervised training based on quadrant structure.
    
    Args:
        batch_size (int): Number of samples in the batch.
        z (torch.Tensor): Latent representations [batch, H, W].
        outputs (dict): Forward pass outputs containing 'q' (quadrants dict).
    
    Returns:
        dict: Dictionary containing TP, TN, FP, FN, Precision, Recall, F1.
    """
    # Generate all unique pairs
    pairs = [(i, j) for i in range(batch_size) for j in range(i + 1, batch_size)]
    
    quadrants = outputs["q"]
    z = z.squeeze()
    center_y, center_x = 31, 31  # patch center
    patch_labels = z[:, center_y, center_x]
    
    # Collect labels for each quadrant
    quadrant_pair_labels = {}
    for quadrant, pair_indices in quadrants.items():
        selected_pairs = [pairs[i] for i in pair_indices.tolist()]
        labels = [(patch_labels[i].item(), patch_labels[j].item()) for i, j in selected_pairs]
        quadrant_pair_labels[quadrant] = labels
    
    # Define expected similarity per quadrant
    quadrant_expectation = {
        "top_left": 0,
        "top_right": 0,
        "bottom_left": 1,
        "bottom_right": 1,
    }
    
    # Compare predicted vs expected similarity
    y_true, y_pred = [], []
    for quadrant, labels in quadrant_pair_labels.items():
        expected = quadrant_expectation[quadrant]
        for label_i, label_j in labels:
            pred = int(label_i == label_j)
            y_true.append(expected)
            y_pred.append(pred)
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
