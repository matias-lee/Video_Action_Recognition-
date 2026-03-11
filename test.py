"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_multiclass_metrics: Computes Macro F1 and Macro AUC scores for multiclass evaluation.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm


def test(model, dataloader, device):
    """
    Evaluate the model on the test dataset and compute overall accuracy.
    Returns targets, outputs, output_probs, and accuracy.
    """
    model.eval()
    with torch.no_grad():
        total_correct_preds = 0.0
        len_dataset = len(dataloader.dataset)
        targets, outputs, output_probs = [], [], []

        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Get raw logits from the model
            logits = model(x_batch)

            # Convert logits to probabilities for AUC calculation
            probs = F.softmax(logits, dim=1)

            # Get predicted classes
            pred = logits.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(y_batch.view_as(pred)).sum().item()
            total_correct_preds += correct_preds

            # Store targets, predictions, and probabilities
            outputs.extend(pred.view(-1).detach().cpu().numpy().tolist())
            targets.extend(y_batch.detach().cpu().numpy().tolist())
            output_probs.extend(probs.detach().cpu().numpy().tolist())

        accuracy = total_correct_preds / float(len_dataset)

    return targets, outputs, output_probs, accuracy


def get_multiclass_metrics(targets, outputs, output_probs):
    """
    Compute Macro F1 and Macro AUC (One-vs-Rest) scores for multiclass evaluation.
    """
    # Calculate Macro F1
    f1_macro = f1_score(targets, outputs, average="macro")

    # Calculate Macro AUC (One-vs-Rest handles multiclass by comparing each class against all others)
    auc_macro = roc_auc_score(targets, output_probs, multi_class="ovr", average="macro")

    return f1_macro, auc_macro


def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.

    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.

    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.

    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)


def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.

    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.

    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.

    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return dict(zip(all_cats, confusion_mat))


def plot_and_save_confusion_matrix(
    targets, outputs, target_names, save_path="ucf50_confusion_matrix.png"
):
    """
    Computes a standard multiclass confusion matrix, renders it as a heatmap,
    and saves it to disk.
    """
    # Compute the 50x50 confusion matrix
    cm = confusion_matrix(targets, outputs)

    # Set up the matplotlib figure (large enough to read 50 class labels)
    plt.figure(figsize=(22, 20))

    # Create a beautifully formatted heatmap using seaborn
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)

    plt.ylabel("True Action Class", fontsize=16)
    plt.xlabel("Predicted Action Class", fontsize=16)
    plt.title("UCF50 LRCN Confusion Matrix", fontsize=20)

    # Rotate the x-axis labels so they don't overlap
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save to disk
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Confusion matrix heatmap saved to: {save_path}")
    plt.close()
