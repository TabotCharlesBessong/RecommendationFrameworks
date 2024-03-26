import torch

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# setting up timer function
from timeit import default_timer as timer


def print_train_time(start: float, end: float, devive: torch.device = None):
    """Prints difference between start time and end time"""
    total_time = end - start
    print(f"Train time on {devive}: {total_time:.3f} seconds")
    return total_time
