import numpy as np

def wmape(y_true, y_pred):
    """
    MAPE is not a suitable accuracy metric since the percentage error is high when the ground truth is near-zero.
    Weighted mean absolute percent error is a more reliable metric that deals with these near-zero cases.

    :param y_true: ground truth values
    :param y_pred: output from model inference
    :return: WMAPE accuracy metric
    """
    return 100 * np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)
