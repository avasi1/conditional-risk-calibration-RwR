import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Any

from constants import SEED, DATA_SPLIT_RATIO


def get_trained_model_f_and_processed_data(
    data: Any,
    split_index: int,
    model_f: Any,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train model 'f' and prepare datasets for training/testing model 'L'.

    Args:
    - data (Any): UCI dataset object.
    - split_index (int): Index indicating what cross-validation split to use.
    - model_f (Any): An instance of a machine learning model that supports the 'fit' and 'predict' methods.

    Returns:
    - Tuple containing the trained model 'f' and the split that will be used for training/testing model 'L'
    """

    # Get cross-validation split
    X_train, Y_train, X_test, Y_test = data.get_split(split_index) # built-in function in the uci_datasets package
    Y_train, Y_test = np.squeeze(Y_train), np.squeeze(Y_test) # convert the target to a 1d array

    # Further split training data
    X_train_f, X_train_L, Y_train_f, Y_train_L = train_test_split(
        X_train, Y_train, test_size=DATA_SPLIT_RATIO, random_state=SEED
    )
    
    # Train model 'f'
    model_f.fit(X_train_f, Y_train_f)

    return model_f, X_train_L, X_test, Y_train_L, Y_test


def get_model_f_output_and_compute_losses(
    model_f_trained: Any, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    Y_train: np.ndarray, 
    Y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtain predictions from the trained model 'f' and compute losses to train/test model 'L'.

    Args:
    - model_f_trained (Any): The trained machine learning model.
    - X_train (np.ndarray): Training features dataset.
    - X_test (np.ndarray): Testing features dataset.
    - Y_train (np.ndarray): Actual training outputs.
    - Y_test (np.ndarray): Actual testing outputs.

    Returns:
    - Tuple containing predictions yielded by model 'f' and computed losses to train/test model 'L':
      output_pred_train, output_pred_test, target_train, target_test.
    """
    # Obtain predictions yielded by trained model 'f'
    output_pred_train = model_f_trained.predict(X_train)
    output_pred_test = model_f_trained.predict(X_test)

    # Obtain targets (losses) to train/test model 'L'
    target_train = (Y_train - output_pred_train) ** 2
    target_test = (Y_test - output_pred_test) ** 2

    return output_pred_train, output_pred_test, target_train, target_test
