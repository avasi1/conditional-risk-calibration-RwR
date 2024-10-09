import numpy as np
import pandas as pd
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Any, Dict, List, Tuple, Union

from constants import NUMBER_OF_FOLDS
from processing import get_model_f_output_and_compute_losses, get_trained_model_f_and_processed_data


def evaluate_one_instance(
    data: Any,
    model_f: Any,
    model_L: Any,
    cost: float,
) -> Dict[str, str]:
    """
    Evaluate the performance of model 'L' using cross-validation splits and compute various metrics.

    Args:
    - data (Any): The dataset object containing the input features and target labels.
    - model_f (Any): An instance of a machine learning model that supports 'fit' and 'predict' methods for training and testing.
    - model_L (Any): A model instance (calibrator) used for estimating errors based on the outputs from model 'f'.
    - cost (float): The rejection cost.

    Returns:
    - Dict[str, str]: A dictionary containing the mean and standard deviation of several evaluation metrics.
    """
    RwR_loss_list, human_rate_list, mae_list, rmse_list, bias_list, wape_list = [], [], [], [], [], []
    for split_index in range(NUMBER_OF_FOLDS):

        # Train model 'f' and prepare datasets for training/testing model 'L'
        model_f_trained, X_train, X_test, Y_train, Y_test = get_trained_model_f_and_processed_data(
            data=data,
            split_index=split_index,
            model_f=model_f,
        )

        # Obtain predictions from the trained model 'f' and compute errors to train/test model 'L'
        output_pred_train, output_pred_test, target_train, target_test = get_model_f_output_and_compute_losses(
            model_f_trained=model_f_trained,
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test
        )

        # Train model 'L' and obtain estimates of the errors for the test data
        model_L_trained = model_L.fit(X_train, target_train)
        target_pred_test = model_L_trained.predict(X_test)

        # Compute human rate and RwR loss
        RwR_loss, human_rate = compute_RwR_loss_and_human_rate(
            model_f_trained,
            X_test, 
            Y_test, 
            target_pred_test,
            cost,
        )

        # Evaluate the performance of the calibrator
        mae, rmse, bias, wape = compute_regression_metrics(target_test, target_pred_test)

        RwR_loss_list.append(RwR_loss)
        human_rate_list.append(human_rate)
        mae_list.append(mae)
        rmse_list.append(rmse)
        bias_list.append(bias)
        wape_list.append(wape)

    RwR_loss_mean, RwR_loss_std = np.round(np.mean(RwR_loss_list), 3), np.round(np.std(RwR_loss_list), 3)
    human_rate_mean, human_rate_std = np.round(np.mean(human_rate_list), 3), np.round(np.std(human_rate_list), 3)
    mae_mean, mae_std = np.round(np.mean(mae_list), 3), np.round(np.std(mae_list), 3)
    rmse_mean, rmse_std = np.round(np.mean(rmse_list), 3), np.round(np.std(rmse_list), 3)
    bias_mean, bias_std = np.round(np.mean(bias_list), 3), np.round(np.std(bias_list), 3)
    wape_mean, wape_std = np.round(np.mean(wape_list), 3), np.round(np.std(wape_list), 3)
    
    output = {
        'RwR_loss': f'{RwR_loss_mean} ({RwR_loss_std})',
        'human_rate': f'{human_rate_mean} ({human_rate_std})',
        'mae': f'{mae_mean} ({mae_std})',
        'rmse': f'{rmse_mean} ({rmse_std})',
        'bias': f'{bias_mean} ({bias_std})',
        'wape': f'{wape_mean} ({wape_std})',
    }
    return output


def compute_RwR_loss_and_human_rate(
    model_f_trained: Any,
    X_test: np.ndarray, 
    Y_test: np.ndarray, 
    target_pred_test: np.ndarray,
    cost: float,
) -> Tuple[float, float]:
    """
    Compute human rate and RwR loss for a given model.

    Args:
    - model_f_trained (Any): The trained machine learning model.
    - X_test (np.ndarray): Testing features dataset.
    - Y_test (np.ndarray): Actual testing outputs.
    - target_pred_test (np.ndarray): Predicted losses.
    - cost (float): Rejection cost.

    Returns:
    - Tuple containing the RwR loss and the human rate
    """
    flag = (target_pred_test < cost)

    if np.sum(flag) > 0:
        accept_rate = np.sum(flag) / X_test.shape[0]

        # Use only the accepted instances for evaluation
        X_test_m = X_test[flag]
        Y_test_m = Y_test[flag]
        Y_pred_m = model_f_trained.predict(X_test_m)

        # Total RwR loss includes the cost of rejecting instances
        RwR_loss = (np.sum((Y_pred_m - Y_test_m) ** 2) + cost * (X_test.shape[0] - np.sum(flag))) / X_test.shape[0]
        return round(RwR_loss, 3), round(1.0 - accept_rate, 3)
    else:
        return cost, 1.0


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray], 
    y_pred: Union[pd.Series, np.ndarray]
) -> Tuple[float, float, float, float]:
    """
    Compute and round to two digits key performance metrics given targets and predictions.

    Args:
    - y_true (Union[pd.Series, np.ndarray]): The actual target values.
    - y_pred (Union[pd.Series, np.ndarray]): The predicted target values from the model.

    Returns:
    - Tuple containing the following four metrics:
      - mae (float): Mean Absolute Error between y_true and y_pred.
      - rmse (float): Root Mean Square Error between y_true and y_pred.
      - bias (float): The difference between the mean of y_pred and the mean of y_true.
      - wape (float): Weighted Absolute Percentage Error, expressed as a percentage.
    """
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
    bias = round(np.mean(y_pred) - np.mean(y_true), 3)
    wape = round(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100, 3)
    return mae, rmse, bias, wape


def create_df_with_results(
    results_all_pairs: Dict[str, Dict[str, List[float]]],
    cost_list: List[float],
) -> pd.DataFrame:
    """
    Create a DataFrame that organizes model evaluation results by metrics and models.

    Args:
    - results_all_pairs (Dict[str, Dict[str, List[float]]]): A dictionary where keys are model names and values are 
      dictionaries mapping metric names to lists of metric values for different rejection costs.
    - cost_list (List[float]): A list of cost values used as the index for the DataFrame.

    Returns:
    - pd.DataFrame: The results DataFrame
    """
    df_dict = {}

    for model, metrics in results_all_pairs.items():
        for metric, values in metrics.items():
            df_dict[(model, metric)] = values

    # Create a DataFrame from the dictionary with the given cost values as the index
    df = pd.DataFrame(df_dict, index=cost_list)
    df.index.name = 'Cost'

    # Swap the levels of the MultiIndex so that metrics are the main columns and models are the subcolumns
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Model", "Metric"])
    df = df.swaplevel(axis=1).sort_index(axis=1)
    return df


def round_value_in_table(entry: str) -> str:
    """
    Round the mean and standard deviation values in a string formatted as 'mean (std)'.

    Args:
    - entry (str): A string containing a numerical value (mean) and its uncertainty (standard deviation) 
      in the format 'value (uncertainty)'.

    Returns:
    - str: A string with the mean and uncertainty rounded to two decimal places.
    """
    # Use regex to capture the mean value and the std in parentheses
    match = re.match(r'([0-9.]+)\s*\((.+)\)', entry)
    if match:
        value = float(match.group(1))
        uncertainty = float(match.group(2))
        
        rounded_value = round(value, 2)
        rounded_uncertainty = round(uncertainty, 2)
        
        return f'{rounded_value:.2f} ({rounded_uncertainty:.2f})'
    else:
        # If the format is not correct, just return the original entry
        return entry
