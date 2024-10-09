import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from uci_datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

from constants import SEED
from evaluate_models import create_df_with_results, evaluate_one_instance, round_value_in_table


# Define parameters for different models
random_forest_params = {
    'criterion': 'squared_error',
    'max_depth': None,
    'random_state': SEED,
}

mlp_params = {
    'hidden_layer_sizes': (64,),
    'activation': 'relu',
    'solver': 'adam',
    'batch_size': 256,
    'learning_rate_init': 0.0005,
    'max_iter': 800,
    'random_state': SEED
}

mlp_params2 = {
    'hidden_layer_sizes': (64, 64),
    'activation': 'relu',
    'solver': 'adam',
    'batch_size': 256,
    'learning_rate_init': 0.0005,
    'max_iter': 800,
    'random_state': SEED
}


# Define constants and parameters for the evaluation
cost_list = [0.2, 0.5, 1, 2]
metrics_list = ['RwR_loss', 'human_rate', 'mae', 'rmse', 'bias', 'wape']
data_names = ['concrete', 'wine', 'airfoil', 'energy', 'housing', 'solar', 'forest', 'parkinsons']


# Define model pairs for comparison (each pair is [model_f, model_L])
model_pairs = {
    'LR+LR': [Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())]), Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())])],
    'LR+RF': [Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())]), RandomForestRegressor(**random_forest_params)],
    'LR+MLP': [Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())]), Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params))])],
    'LR+MLP2': [Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())]), Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params2))])],
    'RF+LR': [RandomForestRegressor(**random_forest_params), Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())])],
    'RF+RF': [RandomForestRegressor(**random_forest_params), RandomForestRegressor(**random_forest_params)],
    'RF+MLP': [RandomForestRegressor(**random_forest_params), Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params))])],
    'RF+MLP2': [RandomForestRegressor(**random_forest_params), Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params2))])],
    'MLP+LR': [Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params))]), Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())])],
    'MLP+RF': [Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params))]), RandomForestRegressor(**random_forest_params)],
    'MLP+MLP': [Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params))]), Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params))])],
    'MLP+MLP2': [Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params))]), Pipeline([('scaler', StandardScaler()), ('mlp2', MLPRegressor(**mlp_params2))])],
    'MLP2+LR': [Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params2))]), Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())])],
    'MLP2+RF': [Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params2))]), RandomForestRegressor(**random_forest_params)],
    'MLP2+MLP': [Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params2))]), Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params))])],
    'MLP2+MLP2': [Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params2))]), Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**mlp_params2))])],
}


# Iterate over each dataset name in data_names
for data_name in data_names:
    results_all_pairs = dict()

    # Iterate over each model pair in model_pairs
    for model_name, (model_f, model_L) in model_pairs.items():
        results_one_pair = dict()

        # Iterate over each cost value in cost_list
        for metrics in metrics_list:
            results_one_pair[metrics] = []

        for cost in cost_list:
            print(f"==== Dataset: {data_name}, model_name: {model_name}, cost: {cost}")
            
            data = Dataset(data_name)
            results_one_cost = evaluate_one_instance(data, model_f, model_L, cost)
            for metrics in metrics_list:
                results_one_pair[metrics].append(results_one_cost[metrics])

        results_all_pairs[model_name] = results_one_pair

    # Create a DataFrame from the results and save to csv files
    df_output = create_df_with_results(results_all_pairs, cost_list)
    for metrics in metrics_list:
        df_output[metrics].to_csv(f'output_folder/{metrics}_{data_name}.csv')


# Compile results for each metric across all datasets
for metrics in metrics_list:
    df_one_metrics_all_datasets_list = []

    # Read and process each dataset's csv file for the current metric
    for data_name in data_names:
        df_temp = pd.read_csv(f'output_folder/{metrics}_{data_name}.csv')
        df_temp = df_temp.reindex(sorted(df_temp.columns), axis=1)
        df_temp.insert(loc=0, column='Dataset', value=data_name)
        df_one_metrics_all_datasets_list.append(df_temp)

    df_one_metrics_all_datasets = pd.concat(df_one_metrics_all_datasets_list, ignore_index=True)

    # Round the values in the DataFrame
    for column in df_one_metrics_all_datasets.columns[2:]:
        df_one_metrics_all_datasets[column] = df_one_metrics_all_datasets[column].apply(round_value_in_table)

    # Drop the 'Cost' column and remove duplicates for specific metrics
    if metrics not in ['RwR_loss', 'human_rate']:
        df_one_metrics_all_datasets = df_one_metrics_all_datasets.drop('Cost', axis=1)
        df_one_metrics_all_datasets.drop_duplicates(inplace=True)

    df_one_metrics_all_datasets.to_csv(f'output_folder/{metrics}_all.csv')
    