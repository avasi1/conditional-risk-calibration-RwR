import matplotlib.pyplot as plt

from constants import (
    ESTIMATED_ERROR_COL,
    PREDICTIONS_COL,
    TARGET_COL,
)


def plot_conditional_loss(df, title=''):
    """
    Plot the target values, predictions, and estimated errors from a DataFrame.

    Args:
    - df (DataFrame): The DataFrame containing target, predictions, and estimated errors data.
    - title (str, optional): The title of the plot.
    """
    _, ax = plt.subplots(figsize=(16, 4))

    # Plot target and predictions
    ax.plot(df[TARGET_COL], label=TARGET_COL, color='blue', marker='s', markersize=4)
    ax.plot(df[PREDICTIONS_COL], label=PREDICTIONS_COL, color='darkorange', marker='o', markersize=4)

    # Error bars for estimated error
    ax.errorbar(
        df.index, df[TARGET_COL], 
        yerr=df[ESTIMATED_ERROR_COL], fmt=' ', color='gray', 
        elinewidth=1.5, capsize=2.5, label='Loss estimated by calibrator'
    )

    ax.set_xlabel('Index', size=11)
    ax.set_ylabel('Value', size=11)
    ax.set_title(title, size=13)
    
    ax.legend()
    plt.show()
