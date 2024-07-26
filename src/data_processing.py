import pandas as pd
import numpy as np


def load_and_clean_data(filepath):
    """
    Load the dataset, clean it by removing unnecessary columns and combining failure types into a single column.

    Parameters:
    - filepath: str, path to the CSV file.

    Returns:
    - pred_df_cleaned_celcius: DataFrame, cleaned dataset with temperatures converted to Celsius.
    - columns_to_bootstrap: list, columns for which bootstrapping will be performed.
    """
    # Load the dataset
    pred_df = pd.read_csv(filepath)

    # Drop unnecessary columns
    columns_to_drop = ['UDI', 'Product ID']
    pred_df_cleaned = pred_df.drop(columns=columns_to_drop, errors='ignore')

    # Combine failures into a single column
    def combine_failures(row):
        failures = []
        if row['TWF']: failures.append('TWF')  # Tool Wear Failure
        if row['HDF']: failures.append('HDF')  # Heat Dissipation Failure
        if row['PWF']: failures.append('PWF')  # Power Failure
        if row['OSF']: failures.append('OSF')  # Overstrain Failure
        if row['RNF']: failures.append('RNF')  # Random Failure
        return ', '.join(failures) if failures else 'No Failure'

    pred_df_cleaned['Failure type'] = pred_df_cleaned.apply(combine_failures, axis=1)
    pred_df_cleaned.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)

    # Columns to bootstrap
    columns_to_bootstrap = ['Air temperature [C]', 'Process temperature [C]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    # Convert temperatures from Kelvin to Celsius
    pred_df_cleaned_celcius = pred_df_cleaned.copy()
    if 'Air temperature [K]' in pred_df_cleaned_celcius.columns:
        pred_df_cleaned_celcius['Air temperature [C]'] = pred_df_cleaned_celcius['Air temperature [K]'] - 273.15
        pred_df_cleaned_celcius.drop('Air temperature [K]', axis=1, inplace=True)

    if 'Process temperature [K]' in pred_df_cleaned_celcius.columns:
        pred_df_cleaned_celcius['Process temperature [C]'] = pred_df_cleaned_celcius['Process temperature [K]'] - 273.15
        pred_df_cleaned_celcius.drop('Process temperature [K]', axis=1, inplace=True)

    # Define the new column order for pred_df_cleaned_celcius, ensuring it matches the updated DataFrame
    new_column_order = ['Type', 'Air temperature [C]', 'Process temperature [C]',
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                        'Machine failure', 'Failure type']
    pred_df_cleaned_celcius = pred_df_cleaned_celcius.reindex(columns=new_column_order)

    return pred_df_cleaned_celcius, columns_to_bootstrap


def bootstrap_resample(data, columns_to_bootstrap, n_bootstrap=1000):
    """
    Perform bootstrapping on the specified columns of the data and return the sample means for each column.

    Parameters:
    - data: DataFrame, input data for bootstrapping.
    - columns_to_bootstrap: list, columns on which to perform bootstrapping.
    - n_bootstrap: int, number of bootstrap samples to generate.

    Returns:
    - bootstrap_means: dict, bootstrap means for each column.
    """
    # Ensure all specified columns exist in the data
    assert all(column in data.columns for column in columns_to_bootstrap), "Some specified columns do not exist in the data."

    bootstrap_means = {column: np.zeros(n_bootstrap) for column in columns_to_bootstrap}
    n = len(data)

    # Perform bootstrapping for each specified column
    for column in columns_to_bootstrap:
        for i in range(n_bootstrap):
            sample = np.random.choice(data[column], size=n, replace=True)
            bootstrap_means[column][i] = np.mean(sample)

    return bootstrap_means
