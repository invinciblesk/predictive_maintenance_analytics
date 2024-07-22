import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import make_pipeline
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import ttest_ind, mannwhitneyu


def load_and_clean_data(file_path):
    pred_df = pd.read_csv(file_path)

    if 'UDI' in pred_df.columns:
        pred_df.drop('UDI', axis=1, inplace=True)

    def combine_failures(row):
        failures = []
        if row['TWF']: failures.append('TWF')
        if row['HDF']: failures.append('HDF')
        if row['PWF']: failures.append('PWF')
        if row['OSF']: failures.append('OSF')
        if row['RNF']: failures.append('RNF')
        return ', '.join(failures) if failures else 'No Failure'

    pred_df['Failure type'] = pred_df.apply(combine_failures, axis=1)
    pred_df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)

    # Convert temperatures from Kelvin to Celsius
    pred_df[['Air temperature [C]', 'Process temperature [C]']] = pred_df[['Air temperature [K]', 'Process temperature [K]']] - 273.15

    # Define the new column order, replacing Kelvin columns with Celsius columns
    new_column_order = ['Product ID', 'Type', 'Air temperature [C]', 'Process temperature [C]',
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                        'Machine failure', 'Failure type']
    pred_df = pred_df.reindex(columns=new_column_order)

    return pred_df


    def bootstrap_resample(data, n_bootstrap=1000):
        """Perform bootstrapping on the data and return the sample means."""
    
    bootstrap_means = np.zeros(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)
    return bootstrap_means

    # Columns to bootstrap
    columns_to_bootstrap = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
