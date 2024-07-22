import argparse
import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from data_processing import load_and_clean_data
from visualization import plot_bootstrap_means, plot_boxplots, plot_failure_distribution, plot_mean_by_failure_type
from regression_testing import perform_regression, plot_and_save_precision_recall_curve, save_results_to_csv

# Setup argparse
parser = argparse.ArgumentParser(description='Run predictive maintenance analytics.')
parser.add_argument('--data_path', type=str, help='Path to the dataset', required=True)
args = parser.parse_args()

# Load and clean the data
pred_df_cleaned_celcius = load_and_clean_data(args.data_path)
columns_to_plot = ['Air temperature [C]', 'Process temperature [C]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Ensure the img directory exists
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Navigate up to the root directory
img_dir = os.path.join(root_dir, 'img')
os.makedirs(img_dir, exist_ok=True)

# Visualizations
plot_bootstrap_means(pred_df_cleaned_celcius, img_dir)
plot_boxplots(pred_df_cleaned_celcius, img_dir)
plot_failure_distribution(pred_df_cleaned_celcius, img_dir)
plot_mean_by_failure_type(pred_df_cleaned_celcius, columns_to_plot, img_dir)

# Prepare the data for regression testing
X = pred_df_cleaned_celcius.drop(['Product ID', 'Type', 'Machine failure', 'Failure type'], axis=1)
y = pred_df_cleaned_celcius['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = [
    ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42))
]


# Ensure the reports directory exists in the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Navigate up to the root directory
reports_dir = os.path.join(root_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)

# CSV file path
csv_file_path = os.path.join(reports_dir, 'model_evaluation_reports.csv')

# Check if the file exists to decide on writing headers
file_exists = os.path.isfile(csv_file_path)

with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers if the file is being created
    if not file_exists:
        writer.writerow(['Model Name', 'Accuracy', 'Recall', 'F1 Score', 'Classification Report'])
    
    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Prepare the classification report to write into a single cell
        classification_report_str = classification_report(y_test, y_pred).replace('\n', '\\n')
        
        # Write model evaluation metrics
        writer.writerow([model_name, f"{accuracy:.2f}", f"{recall:.2f}", f"{f1:.2f}", classification_report_str])

        print(f"Results for {model_name} saved to {csv_file_path}")

# Perform regression testing
perform_regression(pred_df_cleaned_celcius, reports_dir)
