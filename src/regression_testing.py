import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import csv

def plot_and_save_roc_curve(fpr, tpr, model_name, reports_dir):
    """
    Plot and save the ROC curve for a model.

    Parameters:
    - fpr: array-like, false positive rates.
    - tpr: array-like, true positive rates.
    - model_name: str, name of the model.
    - reports_dir: str, directory to save the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line for random performance
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Construct the full path for the file to be saved
    file_path = os.path.join(reports_dir, f'{model_name}_roc_curve.png')
    plt.savefig(file_path)
    plt.close()

def plot_and_save_precision_recall_curve(precision, recall, model_name, reports_dir):
    """
    Plot and save the Precision-Recall curve for a model.

    Parameters:
    - precision: array-like, precision values.
    - recall: array-like, recall values.
    - model_name: str, name of the model.
    - reports_dir: str, directory to save the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    # Construct the full path for the file to be saved
    file_path = os.path.join(reports_dir, f'{model_name}_precision_recall_curve.png')
    plt.savefig(file_path)
    plt.close()

def save_results_to_csv(results, reports_dir):
    """
    Save the evaluation results of models to a CSV file.

    Parameters:
    - results: list of dicts, evaluation metrics for each model.
    - reports_dir: str, directory to save the CSV file.
    """
    csv_file_path = os.path.join(reports_dir, 'model_evaluation_reports.csv')
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file does not exist
        if not file_exists:
            writer.writerow(['Model', 'Accuracy', 'Confusion Matrix', 'Recall', 'F1 Score', 'Classification Report'])

        # Write each model's results
        for result in results:
            writer.writerow([
                result['Model'],
                f"{result['Accuracy']:.2f}",
                result['Confusion Matrix'],
                f"{result['Recall']:.2f}",
                f"{result['F1 Score']:.2f}",
                str(result['Classification Report'])  # Convert dict to string
            ])

def perform_regression(df, reports_dir):
    """
    Perform regression analysis using multiple models and save the evaluation results.

    Parameters:
    - df: DataFrame, input data with features and target.
    - reports_dir: str, directory to save evaluation plots and CSV.
    """
    # Split data into features and target
    X = df.drop(['Type', 'Machine failure', 'Failure type'], axis=1)
    y = df['Machine failure']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    logistic_model = LogisticRegression(random_state=42, max_iter=1000)
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    random_forest_model = RandomForestClassifier(random_state=42)

    models = [logistic_model, decision_tree_model, random_forest_model]

    results = []

    # Train and evaluate each model
    for model in models:
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Confusion Matrix': confusion.tolist(),
            'Recall': recall,
            'F1 Score': f1,
            'Classification Report': classification_rep
        })

        # Calculate scores for ROC and Precision-Recall curves
        y_score = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])

        # Save ROC and Precision-Recall curves
        plot_and_save_roc_curve(fpr, tpr, model_name, reports_dir)
        plot_and_save_precision_recall_curve(precision, recall, model_name, reports_dir)

    # Save results to CSV
    save_results_to_csv(results, reports_dir)
