import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import numpy as np



def plot_bootstrap_means(bootstrap_means, img_dir):
    """Plot the bootstrap means for each column."""
    
    for column, means in bootstrap_means.items():
        plt.figure()
        plt.hist(means, bins=50, alpha=0.75)
        plt.title(f'Bootstrap Means for {column}')
        plt.xlabel('Mean Value')
        plt.ylabel('Frequency')
        
        # Save the plot to the specified directory
        plt.savefig(f'{img_dir}/{column}_bootstrap_means.png')
        plt.close()

def plot_boxplots(pred_df_cleaned, columns_to_bootstrap, img_dir):
    columns_to_bootstrap = ['Air temperature [C]', 'Process temperature [C]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_bootstrap):
        sns.boxplot(x='Machine failure', y=column, data=pred_df_cleaned, ax=axes[i])
        axes[i].set_title(f'{column} vs Machine failure')

    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'boxplots.png'))
    plt.close()

def plot_failure_distribution(pred_df_cleaned, img_dir):
    fig = px.histogram(pred_df_cleaned, x='Failure type', text_auto=True, histnorm='percent')
    fig.update_layout(yaxis_title='Percentage (%)')
    
    fig.write_image(os.path.join(img_dir, 'failure_distribution.png'))

def plot_mean_by_failure_type(pred_df_cleaned, columns_to_plot, img_dir):
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        sns.barplot(x='Failure type', y=column, hue='Failure type', data=pred_df_cleaned, ax=axes[i], palette='Set2', errorbar='sd', legend=False)
        axes[i].set_title(f'Mean {column} by Failure type')
        axes[i].tick_params(axis='x', rotation=45)

    
    plt.tight_layout()
    
    plt.savefig(os.path.join(img_dir, 'mean_by_failure_type.png'))
    plt.close()


# def plot_and_save_roc_curve(fpr, tpr, model_name, img_dir):
#     # Plot ROC curve
#     plt.figure()
#     plt.plot(fpr, tpr, label='ROC curve')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve for {model_name}')
#     plt.legend(loc="lower right")
    
#     # Ensure the img_dir is correctly referenced and exists
#     if not os.path.exists(img_dir):
#         os.makedirs(img_dir)
    
#     # Save the figure in the 'img' directory
#     plt.savefig(os.path.join(img_dir, f'{model_name}_roc_curve.png'))
#     plt.close()