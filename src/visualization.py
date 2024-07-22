import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import numpy as np

def plot_bootstrap_means(bootstrap_means, img_dir):
    """Plot and save histograms of bootstrap means for each column."""
    for column, means in bootstrap_means.items():
        plt.figure()
        plt.hist(means, bins=50, alpha=0.75)
        plt.title(f'Bootstrap Means for {column}')
        plt.xlabel('Mean Value')
        plt.ylabel('Frequency')
        plt.savefig(f'{img_dir}/{column}_bootstrap_means.png')  # Save the plot to the specified directory
        plt.close()

def plot_boxplots(pred_df_cleaned, columns_to_bootstrap, img_dir):
    """Create and save boxplots for selected columns against machine failure."""
    columns_to_bootstrap = ['Air temperature [C]', 'Process temperature [C]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_bootstrap):
        sns.boxplot(x='Machine failure', y=column, data=pred_df_cleaned, ax=axes[i])
        axes[i].set_title(f'{column} vs Machine failure')

    fig.delaxes(axes[-1])  # Remove the unused subplot
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'boxplots.png'))
    plt.close()

def plot_failure_distribution(pred_df_cleaned, img_dir):
    """Plot and save the distribution of failure types as a histogram."""
    fig = px.histogram(pred_df_cleaned, x='Failure type', text_auto=True, histnorm='percent')
    fig.update_layout(yaxis_title='Percentage (%)')
    fig.write_image(os.path.join(img_dir, 'failure_distribution.png'))

def plot_mean_by_failure_type(pred_df_cleaned, columns_to_plot, img_dir):
    """Create and save bar plots of mean values for each failure type."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        sns.barplot(x='Failure type', y=column, hue='Failure type', data=pred_df_cleaned, ax=axes[i], palette='Set2', errorbar='sd', legend=False)
        axes[i].set_title(f'Mean {column} by Failure type')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'mean_by_failure_type.png'))
    plt.close()

def plot_operational_parameters(pred_df_cleaned, columns_to_bootstrap, img_dir):
    """Create and save bar plots of operational parameters by failure type, with standard deviation annotations."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_bootstrap):
        sns.barplot(x='Failure type', y=column, hue='Failure type', data=pred_df_cleaned, ax=axes[i], palette='Set2', errorbar='sd', legend=False)
        axes[i].set_title(f'Mean {column} by Failure type')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Annotate each bar with the standard deviation
        for failure_type in pred_df_cleaned['Failure type'].unique():
            subset = pred_df_cleaned[pred_df_cleaned['Failure type'] == failure_type]
            mean = subset[column].mean()
            sd = subset[column].std()
            axes[i].annotate('SD', xy=(failure_type, mean + sd), xytext=(0, 10),
                             textcoords='offset points', ha='center', va='bottom',
                             arrowprops=dict(facecolor='black', shrink=0.05))

    fig.delaxes(axes[-1])  # Remove the unused subplot
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'operational_parameters_mean_dist.png'))
    plt.close()

def plot_mean_parameters_by_failure_type(pred_df_cleaned, columns_to_bootstrap, img_dir):
    """Create and save bar plots of mean parameters for each failure type."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_bootstrap):
        sns.barplot(x='Failure type', y=column, hue='Failure type', data=pred_df_cleaned, ax=axes[i], palette='Set2', errorbar='sd', legend=False)
        axes[i].set_title(f'Mean {column} by Failure type')
        axes[i].tick_params(axis='x', rotation=45)

    fig.delaxes(axes[-1])  # Remove the unused subplot
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'mean_parameters_by_failure_type.png'))
    plt.close()

def plot_failure_distributions(filtered_df, columns_to_plot, custom_palette, img_dir):
    """Plot and save the distribution of each column by failure type, excluding 'No Failure'."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        sns.histplot(data=filtered_df, x=column, hue='Failure type', multiple="stack", palette=custom_palette, ax=axes[i], bins=20)
        axes[i].set_title(f'Distribution of {column} by Failure State (Excluding No Failure)')

    if len(columns_to_plot) % 2 != 0:
        fig.delaxes(axes[-1])  # Remove the unused subplot if the number of columns is odd

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'failure_distributions.png'))
    plt.close()

def plot_failure_distributions_2(filtered_df, columns_to_plot, custom_palette, img_dir):
    """Plot and save the distribution of each column by failure type, with 'No Failure' filtered out."""
    filtered_df = filtered_df[filtered_df['Failure type'] != 'No Failure']

    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        sns.histplot(data=filtered_df, x=column, hue='Failure type', multiple="stack", palette=custom_palette, ax=axes[i], bins=20)
        axes[i].set_title(f'Distribution of {column} by Failure State (Excluding No Failure)')

    if len(columns_to_plot) % 2 != 0:
        fig.delaxes(axes[-1])  # Remove the unused subplot if the number of columns is odd

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'failure_distributions.png'))
    plt.close()

def plot_failure_type_counts(filtered_df, img_dir):
    """Plot and save a horizontal bar chart of the count of each failure type."""
    failure_counts = filtered_df.groupby('Failure type').size().sort_values(ascending=False).reset_index(name='count')
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=failure_counts, x='count', y='Failure type')
    plt.title('Count of Each Failure Type')
    plt.xlabel('Count')
    plt.ylabel('Failure Type')
    plt.savefig(os.path.join(img_dir, 'failure_counts.png'))
    plt.close()
