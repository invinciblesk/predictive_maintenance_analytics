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


def plot_operational_parameters(pred_df_cleaned, columns_to_bootstrap, img_dir):
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))  
    axes = axes.flatten()  # For easy iteration

    for i, column in enumerate(columns_to_bootstrap):
        sns.barplot(x='Failure type', y=column, hue='Failure type', data=pred_df_cleaned, ax=axes[i], palette='Set2', errorbar='sd', legend=False)
        axes[i].set_title(f'Mean {column} by Failure type')
        axes[i].tick_params(axis='x', rotation=45)
        
        for failure_type in pred_df_cleaned['Failure type'].unique():
            subset = pred_df_cleaned[pred_df_cleaned['Failure type'] == failure_type]
            mean = subset[column].mean()
            sd = subset[column].std()
            
            axes[i].annotate('SD', xy=(failure_type, mean + sd), xytext=(0, 10),
                             textcoords='offset points', ha='center', va='bottom',
                             arrowprops=dict(facecolor='black', shrink=0.05))

    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'operational_parameters_mean_dist.png'))
    plt.close()

def plot_mean_parameters_by_failure_type(pred_df_cleaned, columns_to_bootstrap, img_dir):
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_bootstrap):
        sns.barplot(x='Failure type', y=column, hue='Failure type', data=pred_df_cleaned, ax=axes[i], palette='Set2', errorbar='sd', legend=False)
        axes[i].set_title(f'Mean {column} by Failure type')
        axes[i].tick_params(axis='x', rotation=45)

    fig.delaxes(axes[-1])  # Remove the last subplot as before
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'mean_parameters_by_failure_type.png'))
    plt.close()


def plot_failure_distributions(filtered_df, columns_to_plot, custom_palette, img_dir):
    # Setup for subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()

    # Iterate through each operational parameter to create histograms
    for i, column in enumerate(columns_to_plot):
        sns.histplot(data=filtered_df, x=column, hue='Failure type', multiple="stack", palette=custom_palette, ax=axes[i], bins=20)
        axes[i].set_title(f'Distribution of {column} by Failure State (Excluding No Failure)')

    # Remove the last subplot if the number of columns is odd
    if len(columns_to_plot) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'failure_distributions.png'))
    plt.close()

def plot_failure_distributions_2(filtered_df, columns_to_plot, custom_palette, img_dir):
    # Filter out 'No Failure' data
    filtered_df = filtered_df[filtered_df['Failure type'] != 'No Failure']

    # Setup for subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()

    # Iterate through each operational parameter to create histograms
    for i, column in enumerate(columns_to_plot):
        sns.histplot(data=filtered_df, x=column, hue='Failure type', multiple="stack", palette=custom_palette, ax=axes[i], bins=20)
        axes[i].set_title(f'Distribution of {column} by Failure State (Excluding No Failure)')

    # Remove the last subplot if the number of columns is odd
    if len(columns_to_plot) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'failure_distributions.png'))
    plt.close()



def plot_failure_type_counts(filtered_df, img_dir):
    # Group by 'Failure type' and count occurrences
    failure_counts = filtered_df.groupby('Failure type').size().sort_values(ascending=False).reset_index(name='count')
    
    # Create a figure and a single subplot
    plt.figure(figsize=(10, 8))
    
    # Create a horizontal bar plot without specifying palette
    sns.barplot(data=failure_counts, x='count', y='Failure type')
    
    # Set the title and labels
    plt.title('Count of Each Failure Type')
    plt.xlabel('Count')
    plt.ylabel('Failure Type')
    
    # Save the plot
    plt.savefig(os.path.join(img_dir, 'failure_counts.png'))
    plt.close()