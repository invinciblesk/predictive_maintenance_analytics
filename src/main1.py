import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, label_binarize
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind, mannwhitneyu
import plotly.graph_objects as go


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report, precision_score, precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline


pred_df = pd.read_csv('../data/ai4i2020.csv')


if 'UDI' in pred_df.columns:
    pred_df.drop('UDI', axis=1, inplace=True)
pred_df.head(80)

# Combine failure types into a single column
pred_df_cleaned = pd.DataFrame(pred_df)
def combine_failures(row):
    failures = []
    if row['TWF']: failures.append('TWF')
    if row['HDF']: failures.append('HDF')
    if row['PWF']: failures.append('PWF')
    if row['OSF']: failures.append('OSF')
    if row['RNF']: failures.append('RNF')
    return ', '.join(failures) if failures else 'No Failure'

# Apply the function to each row
pred_df_cleaned['Failure type'] = pred_df.apply(combine_failures, axis=1)

# Drop the original failure type columns
pred_df_cleaned.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)

pred_df_cleaned.head(80)

# Keep a copy of the cleaned dataframe
unbootstrapped_pred_df_cleaned = pred_df_cleaned.copy()
unbootstrapped_pred_df_cleaned.head()

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

# Initialize a dictionary to hold bootstrapped means for each column
bootstrap_results = {}

# Perform bootstrapping for each column
for column in columns_to_bootstrap:
    bootstrap_means = bootstrap_resample(pred_df_cleaned[column])
    bootstrap_results[column] = bootstrap_means



fig = make_subplots(rows=len(columns_to_bootstrap)//2 + len(columns_to_bootstrap)%2, cols=2, subplot_titles=[f'Bootstrapped Means for {column}' for column in columns_to_bootstrap])

for i, column in enumerate(columns_to_bootstrap):
    overall_mean = pred_df_cleaned[column].mean()  # Calculate the overall mean for the column
    # Determine the row and column position for the current subplot
    row = i // 2 + 1
    col = i % 2 + 1
    # Create histogram for the current column
    histogram = go.Histogram(x=bootstrap_results[column], name=column)
    fig.add_trace(histogram, row=row, col=col)
    # Add a vertical line for the overall mean
    fig.add_vline(x=overall_mean, line_color="red", line_dash="dash", annotation_text=f"Overall Mean: {overall_mean:.2f}", 
                  annotation_position="top right", row=row, col=col)

fig.update_layout(height=1000, width=1200, title_text="Bootstrapped Means Visualization", showlegend=False, margin=dict(l=20, r=20, t=100, b=20))

fig.show()


# Initialize the subplot
fig, axes = plt.subplots(3, 2, figsize=(20, 18)) 
axes = axes.flatten()  

# Iterate through each operational parameter
for i, column in enumerate(columns_to_bootstrap):
    sns.boxplot(x='Machine failure', y=column, data=pred_df_cleaned, ax=axes[i])
    axes[i].set_title(f'{column} vs Machine failure')
    

# Remove the last empty subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()


fig = px.histogram(pred_df_cleaned, x='Failure type', text_auto=True, histnorm='percent')
fig.update_layout(yaxis_title='Percentage (%)')
fig.update_layout(title_text='Distribution of Failure Types')
fig.show()


# Initialize the subplot
fig, axes = plt.subplots(3, 2, figsize=(20, 22))  
axes = axes.flatten()  # For easy iteration

# Iterate through each operational parameter to create bar plots
for i, column in enumerate(columns_to_bootstrap):
    
    # Create the bar plot
    sns.barplot(x='Failure type', y=column, hue='Failure type', data=pred_df_cleaned, ax=axes[i], palette='Set2', errorbar='sd', legend=False)
    axes[i].set_title(f'Mean {column} by Failure type')
    axes[i].tick_params(axis='x', rotation=45)
    
    # Calculate mean and standard deviation for annotations
    for failure_type in pred_df_cleaned['Failure type'].unique():
        subset = pred_df_cleaned[pred_df_cleaned['Failure type'] == failure_type]
        mean = subset[column].mean()
        sd = subset[column].std()
        
        # Annotate error bar per subplot
        axes[i].annotate('SD', xy=(failure_type, mean + sd), xytext=(0, 10),
                         textcoords='offset points', ha='center', va='bottom',
                         arrowprops=dict(facecolor='black', shrink=0.05))

# Remove the last empty subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()


# Create a copy of pred_df_cleaned to preserve the original data
pred_df_cleaned_celcius = pred_df_cleaned.copy()

columns_to_plot = ['Air temperature [C]', 'Process temperature [C]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Convert temperatures from Kelvin to Celsius
pred_df_cleaned_celcius[['Air temperature [C]', 'Process temperature [C]']] = pred_df_cleaned_celcius[['Air temperature [K]', 'Process temperature [K]']] - 273.15

# Define the new column order, replacing Kelvin columns with Celsius columns
new_column_order = ['Product ID', 'Type', 'Air temperature [C]', 'Process temperature [C]',
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                    'Machine failure', 'Failure type']

# Reorder/reindex the columns
pred_df_cleaned_celcius = pred_df_cleaned_celcius.reindex(columns=new_column_order)

pred_df_cleaned_celcius.head()





# Create a list of unique failure types for color mapping
failure_types = pred_df_cleaned['Failure type'].unique()
colors = px.colors.qualitative.Set2

# Create a subplot figure
fig = make_subplots(rows=3, cols=2, subplot_titles=[f'Mean {column} by Failure type' for column in columns_to_bootstrap])

# Flatten the subplot structure for easy iteration
axes = [(r, c) for r in range(1, 4) for c in range(1, 3)]

# Iterate through each operational parameter to create bar plots
for i, column in enumerate(columns_to_bootstrap):
    # Filter the DataFrame for the current column and calculate mean and standard deviation
    df_grouped = pred_df_cleaned.groupby('Failure type')[column].agg(['mean', 'std']).reset_index()
    
    # Create the bar plot for the current column
    for j, failure_type in enumerate(failure_types):
        # Find the mean and standard deviation for the current failure type
        mean = df_grouped.loc[df_grouped['Failure type'] == failure_type, 'mean'].values[0]
        std = df_grouped.loc[df_grouped['Failure type'] == failure_type, 'std'].values[0]
        
        # Add a bar for the current failure type
        fig.add_trace(go.Bar(x=[failure_type], y=[mean], name=failure_type, marker_color=colors[j % len(colors)],
                             error_y=dict(type='data', array=[std], visible=True)),
                      row=axes[i][0], col=axes[i][1])
    
    # Update the layout for the current subplot
    fig.update_xaxes(title_text="Failure Type", row=axes[i][0], col=axes[i][1])
    fig.update_yaxes(title_text=column, row=axes[i][0], col=axes[i][1])

# Update the layout for the entire figure
fig.update_layout(height=1000, width=1200, showlegend=False, title_text="Mean Operational Parameters by Failure Type")

# Show the figure
fig.show()


# Define a custom palette
base_palette = sns.color_palette('Paired')
custom_palette = {failure_type: color for failure_type, color in zip(pred_df_cleaned_celcius['Failure type'].unique(), base_palette)}

# Adjust subplot grid
n_rows = 3
n_cols = 2  
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))  

# Flatten axes for easy iteration
axes = axes.flatten()

# Iterate through each operational parameter to create stacked bar plots
for i, column in enumerate(columns_to_plot):
    ax = axes[i]  
    # Prepare data for stacked bar plot
    data = pred_df_cleaned_celcius.groupby(['Failure type', column]).size().unstack(fill_value=0)
    data_cum = data.cumsum(axis=1)
    
    # Plot bars
    for failure_type, color in custom_palette.items():
        widths = data.loc[failure_type, :].values
        starts = data_cum.loc[failure_type, :] - widths
        ax.barh(failure_type, widths, left=starts, height=0.5, label=failure_type, color=color)
    
    ax.set_title(f'Distribution of {column} by Failure State')
    ax.legend()

# Remove the last empty subplot 
if len(columns_to_plot) % n_cols != 0:
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()


# Filter out 'No Failure' data
filtered_df = pred_df_cleaned_celcius[pred_df_cleaned_celcius['Failure type'] != 'No Failure']

# Setup for subplots
# Setup for subplots
fig, axes = plt.subplots(3, 2, figsize=(20, 15))
axes = axes.flatten()

# Iterate through each operational parameter to create histograms
for i, column in enumerate(columns_to_plot):
    ax = sns.histplot(data=filtered_df, x=column, hue='Failure type', multiple="stack", palette=custom_palette, ax=axes[i], bins=20)
    axes[i].set_title(f'Distribution of {column} by Failure State (Excluding No Failure)')

if len(columns_to_plot) % 2 != 0:
    fig.delaxes(axes[-1])
    
plt.tight_layout()
plt.show()


# Group by 'Failure type' and count occurrences
failure_counts = filtered_df.groupby('Failure type').size().sort_values(ascending=False).reset_index(name='count')

# Create a bar plot for the failure counts
fig = px.bar(failure_counts, x='count', y='Failure type', color='Failure type', title='Count of Each Failure Type', labels={'count': 'Count', 'Failure type': 'Failure Type'}, orientation='h')
fig.update_traces(texttemplate='%{x}', textposition='outside')
fig.show()



# Create an empty DataFrame to hold concatenated pivot tables
combined_pivot = pd.DataFrame()

# Loop through each column to create pivot tables and concatenate them horizontally
for column in columns_to_plot:
    pivot_table = pred_df_cleaned_celcius.pivot_table(index='Failure type', values=column, aggfunc='mean')
    combined_pivot = pd.concat([combined_pivot, pivot_table], axis=1)

# Display the combined pivot table
display(combined_pivot.style.set_caption("Combined Pivot Tables"))



for column in columns_to_plot:
    print(pd.crosstab(pred_df_cleaned_celcius['Failure type'], pred_df_cleaned_celcius[column]))


color = 'Failure type' if 'Failure type' in filtered_df.columns else None

fig = px.scatter_matrix(filtered_df, dimensions=columns_to_plot, color=color, title="Pairplot of Operational Parameters")
fig.update_layout(height=1200, width=1200)
fig.show()


fig = px.imshow(pred_df_cleaned_celcius[['Air temperature [C]', 'Process temperature [C]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']].corr(), text_auto=True, title='Correlation Matrix')
fig.update_layout(width=800, height=600)
fig.show()


values = pd.DataFrame([(ttest_ind(pred_df_cleaned_celcius[pred_df_cleaned_celcius['Machine failure'] == 1][col], 
                                   pred_df_cleaned_celcius[pred_df_cleaned_celcius['Machine failure'] == 0][col])) 
                       for col in columns_to_plot], 
                      columns=['test-statistic', 'p-value'], index=columns_to_plot)

values['Hypothesis'] = np.where(values['p-value'] < 0.05, 'Reject null hypothesis', 'Fail to reject null hypothesis')
values


values = pd.DataFrame([(mannwhitneyu(pred_df_cleaned_celcius[pred_df_cleaned_celcius['Machine failure'] == 1][col], 
                                      pred_df_cleaned_celcius[pred_df_cleaned_celcius['Machine failure'] == 0][col])) 
                       for col in columns_to_plot], 
                      columns=['statistic', 'p-value'], index=columns_to_plot)

values['Hypothesis'] = np.where(values['p-value'] < 0.05, 'Reject null hypothesis', 'Fail to reject null hypothesis')
values


X = pred_df_cleaned_celcius.drop(['Product ID', 'Type', 'Machine failure', 'Failure type'], axis=1)
y = pred_df_cleaned_celcius['Machine failure']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize models
logistic_model = make_pipeline(MinMaxScaler(), LogisticRegression(random_state=42, max_iter=1000))
decision_tree_model = DecisionTreeClassifier(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)




# List of models for iteration
models = [logistic_model, decision_tree_model, random_forest_model]

# Iterate through models, fit, predict, and print metrics
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

# Print model name and metrics
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    print("--------------------------------------------------")



# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Iterate through models, fit, predict, and plot ROC curve
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot ROC curve for random guessing
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
plt.show()


# confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i, model in enumerate(models):
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion, annot=True, fmt='d', cmap='viridis', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - {model_name}')
    axes[i].set_xlabel('Predicted Labels')
    axes[i].set_ylabel('True Labels')

plt.tight_layout()
plt.show()


def find_threshold_by_criteria(y_test, y_score, tpr_criteria=0.6, fpr_criteria=0.4):
    """
    Find a threshold where TPR is greater than a specified criteria and FPR is less than another specified criteria.

    Parameters:
    - y_test: The true binary labels.
    - y_score: The target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
    - tpr_criteria (float): The minimum TPR value desired.
    - fpr_criteria (float): The maximum FPR value allowed.

    Returns:
    - The threshold meeting the specified criteria or None if no such threshold exists.
    """

    # Calculate FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])

    # Iterate through each FPR, TPR, and threshold
    for f, t, th in zip(fpr, tpr, thresholds):
        if t > tpr_criteria and f < fpr_criteria:
            return th  # Return the threshold once the condition is met

    return None  # Return None if no threshold meets the criteria

# Example usage
tpr_criteria = 0.6
fpr_criteria = 0.4
desired_threshold = find_threshold_by_criteria(y_test, y_score, tpr_criteria, fpr_criteria)
if desired_threshold is not None:
    print(f"Threshold where TPR > {tpr_criteria*100}% and FPR < {fpr_criteria*100}%: {desired_threshold}")
else:
    print(f"No threshold found meeting the criteria of TPR > {tpr_criteria*100}% and FPR < {fpr_criteria*100}%.") 



# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()