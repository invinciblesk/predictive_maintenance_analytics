# Predictive Maintenance Analytics


</a>

## Table of Contents

- [1. Background](#1-background)
- [2. Dataset Overview](#2-dataset-overview)
- [3. Data Processing](#3-data-processing)
- [4. Model Development](#4-model-development)
- [5. Evaluation and Key Findings](#5-evaluation-and-key-findings)
- [6. Future Work](#6-future-work)
- [7. How to Run](#7-how-to-run)
- [8. Visualization](#8-visualization)
- [9. Contributors](#9-contributors)
- [10. License](#10-license)

## 1. Background
This project focuses on predictive maintenance analytics, utilizing machine learning to predict potential machine failures. By analyzing sensor data, we aim to identify conditions that may lead to machine failures, enabling proactive maintenance measures and reducing unexpected downtime.

## 2. Dataset Overview
We utilized a dataset containing machine sensor data, available on [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/data).  It contains various manufacturing process parameters and outcomes essential for understanding machine health and predicting potential failures.

### Dataset Composition
The dataset comprises `14 columns` and 10,000 rows, each providing unique insights into the operational parameters machines:

![sample](/img/pred_df_head.png)

- **UDI**: A sequential unique identifier for each record.
- **Product ID**: Alphanumeric codes representing different products.
- **Type**: A categorical variable indicating the product type, with possible values 'M' (for Model M) and 'L' (for Model L).
- **Air temperature [K]**: The air temperature during the manufacturing process, measured in Kelvin.
- **Process temperature [K]**: The temperature of the manufacturing process itself, also in Kelvin.
- **Rotational speed [rpm]**: The speed at which the machine operates, in revolutions per minute.
- **Torque [Nm]**: The torque produced by the machine, in Newton-meters.
- **Tool wear [min]**: The cumulative wear on the tool used in the manufacturing process, measured in minutes.
- **Machine failure**: A binary indicator of whether a machine failure occurred, with '0' indicating no failure.
- **Failure Modes**: Includes several binary columns indicating specific failure modes:
  - **TWF** (Tool Wear Failure): Indicates failures due to tool wear.
  - **HDF** (Heat Dissipation Failure): Indicates failures due to inadequate heat dissipation.
  - **PWF** (Power Failure): Indicates failures due to power issues.
  - **OSF** (Overstrain Failure): Indicates failures due to overstraining of the machine.
  - **RNF** (Random Failures): Indicates failures that do not fit into the other categories, deemed random.

## 3. Data Processing
The data processing phase encompassed several key steps to prepare the dataset for analysis:

- **Data Cleaning**: Initiated by removing irrelevant data, including the dropping of the `UDI` and `Product ID` columns, to streamline the dataset for more focused analysis.

- **Data Transformation**: 
  - **Categorical Variables**: The failure mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were consolidated into a single column to simplify the analysis and model training process.
  - **Normalization**: Implemented bootstrapping techniques to normalize the data, ensuring that the model is not biased by the scale of any feature.

  ![Bootstrapped Means Visualization](/img/bootstrapped_means1.png)

- **Outlier Handling**: Outliers were identified through statistical analysis. However, given the nature of sensor data, these outliers were retained. This decision was made to preserve potential insights into extreme operational conditions that could be critical for predicting machine failures.

![No_Outliers](/img/boxplots.png)

**Missing values**:
- No missing values were found, so no imputation was necessary.

## 4. Initial Exploration
**Question 1:** What is the average distribution between the operational parameters and failure types?

![Distribution](/img/failure_type_counts.png)

**Question 2:** Calculate and illustrate most common failure type in the dataset.

![Failure_Count](/img/failure_distributions.png)

## 5. Hypothesis Testing
**Question 1:** Does a relationship exist between operational parameters and the occurrence of machine failures?

`Null Hypothesis (H0)`: There is no significant relationship between the operational parameters and `machine failure`.

`Alternate Hypothesis (H1)`: There is a significant relationship between the operational parameters `machine failure`.

**Question 2**: Given the operational parameters and machine states available in the dataset, can we identify patterns or conditions indicative of an impending failure?

`Null Hypothesis (H0)`: There are no patterns or conditions in the operational parameters and machine states available in the dataset that are indicative of an impending failure.

`Alternate Hypothesis (H1)`: There are patterns or conditions in the operational parameters and machine states available in the dataset that are indicative of an impending failure.

![Relationships](/img/pairplot_ops_params1.png)

![Heatmap](/img/corr_matrix1.png)


![Test](/img/mannwhitneyu_test.png)

![TTest](/img/ttest.png)

### Hypothesis Testing Conclusion:
- Patterns or conditions in the operational parameters and machine states available in the dataset are indicative of an impending failure.
- There is a significant relationship between the operational parameters and machine failure.
- The `Null Hypothesis` is `rejected` in both instances.

## 6. Model Development
Experimented with three machine learning models to predict potential machine failures:
- **Logistic Regression:** Provided a solid baseline for performance with good interpretability.
- **Decision Tree:** Delivered the best performance overall, with high accuracy and robustness against overfitting.
- **Random Forest:** Delivered good performance, with high accuracy and robustness against overfitting.

## 7. Evaluation and Key Findings
Our evaluation process highlighted the following key findings:
- The Random Forest model outperformed other models in terms of accuracy and robustness.
- Feature importance analysis revealed that rotational speed and torque are significant predictors of machine failure.

## 8. Future Work
For future enhancements, we plan to:
- Explore hyperparameter tuning and feature engineering to further enhance model performance.
- Integrate additional data sources to provide a more comprehensive view for predictions.
- Develop a system for real-time predictions and alerts to facilitate immediate maintenance actions.

## 9. How to Run
To set up and run the project, follow these steps:
1. Clone the repository to your local machine.
2. CD to src directory
3. Run the main script with `python main.py --data_path ../data/ai4i2020.csv `.

## 10. Visualization
To view the generated visualization images, navigate to the `img/` directory after running the project. You will find images such as:
- `rotational_speed_vs_torque.png` - A scatter plot showing the relationship between rotational speed and torque.
- `tool_wear_over_time.png` - A line graph depicting tool wear over time.

These images provide insights into the data and model performance.

## 11. Contributors
- Simon Kanyiva

