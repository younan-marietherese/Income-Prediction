# Predicting Income from the UCI Adult Dataset

This project aims to predict whether an individual's income exceeds $50K/year based on demographic and work-related factors. Using the **Adult Income Dataset** from the UCI Machine Learning Repository, we build and evaluate various machine learning models to perform binary classification.

---

## Project Overview
The **Adult Income Dataset** contains demographic data about individuals, including features such as age, work class, education, marital status, and occupation. The task is to predict whether an individual’s income is above or below $50,000 per year. This project explores different machine learning models and techniques to optimize predictive performance.

---

## Objectives
- Preprocess and clean the dataset from the UCI repository.
- Build machine learning models to classify individuals based on their income.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Explore feature importance and analyze the relationship between demographic factors and income.

---

## Technologies Used
- **Python**: For data manipulation, model building, and evaluation.
- **Pandas/Numpy**: For data cleaning, preprocessing, and analysis.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Matplotlib/Seaborn**: For data visualization and exploratory data analysis.

---

## Dataset
The **Adult Income Dataset** contains 48,842 rows and 15 columns, where each row represents an individual and each column represents a feature such as:
- **Age**: The age of the individual.
- **Work Class**: Type of employment (e.g., private, self-employed).
- **Education**: Level of education achieved.
- **Marital Status**: Marital status of the individual.
- **Occupation**: Type of occupation.
- **Race**: Ethnic group of the individual.
- **Sex**: Gender of the individual.
- **Hours per Week**: Number of hours worked per week.
- **Income**: Binary target variable indicating whether the individual earns more than $50K/year.

The dataset is sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/).

---

## Key Steps

1. **Data Preprocessing**:
   - Load the dataset from the UCI Machine Learning Repository.
   - Handle missing values, encode categorical variables, and normalize numerical features.
   - Split the dataset into training and testing sets for model evaluation.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of features such as age, education, and hours worked.
   - Analyze correlations between different features and their impact on income.
   - Identify potential outliers or skewed distributions that may affect model performance.

3. **Modeling**:
   - Train machine learning models such as **Logistic Regression**, **Random Forest**, and **Support Vector Machines (SVM)** to predict income.
   - Use **GridSearchCV** for hyperparameter tuning and model optimization.

4. **Evaluation**:
   - Evaluate the models using accuracy, precision, recall, F1-score, and ROC curves.
   - Compare the performance of different models and select the best-performing model.

5. **Feature Importance**:
   - Analyze feature importance to understand which factors have the greatest influence on predicting income.
   - Visualize feature importance using bar charts or heatmaps.

---

## How to Use

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
