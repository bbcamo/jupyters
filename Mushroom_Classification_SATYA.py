# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# > This notebook is refer to SATYA's notebook for data analysis studying. <br>
# (https://www.kaggle.com/code/satyaprakashshukl/mushroom-classification-analysis/notebook)

# # 데이터셋 설명
# The goal is to predict whether a mushroom is edible(e) or poisonous(p) based on its physical features, such as color, shape...

# # Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
import gc

# # Loading Dataset

df_sub = pd.read_csv("sample_submission.csv/sample_submission.csv")
df_train = pd.read_csv("train.csv/train.csv")
df_test = pd.read_csv("test.csv/test.csv")

# # Check on Data

df_train.head()

df_test.head()

df_test.shape, df_train.shape

df_train = df_train.drop(columns=["id"])
df_test = df_test.drop(columns=['id'])

# - Drop columns ID
#
# - There are a lot of features and missing values

# ## Checking distribution of categorical features

df_train.info()

# select categorical columns for train
categorical_columns = df_train.select_dtypes(include=['object']).columns

# dictionary comprehension for train
unique_values = {col: df_train[col].nunique() for col in categorical_columns}
unique_values

# for train
for col, unique_count in unique_values.items():
    print(f"{col}: {unique_count} unique values")

# iteration on test
categorical_columns = df_test.select_dtypes(include=['object']).columns
unique_values = {col: df_test[col].nunique() for col in categorical_columns}
for col, unique_count in unique_values.items():
    print(f'{col}: {unique_count} unique values')

# > Categorical features : 18(include labels), numerical features : 3 for df_train <br>
# > Categorical features : 17, numerical features : 3 for df_test

# # Exploratory Data Analysis(EDA)

import seaborn as sns

# missing value ratio of data
df_train.isna().mean()

# more than 10%
df_train.isna().mean() > 0.10

# filtering as condition
df_train.isna().mean()[df_train.isna().mean() > 0.10]

# +
missing_train = df_train.isna().mean() * 100
missing_test = df_test.isna().mean() * 100

print('Columns in df_train with more than 10% missing values:')
print(missing_train[missing_train > 10])

print('\n Columns in df_test with more than 10% missing values:')
print(missing_test[missing_test > 10])
# -

# > Both df_train and df_test share the same columns with more than 10% missing values. <br>
# > Stem_root, veil-type, veil-color, and spore-print-color have a lot of missing values, excluding 80% in both datasets. <br>
# > Cap-surface, gill-attachment, gill-spacing, and stem-sureface have moderate missing values(16% ~ 64%).

# +
# visualization
missing_values = df_train.isna().mean() * 100
missing_values = missing_values[missing_values > 0]
missing_values = missing_values.sort_values(ascending=False)

#missing_values.index

plt.figure(figsize=(10,6))
sns.barplot(x=missing_values.index, y=missing_values.values, hue=missing_values.index)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Percentage of Missing Values')
plt.title('Missing Values Distribution of df_train')
plt.show()
# -

# !pip install dython

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce
# +
# drop high missing columns
missing_threshold = 0.95
high_missing_columns = df_train.columns[df_train.isna().mean() > missing_threshold]

df_train = df_train.drop(columns=high_missing_columns)
df_test = df_test.drop(columns=high_missing_columns)


# +
# target = 'class'
# -

# 17 columns have missing value at leat one
for column in df_train.columns:
    if df_train[column].isna().any():
        print(column)


