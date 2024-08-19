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


