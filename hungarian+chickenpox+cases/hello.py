import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import timeit
import numpy as np
from scipy.stats import skew, kurtosis
import plotly.express as px
from sklearn.datasets import make_classification


# Provide the local file path
file_path = './hungary_chickenpox.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)







# Q1 Perform steps 1 to 6 and 12 to 17 from assignment 1 on the given dataset.

# 1. Display shape of the data frame

# print("Shape of the DataFrame:", df.shape)


# # 2. Display column names
# print("Column names:", df.columns)


# # 3. Display 5 quantiles of the dataset
# quantiles = df.describe(percentiles=[.25, .5, .75])

# # Print the result
# print(quantiles)



# # 4. Display count of rows having null in any column
# null_count = df.isnull().any(axis=1).sum()

# # Print the result
# print("Count of rows with null values:", null_count)

# # 5. Display first 10 rows
# first_10_rows = df.head(10)

# # Print the result
# print(first_10_rows)

# # 6. Display last 10 rows
# # Display the last 10 rows of the DataFrame
# last_10_rows = df.tail(10)

# # Print the result
# print(last_10_rows)


# # 12. Replace/eliminate missing values
# df_filled = df.fillna(0)

# # Drop rows with any remaining NaN values
# df_no_missing = df_filled.dropna()

# # Display the modified DataFrame
# print(df_no_missing)

# # 13. Change column name(s) to short/easy names if required.
# short_column_names = {
#     'Date': 'Date',
#     'BUDAPEST': 'BP',
#     'BARANYA': 'BAR',
#     'BACS': 'BCS',
#     'BEKES': 'BEK',
#     'BORSOD': 'BOR',
#     'CSONGRAD': 'CSG',
#     'FEJER': 'FEJ',
#     'GYOR': 'GYR',
#     'HAJDU': 'HAJ',
#     'HEVES': 'HEV',
#     'JASZ': 'JAS',
#     'KOMAROM': 'KOM',
#     'NOGRAD': 'NOG',
#     'PEST': 'PST',
#     'SOMOGY': 'SOM',
#     'SZABOLCS': 'SZAB',
#     'TOLNA': 'TOL',
#     'VAS': 'VAS',
#     'VESZPREM': 'VES',
#     'ZALA': 'ZAL'
# }

# # Rename columns using the dictionary
# df_renamed = df.rename(columns=short_column_names)

# # Display the modified DataFrame with the new short column names
# print(df_renamed)


# # 14. Drop unessential columns (feature selection).
# unessential_columns = ['TOLNA', 'ZALA', 'VESZPREM']  

# # Drop unessential columns
# df = df.drop(columns=unessential_columns)

# # Display the DataFrame after dropping columns
# print(df.head())

# # 15. Find mean/min/max of numeric columns.
# selected_columns = ['FEJER', 'PEST', 'SOMOGY']  # Replace with actual column names
# numeric_stats_selected = df[selected_columns].describe().loc[['mean', 'min', 'max']]
# print(numeric_stats_selected)


# # 16. Find mode of all columns.
# column_modes = df.mode()
# print(column_modes)

# # 17. Display unique values in each column
# for column in df.columns:
#     unique_values = df[column].unique()
#     print(f'Unique values in {column}: {unique_values}')



# Q. 2 2.If your dataset does not contain null values, create a dummy copy of the dataset by introducing null values. Perform following operations:
# a.Determine % of null values column-wise.
# b.Decide and perform imputation method for features having null values and justify choice of method.
# df_null = df.copy()
# df_null['New_col'] = None
# count = df_null.isnull().sum().sum()
# print(count/len(df_null) * 100)



# new_record = ['03/01/2005',168,79,30,173,169,42,136,120,162,36,130,57,2,178,66,64,11,29,87,68]
# df1 = df.copy()
# # Create a new record with None values
# new_record = ['04/01/2005', 155, None, 45, 180, 154, None, 145, None, 160, 40, None, 55, 10, 165, None, 70, 75, 25, 75, None, 22]

# new_record2 = ['15/11/2003', 15, None, 45, None, 154, None, 145, None, 160, 40, None, 55, 10, 165, None, 70, 75, 25, 75, None, 22]
# print(df.shape[1])

# # Check if the length of new_record matches the number of columns in the DataFrame
# print(len(new_record))
# print(len(df_null.columns))
# if len(new_record) == len(df_null.columns):
#     # Add the new record to the DataFrame
#     df_null.loc[len(df_null)] = new_record
#     df_null.loc[len(df_null)] = new_record2
#     # Print the DataFrame to check
#     print(df_null.tail(2))
# else:
#     print("Mismatch: Number of columns in DataFrame is not equal to the length of the new record.")

# print(df_null['ZALA'])

# print(df_null.isnull().sum()/len(df_null)*100)



# Q.3

# Select only numerical columns for scatter plot
numerical_columns = df.select_dtypes(include='number')

# # Plot scatter matrix
# scatter_matrix(numerical_columns, figsize=(15, 15), alpha=0.8, diagonal='hist')

# # Display the plot
# plt.show()





# Q.4

# Select only numerical columns for histogram
numerical_columns = df.select_dtypes(include='number')

# Plot histogram for each numerical attribute
plt.figure(figsize=(15, 12))
for column in numerical_columns.columns:
    plt.subplot(4, 5, numerical_columns.columns.get_loc(column) + 1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Calculate skewness and kurtosis
    skewness_val = skew(df[column].dropna())
    kurtosis_val = kurtosis(df[column].dropna())

    print(f"{column} - Skewness: {skewness_val}, Kurtosis: {kurtosis_val}")

plt.tight_layout()
plt.show()





# # Q.5
# categorical_columns = ['BUDAPEST', 'BARANYA', 'BACS', 'BEKES', 'BORSOD', 'CSONGRAD', 'FEJER', 'GYOR', 'HAJDU', 'HEVES',
#                         'JASZ', 'KOMAROM', 'NOGRAD', 'PEST', 'SOMOGY', 'SZABOLCS', 'TOLNA', 'VAS', 'VESZPREM', 'ZALA']

# # Plot bar graphs for each categorical attribute
# plt.figure(figsize=(15, 12))
# for column in categorical_columns:
#     plt.subplot(4, 5, categorical_columns.index(column) + 1)
#     sns.countplot(x=column, data=df, palette='viridis')
#     plt.title(f'Bar Graph of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Count')

# plt.tight_layout()
# plt.show()


# Q.6

# # Select only numerical columns for box plots
# numerical_columns = df.select_dtypes(include='number')

# # Plot box plots for each numerical attribute
# plt.figure(figsize=(15, 12))
# for column in numerical_columns.columns:
#     plt.subplot(4, 5, numerical_columns.columns.get_loc(column) + 1)
#     sns.boxplot(x=df[column])
#     plt.title(f'Box Plot of {column}')
#     plt.xlabel(column)

# plt.tight_layout()
# plt.show()


# Q.7

# # Select only numerical columns for standardization and normalization
# numerical_columns = df.select_dtypes(include='number')

# # Standardization
# scaler_standard = StandardScaler()
# df_standardized = scaler_standard.fit_transform(numerical_columns)
# df_standardized = pd.DataFrame(df_standardized, columns=numerical_columns.columns)

# # Normalization
# scaler_minmax = MinMaxScaler()
# df_normalized = scaler_minmax.fit_transform(numerical_columns)
# df_normalized = pd.DataFrame(df_normalized, columns=numerical_columns.columns)

# # Visualize original, standardized, and normalized distributions for a specific attribute
# selected_attribute = 'BUDAPEST'

# plt.figure(figsize=(15, 5))

# # Original Distribution
# plt.subplot(1, 3, 1)
# sns.histplot(df[selected_attribute], kde=True)
# plt.title(f'Original Distribution of {selected_attribute}')

# # Standardized Distribution
# plt.subplot(1, 3, 2)
# sns.histplot(df_standardized[selected_attribute], kde=True)
# plt.title(f'Standardized Distribution of {selected_attribute}')

# # Normalized Distribution
# plt.subplot(1, 3, 3)
# sns.histplot(df_normalized[selected_attribute], kde=True)
# plt.title(f'Normalized Distribution of {selected_attribute}')

# plt.tight_layout()
# plt.show()


# Q.8

# # Select only numerical columns for dimensionality reduction
# numerical_columns = df.select_dtypes(include='number')

# # Standardize the data
# scaler = StandardScaler()
# df_standardized = scaler.fit_transform(numerical_columns)

# # Perform PCA
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(df_standardized)
# df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# # Perform t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_result = tsne.fit_transform(df_standardized)
# df_tsne = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])

# # Compare visual plots
# plt.figure(figsize=(15, 5))

# # PCA Plot
# plt.subplot(1, 2, 1)
# sns.scatterplot(x='PC1', y='PC2', data=df_pca)
# plt.title('PCA Plot')

# # t-SNE Plot
# plt.subplot(1, 2, 2)
# sns.scatterplot(x='TSNE1', y='TSNE2', data=df_tsne)
# plt.title('t-SNE Plot')

# plt.tight_layout()
# plt.show()


# # Q.9

# def generate_dataset(n, m):
#     """
#     Generate a random dataset with n data points and m features.
#     """
#     return np.random.randn(n, m)

# def measure_execution_time(method, data):
#     """
#     Measure the execution time of a dimensionality reduction method on the given data.
#     """
#     start_time = timeit.default_timer()
#     method.fit_transform(data)
#     elapsed_time = timeit.default_timer() - start_time
#     return elapsed_time

# # Number of data points
# n = 10000

# # List of feature dimensions (m values)
# m_values = [10, 50, 100, 500, 1000, 1500]

# # Initialize lists to store execution times for PCA and t-SNE
# pca_execution_times = []
# tsne_execution_times = []

# # Generate datasets and measure execution times
# for m in m_values:
#     data = generate_dataset(n, m)
    
#     # PCA
#     pca = PCA(n_components=2)
#     pca_time = measure_execution_time(pca, data)
#     pca_execution_times.append(pca_time)

#     # t-SNE
#     tsne = TSNE(n_components=2)
#     tsne_time = measure_execution_time(tsne, data)
#     tsne_execution_times.append(tsne_time)

# # Plot m vs execution time for PCA and t-SNE
# plt.figure(figsize=(10, 6))
# plt.plot(m_values, pca_execution_times, label='PCA', marker='o')
# plt.plot(m_values, tsne_execution_times, label='t-SNE', marker='o')
# plt.xlabel('Number of Features (m)')
# plt.ylabel('Execution Time (seconds)')
# plt.title('Dimensionality Reduction Execution Time Comparison')
# plt.legend()
# plt.show()

# # Generate synthetic classification dataset
# X, y = make_classification(
#     n_features=6,
#     n_classes=3,
#     n_samples=1,
#     n_informative=2,
#     random_state=5,
#     n_clusters_per_class=1,
# )

# # Visualize the dataset using Plotly Express
# fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y, labels={'color': 'Class'})
# fig.update_layout(scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))
# fig.show()