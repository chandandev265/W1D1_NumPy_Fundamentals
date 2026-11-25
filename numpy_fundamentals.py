
import numpy as np

# Create a 1D array
arr_1d = np.array([10, 20, 30, 40, 50])

# Create a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Create a 3D array
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("1D Array:")
print(arr_1d)
print("Shape:", arr_1d.shape)
print("Dimensions:", arr_1d.ndim)

print("\n2D Array:")
print(arr_2d)
print("Shape:", arr_2d.shape)
print("Dimensions:", arr_2d.ndim)

print("\n3D Array:")
print(arr_3d)
print("Shape:", arr_3d.shape)
print("Dimensions:", arr_3d.ndim)

print("\n--- Broadcasting ---")

numbers = np.array([10, 20, 30])
result = numbers + 5

print("Original:", numbers)
print("After adding 5:", result)

print("\n--- Vectorized Operations ---")

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

print("Addition:", a + b)
print("Subtraction:", b - a)
print("Multiplication:", a * b)
print("Division:", b / a)

print("\n--- Matrix Multiplication ---")

matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

result = np.matmul(matrix_a, matrix_b)

print("Matrix A:")
print(matrix_a)

print("Matrix B:")
print(matrix_b)

print("Matrix Multiplication Result:")
print(result)

import pandas as pd

print("\n--- CSV Statistics ---")

df = pd.read_csv("data/student_scores.csv")

study_hours = df["Study_Hours"].to_numpy()
marks = df["Marks"].to_numpy()

print("Dataset:")
print(df)

print("\nMean Study Hours:", np.mean(study_hours))
print("Mean Marks:", np.mean(marks))

print("\nStandard Deviation of Study Hours:", np.std(study_hours))
print("Standard Deviation of Marks:", np.std(marks))

print("\nCorrelation between Study Hours and Marks:")
print(np.corrcoef(study_hours, marks)[0, 1])
