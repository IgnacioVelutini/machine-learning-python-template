from utils import db_connect
engine = db_connect()

# your code here

import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Download the dataset
# Define paths
raw_data_dir = '../data/raw'
raw_data_path = os.path.join(raw_data_dir, 'AB_NYC_2019.csv')

# Create raw data folder if it doesn't exist
os.makedirs(raw_data_dir, exist_ok=True)

# URL of the dataset
url = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'

# Download the data if it doesn't exist
if not os.path.exists(raw_data_path):
    response = requests.get(url)
    with open(raw_data_path, 'wb') as file:
        file.write(response.content)
    print(f"Dataset downloaded and saved to {raw_data_path}")
else:
    print(f"Dataset already exists at {raw_data_path}")

# Step 2: Perform EDA
# Load the dataset
data = pd.read_csv(raw_data_path)

# Check basic info and statistics
print("Dataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())
print("\nSummary Statistics:")
print(data.describe())

# Step 3: Visualizations for key features
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], bins=50)
plt.title('Price Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(y='room_type', data=data)
plt.title('Room Type Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(y='neighbourhood_group', data=data)
plt.title('Neighbourhood Group Distribution')
plt.show()

# Step 4: Clean and preprocess the data
# Drop duplicates
data_cleaned = data.drop_duplicates()

# Handle missing values (example: dropping rows with missing values)
data_cleaned = data_cleaned.dropna()

# Drop irrelevant columns
data_cleaned = data_cleaned.drop(columns=['id', 'name', 'host_name'])

# Check remaining columns
print("Remaining columns after cleaning:", data_cleaned.columns)

# Step 5: Split the data into train and test sets
train, test = train_test_split(data_cleaned, test_size=0.2, random_state=42)

print(f"Train set shape: {train.shape}, Test set shape: {test.shape}")

# Step 6: Save the processed datasets
processed_data_dir = '../data/processed'
os.makedirs(processed_data_dir, exist_ok=True)

# Save train and test datasets
train.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)

print(f"Processed train dataset saved to {os.path.join(processed_data_dir, 'train.csv')}")
print(f"Processed test dataset saved to {os.path.join(processed_data_dir, 'test.csv')}")
