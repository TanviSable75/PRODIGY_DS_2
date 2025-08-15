# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 2. Load Dataset
df = pd.read_csv("train.csv")  # Make sure 'titanic.csv' is in your working directory

# 3. Initial Overview
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe(include='all'))

# 4. Data Cleaning
# Drop 'Cabin' due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Drop 'Ticket' - rarely useful
df.drop(columns=['Ticket'], inplace=True)

# Fill missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop rows with missing 'Fare' if any
df.dropna(subset=['Fare'], inplace=True)

# Convert 'Sex' and 'Embarked' to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 5. EDA

# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Survival by Sex
sns.barplot(x='Sex_male', y='Survived', data=df)
plt.title("Survival Rate by Gender (1 = Male)")
plt.show()

# Survival by Pclass
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

# Age Distribution and Survival
plt.figure(figsize=(12, 6))
sns.histplot(df[df['Survived'] == 1]['Age'], kde=True, color='green', label='Survived', bins=30)
sns.histplot(df[df['Survived'] == 0]['Age'], kde=True, color='red', label='Did Not Survive', bins=30)
plt.legend()
plt.title("Age Distribution by Survival Status")
plt.show()

# Survival by Fare
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Fare vs Survival")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 6. Export Cleaned Data (Optional)
df.to_csv("titanic_cleaned.csv", index=False)
