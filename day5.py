import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from numpy import polyfit, poly1d

# Load dataset
df = pd.read_csv(r"C:\Users\Deepanshu\Downloads\uae_used_cars_10k.csv")
# Problem 5: Price Distribution by Transmission and Fuel
# Objective 5.1: Boxplot by Transmission
sns.boxplot(x='Transmission', y='Price', data=df)
plt.title("Car Price by Transmission")
plt.show()

# Objective 5.2: Heatmap Transmission vs Fuel
cross = pd.crosstab(df['Transmission'], df['Fuel Type'])
sns.heatmap(cross, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Transmission vs Fuel Type")
plt.show()

# Objective 5.3: Mileage by Fuel Type
sns.boxplot(x='Fuel Type', y='Mileage', data=df)
plt.title("Mileage by Fuel Type")
plt.show()

# Objective 5.4: Count Plot of Transmission
sns.countplot(x='Transmission', data=df, palette="pastel")
plt.title("Count of Cars by Transmission Type")
plt.show()
