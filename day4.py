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

# Problem 4: Car Age vs Mileage Relationship

# Objective 4.1: Car Age already calculated
df['Car Age'] = 2025 - df['Year']

# Objective 4.2: Scatterplot
sns.scatterplot(x='Car Age', y='Mileage', data=df, alpha=0.6)
plt.title('Car Age vs Mileage')
plt.xlabel('Car Age')
plt.ylabel('Mileage')
plt.grid(True)
plt.show()

# Objective 4.3: Regression Plot
sns.lmplot(x='Car Age', y='Mileage', data=df, aspect=1.5, scatter_kws={'alpha':0.3})
plt.title('Regression: Car Age vs Mileage')
plt.show()

# Objective 4.4: High Mileage Detection
df['Make_Model'] = df['Make'].astype(str) + ' ' + df['Model'].astype(str)
df['High Mileage'] = (df['Car Age'] < 5) & (df['Mileage'] > 200000)
print(df[df['High Mileage']][['Make_Model', 'Year', 'Mileage']].head())
