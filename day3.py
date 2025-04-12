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

# Problem 3: Price Analysis by Location

# Objective 3.1: Avg Price by Location
avg_price_location = df.groupby('Location')['Price'].mean().sort_values(ascending=False)
print(avg_price_location)

# Objective 3.2: Bar Plot by Location
sns.barplot(x=avg_price_location.index, y=avg_price_location.values, hue=avg_price_location.index, palette="magma", legend=False)
plt.xticks(rotation=45)
plt.title("Average Used Car Price by Location")
plt.ylabel("Price (AED)")
plt.tight_layout()
plt.show()

# Objective 3.3: Pie Chart Top 3
top3 = avg_price_location.head(3)
top3.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Top 3 Locations by Price Share")
plt.ylabel("")
plt.show()

# Objective 3.4: Pie Chart Bottom 3
bottom3 = avg_price_location.tail(3)
bottom3.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Bottom 3 Locations by Price Share")
plt.ylabel("")
plt.show()
