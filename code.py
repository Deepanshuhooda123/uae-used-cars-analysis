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

# Problem 1: Price Prediction using Linear Regression
# Objective 1.1: Preprocessing
df['Car Age'] = 2025 - df['Year']
df['Make_Model'] = df['Make'] + " " + df['Model']

# Objective 1.2: Label Encoding
df_encoded = df.copy()
for col in ['Make', 'Model', 'Body Type', 'Transmission', 'Fuel Type', 'Color', 'Location']:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# Objective 1.3: Train-Test Split
X = df_encoded[['Year', 'Mileage', 'Make', 'Model', 'Body Type', 'Transmission', 'Fuel Type', 'Location']]
y = df_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Objective 1.4: Model Evaluation
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Problem 2: Top Car Brands and Popular Models
# Objective 2.1: Combined Make-Model already done above
# Objective 2.2: Top 10 Models
top_models = df['Make_Model'].value_counts().head(10)
print(top_models)

# Objective 2.3: Bar Plot
sns.barplot(y=top_models.index, x=top_models.values, hue=top_models.index, palette="Blues_r", legend=False)
plt.title("Top 10 Most Popular Car Models")
plt.xlabel("Number of Listings")
plt.ylabel("Car Model")
plt.show()

# Objective 2.4: Avg Price of Top Models
avg_price = df[df['Make_Model'].isin(top_models.index)].groupby('Make_Model')['Price'].mean()
sns.barplot(x=avg_price.index, y=avg_price.values, hue=avg_price.index, palette="coolwarm", legend=False)
plt.xticks(rotation=45)
plt.title("Average Price of Top 10 Models")
plt.ylabel("Avg Price (AED)")
plt.show()

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

# Problem 4: Car Age vs Mileage Relationship
# Objective 4.1: Car Age already calculated
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
df['High Mileage'] = (df['Car Age'] < 5) & (df['Mileage'] > 200000)
print(df[df['High Mileage']][['Make_Model', 'Year', 'Mileage']].head())

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
sns.countplot(x='Transmission', data=df, hue='Transmission', palette="pastel", legend=False)
plt.title("Count of Cars by Transmission Type")
plt.show()

# Problem 6: Depreciation Analysis
# Objective 6.1: Lineplot Avg Price vs Age
avg_price_age = df.groupby('Car Age')['Price'].mean()
sns.lineplot(x=avg_price_age.index, y=avg_price_age.values, marker='o')
plt.title("Average Price vs Car Age")
plt.xlabel("Car Age")
plt.ylabel("Avg Price (AED)")
plt.grid(True)
plt.show()

# Objective 6.2: Yearly Price Drop
price_diff = avg_price_age.diff().dropna()
print(price_diff.head())

# Objective 6.3: Polynomial Fit for Prediction
z = polyfit(df['Car Age'], df['Price'], 2)
p = poly1d(z)
future_price = p(10)
print(f"Predicted Price of Car After 10 Years: AED {future_price:.2f}")

# Objective 6.4: Plot Regression Curve
x = np.linspace(0, 20, 100)
y = p(x)
plt.plot(x, y, label='Polynomial Fit', color='red')
plt.scatter(df['Car Age'], df['Price'], alpha=0.1)
plt.title("Depreciation Curve (Polynomial Regression)")
plt.xlabel("Car Age")
plt.ylabel("Price")
plt.legend()
plt.show()
