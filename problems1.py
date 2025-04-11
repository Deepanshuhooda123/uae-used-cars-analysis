import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv(r"C:\Users\Deepanshu\Downloads\uae_used_cars_10k.csv")


# Problem 2: Top Car Brands and Popular Models
# Objective 2.1: Combined Make-Model already done above
# Objective 2.2: Top 10 Models
top_models = df['Make_Model'].value_counts().head(10)
print(top_models)

# Objective 2.3: Bar Plot
sns.barplot(y=top_models.index, x=top_models.values, palette="Blues_r")
plt.title("Top 10 Most Popular Car Models")
plt.xlabel("Number of Listings")
plt.ylabel("Car Model")
plt.show()

# Objective 2.4: Avg Price of Top Models
avg_price = df[df['Make_Model'].isin(top_models.index)].groupby('Make_Model')['Price'].mean()
sns.barplot(x=avg_price.index, y=avg_price.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Average Price of Top 10 Models")
plt.ylabel("Avg Price (AED)")
plt.show()
