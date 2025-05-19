
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/content/phase 2 data set (1).csv")

# Drop unnecessary or mostly empty columns
drop_cols = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
df = df.drop(columns=drop_cols, errors='ignore')

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill missing numerical values with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill missing categorical values with mode
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Add dummy SalePrice if not present
if 'SalePrice' not in df.columns:
    df['SalePrice'] = np.random.randint(100000, 500000, df.shape[0])

# Define features (X) and target (y)
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
# Evaluate model
# rmse = mean_squared_error(y_test, y_pred, squared=False) # Remove squared=False
mse = mean_squared_error(y_test, y_pred) # Calculate MSE first
rmse = np.sqrt(mse) # Then calculate RMSE
r2 = r2_score(y_test, y_pred)

# Display results
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Show actual vs predicted scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted Sale Price")
plt.grid(True)
plt.show()

# Predict first 5 rows
print("Predictions for first 5 rows:")
print(model.predict(X.head(5)))
