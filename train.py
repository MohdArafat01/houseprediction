import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

csv_path = 'data/housing.csv'

# 1. Dataset Generation
np.random.seed(42)
n_samples = 1500

locations = ['Urban Central', 'Suburban Area', 'Downtown Luxury', 'Rural Outskirts']
loc_factor = {'Urban Central': 1.4, 'Suburban Area': 1.0, 'Downtown Luxury': 1.8, 'Rural Outskirts': 0.7}

loc_data = np.random.choice(locations, n_samples)
sqft = np.random.randint(600, 4500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 5, n_samples)
age = np.random.randint(0, 35, n_samples)

price = []
for i in range(n_samples):
    base = (sqft[i] * 180) + (bedrooms[i] * 12000) + (bathrooms[i] * 18000) - (age[i] * 1200)
    adjusted_price = base * loc_factor[loc_data[i]] + np.random.normal(0, 12000)
    price.append(round(adjusted_price, 2))

df = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'location': loc_data,
    'price': price
})
df.to_csv(csv_path, index=False)

# 2. Preprocessing & Encoding
df = pd.get_dummies(df, columns=['location'], drop_first=False)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluation Metrics for Academic Viva
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"--- Model Performance Metrics ---")
print(f"R² Score: {r2:.4f} (Accuracy Factor: {r2*100:.2f}%)")
print(f"Mean Absolute Error: ${mae:,.2f}")

# Save artifacts & feature column layout
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/columns.pkl')