import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('telco_customer_churn.csv')
df.columns = df.columns.str.strip()

# Sample preprocessing
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
categorical = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model and preprocessor
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('preprocessor.pkl', 'wb'))