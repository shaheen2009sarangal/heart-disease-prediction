import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load Dataset
df = pd.read_csv('heart.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget value counts:")
print(df['HeartDisease'].value_counts())

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Accuracy
lr_acc = round(accuracy_score(y_test, lr.predict(X_test)) * 100, 2)
dt_acc = round(accuracy_score(y_test, dt.predict(X_test)) * 100, 2)
rf_acc = round(accuracy_score(y_test, rf.predict(X_test)) * 100, 2)

print("\nLogistic Regression:", lr_acc, "%")
print("Decision Tree      :", dt_acc, "%")
print("Random Forest      :", rf_acc, "%")

# Save best model
if rf_acc >= lr_acc and rf_acc >= dt_acc:
    best_model = rf
    best_name = "Random Forest"
    best_acc = rf_acc
elif lr_acc >= dt_acc:
    best_model = lr
    best_name = "Logistic Regression"
    best_acc = lr_acc
else:
    best_model = dt
    best_name = "Decision Tree"
    best_acc = dt_acc

print("\nBest Model:", best_name, "—", best_acc, "%")

pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("Model saved successfully!")