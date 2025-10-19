import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
 

os.makedirs('model', exist_ok=True)

df = pd.read_csv('data/data.csv')

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, 'model/model.joblib')

with open('metrics.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')

print(f"Model trained with accuracy: {accuracy}")
print("Model saved to model/model.joblib")
print("Metrics saved Metrics.txt")
print("hellp")
