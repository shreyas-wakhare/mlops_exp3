import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Step 1: Load the saved model
with open('model_v1.pkl', 'rb') as f:
    model = pickle.load(f)

print("Loaded model_v1.pkl successfully!")

# Step 2: Load the Iris dataset again
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

X = df[iris.feature_names]
y = df['species']

# Step 3: Make predictions
y_pred = model.predict(X)

# Step 4: Evaluate model performance
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy of loaded model: {accuracy:.2f}")
