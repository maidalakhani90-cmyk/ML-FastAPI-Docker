
# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Machine Learning model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save trained model
joblib.dump(model, "model/iris_model.joblib")

print("Model training completed!")
print("Model accuracy:", accuracy)
print("Model saved successfully!")
