from azureml.core import Workspace, Experiment, Run
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connect to Azure ML workspace
ws = Workspace.from_config()

# Create experiment
experiment = Experiment(workspace=ws, name="simple-iris-classification")

# Start run
run = experiment.start_logging()

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Log metric to Azure ML
run.log("accuracy", accuracy)

print("Model Accuracy:", accuracy)

# Complete run
run.complete()
