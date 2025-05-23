import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Create directory to save models
os.makedirs("trained_data", exist_ok=True)

# Load dataset
df = pd.read_csv('csv/student_performance_dataset.csv')

# Drop unneeded columns
df = df.drop(columns=["Student_ID", "Final_Exam_Score", "Parental_Education_Level"])  # also dropped if no longer used

# Encode target variable
label_encoder = LabelEncoder()
df['Pass_Fail'] = label_encoder.fit_transform(df['Pass_Fail'])  # Pass=1, Fail=0
joblib.dump(label_encoder, 'trained_data/label_encoder.pkl')

# One-hot encode categorical features
X_raw = df.drop(columns=['Pass_Fail'])
X = pd.get_dummies(X_raw)
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

# Prepare target
y = df['Pass_Fail']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='logistic',  # this is sigmoid
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

# Train the model
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'trained_data/model_cls.pkl')

print("✅ Training complete. Model saved in 'trained_data/'")
