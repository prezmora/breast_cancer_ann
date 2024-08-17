import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = load_breast_cancer()
data_df = pd.DataFrame(data.data, columns=data.feature_names)
data_df['target'] = data.target

# Data preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(data_df.drop(columns=['target']))
y = data_df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=1000)
model.fit(X_train, y_train)

# Streamlit app
st.title("Breast Cancer Prediction App")

# User inputs
st.sidebar.header("Input Parameters")
user_input = []
for feature in data_df.columns[:-1]:
    value = st.sidebar.slider(f"{feature}", float(data_df[feature].min()), float(data_df[feature].max()), float(data_df[feature].mean()))
    user_input.append(value)

# Predict based on user inputs
user_input = scaler.transform([user_input])
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display results
st.write(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")
st.write(f"Prediction Probability: {prediction_proba[0][1]:.2%} Malignant, {prediction_proba[0][0]:.2%} Benign")

# Display model evaluation
st.write("Model Evaluation")
st.write(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
st.text(classification_report(y_test, model.predict(X_test)))
