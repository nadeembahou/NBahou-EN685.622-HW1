import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def train_and_evaluate_classification_models(X_train, X_test, y_train, y_test):
    """Train and evaluate classification models with cross-validation."""
    if not pd.api.types.is_numeric_dtype(y_train):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Train model on the full training set
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, predictions)
        
        results[name] = {
            'predictions': predictions,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'report': classification_report(y_test, predictions)
        }
    
    return results

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(data.head())
    
    # Target selection
    target = st.selectbox('Select the target variable', data.columns)
    
    if st.button('Train Models'):
        # Split features and target
        X = data.drop(target, axis=1)
        y = data[target]
        
        # Use stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        st.write("Training set size:", len(X_train))
        st.write("Test set size:", len(X_test))
        
        # Train and evaluate models
        results = train_and_evaluate_classification_models(X_train, X_test, y_train, y_test)
        
        # Display results for each model
        for name, result in results.items():
            st.subheader(f"{name} Results")
            st.write(f"Test Set Accuracy: {result['test_accuracy']:.2%}")
            st.write(f"5-Fold Cross-Validation Mean Accuracy: {result['cv_mean']:.2%} (±{result['cv_std']*2:.2%})")
            
            st.text("Classification Report:")
            st.text(result['report'])
            
            # Plot confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap(
                confusion_matrix(y_test, result['predictions']), 
                annot=True, 
                fmt='d', 
                ax=ax,
                cmap='Blues'
            )
            ax.set_title(f'{name} Confusion Matrix')
            st.pyplot(fig)
else:
    st.info("Please upload a CSV file to begin.")
