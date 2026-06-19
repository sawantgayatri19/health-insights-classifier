"""Model Training for Health Insights Classifier"""
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_pipeline import HealthDataPipeline

class HealthRiskClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.metrics = {}
    
    def train(self, X_train, y_train):
        print("🔄 Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("✓ Model training complete!")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        for metric, value in self.metrics.items():
            print(f"{metric.capitalize():15s}: {value:.4f}")
        
        return self.metrics
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {filepath}")

if __name__ == "__main__":
    # Prepare data
    pipeline = HealthDataPipeline()
    df = pipeline.create_sample_data(n_samples=500)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    
    # Train model
    classifier = HealthRiskClassifier()
    classifier.train(X_train, y_train)
    classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save_model('models/classifier.pkl')
    print("\n✓ Training complete!")
