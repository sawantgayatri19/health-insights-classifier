"""Data Pipeline for Health Insights Classifier"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class HealthDataPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def create_sample_data(self, n_samples=500):
        """Create synthetic health data"""
        np.random.seed(self.random_state)
        
        data = {
            'age': np.random.randint(18, 85, n_samples),
            'bmi': np.random.uniform(15, 40, n_samples),
            'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
            'blood_pressure_diastolic': np.random.randint(60, 120, n_samples),
            'cholesterol': np.random.randint(120, 300, n_samples),
            'glucose': np.random.randint(70, 200, n_samples),
            'exercise_hours_per_week': np.random.uniform(0, 20, n_samples),
            'sleep_hours': np.random.uniform(4, 12, n_samples),
            'stress_level': np.random.randint(1, 10, n_samples),
            'smoking_status': np.random.choice(['never', 'former', 'current'], n_samples),
            'alcohol_consumption': np.random.choice(['none', 'moderate', 'frequent'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create health risk target - FIXED to avoid NaN
        risk_score = (
            (df['age'] > 50).astype(int) * 2 +
            (df['bmi'] > 25).astype(int) * 2 +
            (df['blood_pressure_systolic'] > 140).astype(int) * 3 +
            (df['cholesterol'] > 240).astype(int) * 2 +
            (df['glucose'] > 125).astype(int) * 2 +
            (df['exercise_hours_per_week'] < 3).astype(int) * 1 +
            (df['sleep_hours'] < 7).astype(int) * 1 +
            (df['stress_level'] > 7).astype(int) * 2 +
            (df['smoking_status'] == 'current').astype(int) * 3
        )
        
        # Create categories without NaN
        df['health_risk'] = pd.cut(
            risk_score, 
            bins=[-1, 5, 10, 100],  # Changed to avoid edge cases
            labels=['low_risk', 'moderate_risk', 'high_risk'],
            include_lowest=True
        )
        
        # Drop any remaining NaN rows (should be none now)
        df = df.dropna()
        
        print(f"✓ Created synthetic dataset with {len(df)} samples")
        return df
    
    def prepare_data(self, df, target_col='health_risk', test_size=0.2):
        """Prepare data for training"""
        numeric_features = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                           'cholesterol', 'glucose', 'exercise_hours_per_week', 'sleep_hours', 'stress_level']
        categorical_features = ['smoking_status', 'alcohol_consumption']
        
        # Encode categorical
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
        
        # Scale numeric
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        
        # Split data
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Check for NaN before split
        X = X.dropna()
        y = y[X.index]
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Data preparation complete!")
        print(f"  Training: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    pipeline = HealthDataPipeline()
    df = pipeline.create_sample_data(n_samples=500)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    print(f"✓ Classes: {y_train.unique()}")
