"""
Predictive Maintenance Dashboard
================================

Real-time equipment health monitoring with human override alerts.
Implements Random Forest regression for Remaining Useful Life (RUL) prediction.

Author: MiniMax Agent
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenance:
    """
    Predictive Maintenance System for Equipment Health Monitoring
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the Predictive Maintenance system.
        
        Args:
            n_estimators (int): Number of trees in Random Forest
            random_state (int): Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_columns = None
        self.target_column = None
        self.is_trained = False
        
    def generate_sample_data(self, n_samples=1000, equipment_types=5):
        """
        Generate synthetic manufacturing sensor data.
        
        Args:
            n_samples (int): Number of data points to generate
            equipment_types (int): Number of different equipment types
            
        Returns:
            pd.DataFrame: Generated sensor data
        """
        np.random.seed(self.random_state)
        
        # Equipment base characteristics
        equipment_base = np.random.uniform(0.8, 1.2, equipment_types)
        
        data = []
        for i in range(n_samples):
            equipment_type = np.random.randint(0, equipment_types)
            base_factor = equipment_base[equipment_type]
            
            # Generate sensor readings
            sample = {
                'equipment_id': f'EQUIP_{equipment_type:02d}_{i:04d}',
                'equipment_type': equipment_type,
                'vibration': np.random.normal(2.5 * base_factor, 0.5),
                'temperature': np.random.normal(75 * base_factor, 5),
                'pressure': np.random.normal(100 * base_factor, 10),
                'runtime_hours': np.random.randint(0, 5000),
                'maintenance_count': np.random.randint(0, 10),
                'operating_load': np.random.uniform(0.6, 1.0),
                'ambient_temp': np.random.normal(22, 3),
                'humidity': np.random.uniform(30, 80),
            }
            
            # Simulate RUL based on sensor data and operational parameters
            degradation_factor = (
                sample['vibration'] * 50 + 
                sample['temperature'] * 2 + 
                sample['pressure'] * 0.5 +
                sample['maintenance_count'] * 100 +
                (1 - sample['operating_load']) * 200
            )
            
            sample['rul'] = max(0, 5000 - sample['runtime_hours'] - degradation_factor)
            sample['failure_probability'] = 1 / (1 + np.exp(-(sample['rul'] - 500) / 200))
            
            data.append(sample)
            
        df = pd.DataFrame(data)
        self.feature_columns = [col for col in df.columns 
                               if col not in ['rul', 'equipment_id', 'failure_probability']]
        self.target_column = 'rul'
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (features, target)
        """
        features = df[self.feature_columns].copy()
        
        # Feature engineering
        features['vibration_temp_ratio'] = features['vibration'] / (features['temperature'] + 1)
        features['pressure_vibration_ratio'] = features['pressure'] / (features['vibration'] + 1)
        features['load_efficiency'] = features['operating_load'] * (100 - features['temperature'])
        
        target = df[self.target_column]
        
        return features, target
    
    def train(self, df, test_size=0.2):
        """
        Train the predictive maintenance model.
        
        Args:
            df (pd.DataFrame): Training data
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Training metrics
        """
        print("🔧 Preparing features...")
        X, y = self.prepare_features(df)
        
        print("📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print("🌲 Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_features': X.shape[1],
            'n_samples': len(X)
        }
        
        self.is_trained = True
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Training completed!")
        print(f"   MAE: {test_mae:.2f} hours")
        print(f"   RMSE: {test_rmse:.2f} hours")
        print(f"   R²: {test_r2:.3f}")
        
        return metrics
    
    def predict(self, df):
        """
        Make predictions on new data.
        
        Args:
            df (pd.DataFrame): Data for prediction
            
        Returns:
            np.array: RUL predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X, _ = self.prepare_features(df)
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance rankings.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def predict_failure_risk(self, df, threshold=500):
        """
        Predict failure risk for equipment.
        
        Args:
            df (pd.DataFrame): Equipment data
            threshold (int): RUL threshold for high risk
            
        Returns:
            pd.DataFrame: Data with failure risk predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        df_result = df.copy()
        df_result['predicted_rul'] = self.predict(df)
        df_result['failure_risk'] = df_result['predicted_rul'] < threshold
        df_result['risk_level'] = pd.cut(
            df_result['predicted_rul'],
            bins=[0, 200, 500, 1000, float('inf')],
            labels=['Critical', 'High', 'Medium', 'Low']
        )
        
        return df_result
    
    def generate_maintenance_schedule(self, df, planning_horizon=30):
        """
        Generate maintenance schedule based on predictions.
        
        Args:
            df (pd.DataFrame): Equipment data with predictions
            planning_horizon (int): Days to plan ahead
            
        Returns:
            pd.DataFrame: Maintenance schedule
        """
        df_with_risk = self.predict_failure_risk(df)
        current_date = datetime.now()
        
        schedule = []
        for _, equipment in df_with_risk.iterrows():
            days_to_failure = equipment['predicted_rul'] / 24  # Convert hours to days
            
            if days_to_failure <= planning_horizon:
                recommended_date = current_date + timedelta(days=days_to_failure)
                priority = 'High' if days_to_failure <= 7 else 'Medium'
                
                schedule.append({
                    'equipment_id': equipment['equipment_id'],
                    'current_rul': equipment['predicted_rul'],
                    'days_to_failure': days_to_failure,
                    'recommended_maintenance': recommended_date.strftime('%Y-%m-%d'),
                    'priority': priority,
                    'risk_level': equipment['risk_level']
                })
        
        return pd.DataFrame(schedule).sort_values('days_to_failure')
    
    def plot_predictions(self, df, save_path=None):
        """
        Create visualization of predictions vs actual values.
        
        Args:
            df (pd.DataFrame): Data with actual and predicted RUL
            save_path (str): Path to save the plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        df_with_pred = df.copy()
        df_with_pred['predicted_rul'] = self.predict(df)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Predictive Maintenance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted RUL
        axes[0, 0].scatter(df['rul'], df_with_pred['predicted_rul'], alpha=0.6, color='steelblue')
        axes[0, 0].plot([0, 5000], [0, 5000], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Actual RUL (hours)')
        axes[0, 0].set_ylabel('Predicted RUL (hours)')
        axes[0, 0].set_title('Actual vs Predicted RUL')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance
        importance_df = self.get_feature_importance(8)
        axes[0, 1].barh(importance_df['feature'], importance_df['importance'])
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_title('Top Feature Importance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RUL Distribution
        axes[1, 0].hist(df['rul'], bins=30, alpha=0.7, color='lightcoral', label='Actual')
        axes[1, 0].hist(df_with_pred['predicted_rul'], bins=30, alpha=0.7, color='skyblue', label='Predicted')
        axes[1, 0].set_xlabel('RUL (hours)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('RUL Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Equipment Health Status
        health_status = df_with_pred['predicted_rul'].apply(
            lambda x: 'Critical' if x < 200 else 'High' if x < 500 else 'Medium' if x < 1000 else 'Low'
        )
        status_counts = health_status.value_counts()
        colors = ['red', 'orange', 'yellow', 'green']
        axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[1, 1].set_title('Equipment Health Status')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        import joblib
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        import joblib
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.feature_names = model_data['feature_names']
        self.target_column = model_data['target_column']
        self.n_estimators = model_data['n_estimators']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"📁 Model loaded from: {filepath}")

def main():
    """
    Example usage of the Predictive Maintenance system.
    """
    print("🚀 Predictive Maintenance System Demo")
    print("=" * 50)
    
    # Initialize system
    pm_system = PredictiveMaintenance(n_estimators=100)
    
    # Generate sample data
    print("\n📊 Generating sample sensor data...")
    data = pm_system.generate_sample_data(n_samples=2000, equipment_types=3)
    
    # Train the model
    print("\n🔧 Training predictive maintenance model...")
    metrics = pm_system.train(data)
    
    # Generate predictions
    print("\n🔮 Making predictions...")
    predictions = pm_system.predict(data[:100])
    
    # Generate maintenance schedule
    print("\n📅 Generating maintenance schedule...")
    schedule = pm_system.generate_maintenance_schedule(data, planning_horizon=30)
    
    if len(schedule) > 0:
        print(f"⚠️  {len(schedule)} equipment items need maintenance within 30 days")
        print(schedule.head())
    else:
        print("✅ All equipment has adequate remaining useful life")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    pm_system.plot_predictions(data[:500])
    
    # Save model
    print("\n💾 Saving trained model...")
    pm_system.save_model('data/processed/predictive_maintenance_model.pkl')
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()