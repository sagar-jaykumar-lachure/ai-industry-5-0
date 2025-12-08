"""
Human-Cobot Collaboration Optimizer
===================================

AI-driven task allocation between humans and collaborative robots using
Gaussian Process Regression and Reinforcement Learning principles.

Author: MiniMax Agent
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class HumanCobotOptimizer:
    """
    Human-Cobot Collaboration Optimization System
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the Human-Cobot Collaboration Optimizer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.gp_model = None
        self.scaler = StandardScaler()
        self.performance_history = []
        self.collaboration_metrics = {}
        self.is_trained = False
        
    def generate_collaboration_data(self, n_samples=2000):
        """
        Generate synthetic human-cobot collaboration data.
        
        Args:
            n_samples (int): Number of collaboration scenarios
            
        Returns:
            pd.DataFrame: Collaboration data
        """
        np.random.seed(self.random_state)
        
        data = []
        task_types = ['assembly', 'quality_control', 'packaging', 'material_handling', 'inspection']
        skill_levels = ['novice', 'intermediate', 'expert']
        
        for i in range(n_samples):
            # Task characteristics
            task_type = np.random.choice(task_types)
            complexity = np.random.uniform(0.3, 1.0)  # Task complexity 0-1
            urgency = np.random.uniform(0.1, 1.0)  # Task urgency 0-1
            precision_required = np.random.uniform(0.4, 1.0)  # Precision requirement
            
            # Human worker characteristics
            human_skill = np.random.choice([0.6, 0.75, 0.9], p=[0.3, 0.5, 0.2])  # Skill level
            human_fatigue = np.random.uniform(0.0, 0.8)  # Current fatigue level
            human_experience = np.random.uniform(0.2, 1.0)  # Experience with task type
            
            # Cobot characteristics
            cobot_accuracy = np.random.uniform(0.85, 0.98)  # Cobot accuracy
            cobot_speed = np.random.uniform(0.7, 1.0)  # Cobot speed factor
            cobot_payload = np.random.uniform(5, 50)  # Payload capacity (kg)
            
            # Environmental factors
            workspace_layout = np.random.uniform(0.5, 1.0)  # Workspace efficiency
            safety_requirements = np.random.uniform(0.3, 1.0)  # Safety criticality
            lighting_conditions = np.random.uniform(0.6, 1.0)  # Lighting quality
            
            # Calculate optimal human allocation ratio using realistic physics
            # High precision tasks favor cobots, complex tasks favor humans
            precision_weight = precision_required * 0.4
            complexity_weight = complexity * 0.3
            urgency_weight = urgency * 0.2
            fatigue_penalty = human_fatigue * 0.3
            experience_bonus = human_experience * 0.2
            
            # Cobot suitability factors
            cobot_suitability = (cobot_accuracy * 0.4 + cobot_speed * 0.3 + 
                               min(cobot_payload/50, 1.0) * 0.3)
            
            # Calculate optimal human allocation
            human_allocation = (
                (complexity + experience_bonus + (1 - precision_required)) / 3 - 
                fatigue_penalty - urgency_weight + 
                (1 - cobot_suitability) * 0.3
            )
            
            # Bound and normalize
            human_allocation = np.clip(human_allocation, 0.05, 0.95)
            
            # Calculate performance metrics
            task_completion_time = (
                20 * complexity * (1 - human_allocation * 0.3) +  # Cobot time reduction
                15 * complexity * human_allocation * (1 - human_skill * 0.2) +  # Human time
                5 * (1 - workspace_layout) +  # Layout overhead
                3 * (1 - lighting_conditions)  # Lighting overhead
            )
            
            quality_score = (
                human_allocation * human_skill * 0.6 +  # Human quality contribution
                (1 - human_allocation) * cobot_accuracy * 0.8 +  # Cobot quality
                0.2 * (1 - complexity) +  # Simpler tasks have higher base quality
                0.1 * workspace_layout +  # Good layout helps quality
                0.05 * lighting_conditions  # Good lighting helps quality
            )
            
            safety_incidents = (
                np.random.poisson(max(0, (1 - human_allocation) * 0.5 +  # Fewer incidents with more human oversight
                                    (1 - safety_requirements) * 0.3 +  # Safety requirements reduce incidents
                                    human_fatigue * 0.2))  # Fatigue increases incident risk
            )
            
            human_workload = (
                human_allocation * complexity * (1 + human_fatigue) * 
                (1 - human_skill * 0.3)  # Higher skill reduces workload
            )
            
            cobot_utilization = (1 - human_allocation) * cobot_speed
            
            data.append({
                'sample_id': i,
                'task_type': task_type,
                'complexity': complexity,
                'urgency': urgency,
                'precision_required': precision_required,
                'human_skill': human_skill,
                'human_fatigue': human_fatigue,
                'human_experience': human_experience,
                'cobot_accuracy': cobot_accuracy,
                'cobot_speed': cobot_speed,
                'cobot_payload': cobot_payload,
                'workspace_layout': workspace_layout,
                'safety_requirements': safety_requirements,
                'lighting_conditions': lighting_conditions,
                'optimal_human_allocation': human_allocation,
                'task_completion_time': task_completion_time,
                'quality_score': np.clip(quality_score, 0, 1),
                'safety_incidents': safety_incidents,
                'human_workload': np.clip(human_workload, 0, 1),
                'cobot_utilization': np.clip(cobot_utilization, 0, 1)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for optimization model.
        
        Args:
            df (pd.DataFrame): Input collaboration data
            
        Returns:
            tuple: (features, target)
        """
        # Select relevant features
        feature_columns = [
            'complexity', 'urgency', 'precision_required', 'human_skill',
            'human_fatigue', 'human_experience', 'cobot_accuracy', 'cobot_speed',
            'cobot_payload', 'workspace_layout', 'safety_requirements', 'lighting_conditions'
        ]
        
        # Task type encoding
        task_type_dummies = pd.get_dummies(df['task_type'], prefix='task')
        
        # Combine features
        features = pd.concat([df[feature_columns], task_type_dummies], axis=1)
        
        # Feature engineering
        features['skill_precision_ratio'] = features['human_skill'] / (features['precision_required'] + 0.1)
        features['fatigue_workload'] = features['human_fatigue'] * features['complexity']
        features['cobot_efficiency'] = features['cobot_accuracy'] * features['cobot_speed']
        features['human_cobot_synergy'] = features['human_skill'] * features['cobot_accuracy']
        
        target = df['optimal_human_allocation']
        
        return features.values, target.values, features.columns.tolist()
    
    def train_optimization_model(self, df):
        """
        Train the Gaussian Process optimization model.
        
        Args:
            df (pd.DataFrame): Training collaboration data
            
        Returns:
            dict: Training metrics
        """
        print("🔧 Preparing features...")
        X, y, feature_names = self.prepare_features(df)
        
        print("📊 Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print("🤖 Training Gaussian Process model...")
        # Use Matern kernel for better handling of non-smooth functions
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train model
        self.gp_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.gp_model.predict(X_train)
        y_test_pred = self.gp_model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Store collaboration metrics
        self.collaboration_metrics = {
            'human_skill_distribution': df['human_skill'].describe().to_dict(),
            'task_complexity_distribution': df['complexity'].describe().to_dict(),
            'optimal_allocation_stats': df['optimal_human_allocation'].describe().to_dict(),
            'quality_vs_allocation_corr': df[['optimal_human_allocation', 'quality_score']].corr().iloc[0,1],
            'safety_incident_rate': df['safety_incidents'].mean()
        }
        
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_features': X.shape[1],
            'n_samples': len(X),
            'quality_correlation': self.collaboration_metrics['quality_vs_allocation_corr'],
            'safety_incident_rate': self.collaboration_metrics['safety_incident_rate']
        }
        
        self.is_trained = True
        self.feature_names = feature_names
        
        print(f"✅ Training completed!")
        print(f"   Test MAE: {test_mae:.4f}")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   Quality correlation: {self.collaboration_metrics['quality_vs_allocation_corr']:.3f}")
        print(f"   Safety incident rate: {self.collaboration_metrics['safety_incident_rate']:.3f}")
        
        return metrics
    
    def optimize_collaboration(self, human_skill, cobot_accuracy, complexity, 
                             urgency=0.5, precision_required=0.7, **kwargs):
        """
        Optimize human-cobot collaboration for a specific task.
        
        Args:
            human_skill (float): Human worker skill level (0-1)
            cobot_accuracy (float): Cobot accuracy (0-1)
            complexity (float): Task complexity (0-1)
            urgency (float): Task urgency (0-1)
            precision_required (float): Precision requirement (0-1)
            **kwargs: Additional parameters
            
        Returns:
            dict: Optimization results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before optimization")
        
        # Set default values for missing parameters
        default_values = {
            'human_fatigue': 0.3,
            'human_experience': 0.7,
            'cobot_speed': 0.85,
            'cobot_payload': 25.0,
            'workspace_layout': 0.8,
            'safety_requirements': 0.6,
            'lighting_conditions': 0.9,
            'task_type': 'assembly'
        }
        
        # Update with provided values
        task_params = {**default_values, **kwargs}
        
        # Prepare input features
        task_features = [
            complexity, urgency, precision_required, human_skill,
            task_params['human_fatigue'], task_params['human_experience'],
            cobot_accuracy, task_params['cobot_speed'], task_params['cobot_payload'],
            task_params['workspace_layout'], task_params['safety_requirements'],
            task_params['lighting_conditions']
        ]
        
        # Add task type encoding (simplified - assume 'assembly')
        task_type_features = [1, 0, 0, 0, 0]  # One-hot encoding for 'assembly'
        
        # Add engineered features
        skill_precision_ratio = human_skill / (precision_required + 0.1)
        fatigue_workload = task_params['human_fatigue'] * complexity
        cobot_efficiency = cobot_accuracy * task_params['cobot_speed']
        human_cobot_synergy = human_skill * cobot_accuracy
        
        engineered_features = [skill_precision_ratio, fatigue_workload, cobot_efficiency, human_cobot_synergy]
        
        # Combine all features
        input_features = task_features + task_type_features + engineered_features
        input_array = np.array(input_features).reshape(1, -1)
        
        # Scale features
        input_scaled = self.scaler.transform(input_array)
        
        # Predict optimal human allocation
        optimal_allocation = self.gp_model.predict(input_scaled)[0]
        allocation_uncertainty = self.gp_model.predict(input_scaled, return_std=True)[1][0]
        
        # Calculate performance predictions
        predicted_performance = self._predict_performance(
            optimal_allocation, human_skill, cobot_accuracy, complexity, **task_params
        )
        
        results = {
            'optimal_human_allocation': np.clip(optimal_allocation, 0.05, 0.95),
            'allocation_uncertainty': allocation_uncertainty,
            'cobot_allocation': 1 - np.clip(optimal_allocation, 0.05, 0.95),
            'predicted_completion_time': predicted_performance['completion_time'],
            'predicted_quality_score': predicted_performance['quality_score'],
            'predicted_safety_risk': predicted_performance['safety_risk'],
            'human_workload': predicted_performance['human_workload'],
            'cobot_utilization': predicted_performance['cobot_utilization'],
            'recommendation': self._generate_recommendation(optimal_allocation, allocation_uncertainty)
        }
        
        return results
    
    def _predict_performance(self, human_allocation, human_skill, cobot_accuracy, 
                           complexity, **kwargs):
        """
        Predict performance metrics based on allocation.
        """
        # Task completion time prediction
        completion_time = (
            20 * complexity * (1 - human_allocation * 0.3) +  # Cobot time
            15 * complexity * human_allocation * (1 - human_skill * 0.2) +  # Human time
            5 * (1 - kwargs.get('workspace_layout', 0.8)) +  # Layout overhead
            3 * (1 - kwargs.get('lighting_conditions', 0.9))  # Lighting overhead
        )
        
        # Quality score prediction
        quality_score = (
            human_allocation * human_skill * 0.6 +  # Human contribution
            (1 - human_allocation) * cobot_accuracy * 0.8 +  # Cobot contribution
            0.2 * (1 - complexity) +  # Base quality for simple tasks
            0.1 * kwargs.get('workspace_layout', 0.8) +  # Layout quality
            0.05 * kwargs.get('lighting_conditions', 0.9)  # Lighting quality
        )
        quality_score = np.clip(quality_score, 0, 1)
        
        # Safety risk prediction
        safety_risk = (
            (1 - human_allocation) * 0.5 +  # Less oversight increases risk
            (1 - kwargs.get('safety_requirements', 0.6)) * 0.3 +  # Safety requirements reduce risk
            kwargs.get('human_fatigue', 0.3) * 0.2  # Fatigue increases risk
        )
        
        # Workload and utilization
        human_workload = np.clip(
            human_allocation * complexity * (1 + kwargs.get('human_fatigue', 0.3)) * 
            (1 - human_skill * 0.3), 0, 1
        )
        
        cobot_utilization = np.clip((1 - human_allocation) * kwargs.get('cobot_speed', 0.85), 0, 1)
        
        return {
            'completion_time': completion_time,
            'quality_score': quality_score,
            'safety_risk': np.clip(safety_risk, 0, 1),
            'human_workload': human_workload,
            'cobot_utilization': cobot_utilization
        }
    
    def _generate_recommendation(self, optimal_allocation, uncertainty):
        """
        Generate human-readable recommendation.
        """
        if uncertainty > 0.2:
            return "High uncertainty - consider pilot testing with different allocations"
        elif optimal_allocation > 0.8:
            return "Task heavily favors human expertise - minimize cobot intervention"
        elif optimal_allocation < 0.2:
            return "Task ideal for automation - maximize cobot utilization"
        else:
            return "Balanced collaboration recommended - both human and cobot contribute significantly"
    
    def simulate_scenarios(self, df, n_scenarios=100):
        """
        Simulate various collaboration scenarios.
        
        Args:
            df (pd.DataFrame): Base collaboration data
            n_scenarios (int): Number of scenarios to simulate
            
        Returns:
            pd.DataFrame: Simulation results
        """
        scenarios = []
        
        for i in range(n_scenarios):
            # Random scenario parameters
            human_skill = np.random.uniform(0.5, 1.0)
            cobot_accuracy = np.random.uniform(0.8, 0.99)
            complexity = np.random.uniform(0.2, 1.0)
            urgency = np.random.uniform(0.1, 1.0)
            precision_required = np.random.uniform(0.3, 1.0)
            
            try:
                result = self.optimize_collaboration(
                    human_skill=human_skill,
                    cobot_accuracy=cobot_accuracy,
                    complexity=complexity,
                    urgency=urgency,
                    precision_required=precision_required
                )
                
                scenarios.append({
                    'scenario_id': i,
                    'human_skill': human_skill,
                    'cobot_accuracy': cobot_accuracy,
                    'complexity': complexity,
                    'urgency': urgency,
                    'precision_required': precision_required,
                    **result
                })
            except:
                continue
        
        return pd.DataFrame(scenarios)
    
    def get_collaboration_insights(self, df):
        """
        Generate insights about human-cobot collaboration patterns.
        
        Args:
            df (pd.DataFrame): Collaboration data
            
        Returns:
            dict: Key insights
        """
        insights = {
            'high_human_allocation_tasks': df[df['optimal_human_allocation'] > 0.7]['task_type'].mode().tolist(),
            'high_cobot_tasks': df[df['optimal_human_allocation'] < 0.3]['task_type'].mode().tolist(),
            'skill_impact': df.groupby('human_skill')['optimal_human_allocation'].mean().to_dict(),
            'complexity_impact': df.groupby(pd.cut(df['complexity'], bins=3))['optimal_human_allocation'].mean().to_dict(),
            'fatigue_threshold': df[df['human_fatigue'] > 0.6]['optimal_human_allocation'].mean(),
            'optimal_allocation_distribution': df['optimal_human_allocation'].describe().to_dict(),
            'quality_by_allocation': df.groupby(pd.cut(df['optimal_human_allocation'], bins=5))['quality_score'].mean().to_dict(),
            'safety_by_allocation': df.groupby(pd.cut(df['optimal_human_allocation'], bins=5))['safety_incidents'].mean().to_dict()
        }
        
        return insights
    
    def plot_optimization_dashboard(self, df, save_path=None):
        """
        Create comprehensive optimization dashboard.
        
        Args:
            df (pd.DataFrame): Collaboration data
            save_path (str): Path to save the plot
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Human Allocation vs Task Complexity', 'Skill Level Impact',
                          'Task Type Performance', 'Quality vs Allocation',
                          'Workload Distribution', 'Safety Analysis'),
            specs=[[{"type": "scatter"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. Human allocation vs complexity
        fig.add_trace(
            go.Scatter(x=df['complexity'], y=df['optimal_human_allocation'],
                      mode='markers', marker=dict(
                          color=df['quality_score'],
                          colorscale='Viridis',
                          colorbar=dict(title="Quality Score")
                      ),
                      name='Allocation vs Complexity'),
            row=1, col=1
        )
        
        # 2. Skill level impact
        skill_groups = pd.cut(df['human_skill'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        fig.add_trace(
            go.Box(x=skill_groups, y=df['optimal_human_allocation'],
                  name='Allocation by Skill'),
            row=1, col=2
        )
        
        # 3. Task type performance
        task_performance = df.groupby('task_type').agg({
            'optimal_human_allocation': 'mean',
            'quality_score': 'mean',
            'safety_incidents': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(x=task_performance['task_type'], 
                  y=task_performance['optimal_human_allocation'],
                  name='Avg Human Allocation', marker_color='lightblue'),
            row=2, col=1
        )
        
        # 4. Quality vs allocation
        fig.add_trace(
            go.Scatter(x=df['optimal_human_allocation'], y=df['quality_score'],
                      mode='markers', marker=dict(
                          color=df['complexity'],
                          colorscale='RdYlBu_r',
                          size=8
                      ),
                      name='Quality vs Allocation'),
            row=2, col=2
        )
        
        # 5. Workload distribution
        fig.add_trace(
            go.Histogram(x=df['human_workload'], nbinsx=30,
                        name='Human Workload Distribution'),
            row=3, col=1
        )
        
        # 6. Safety analysis
        allocation_bins = pd.cut(df['optimal_human_allocation'], bins=10)
        safety_by_allocation = df.groupby(allocation_bins)['safety_incidents'].mean()
        
        fig.add_trace(
            go.Scatter(x=range(len(safety_by_allocation)), y=safety_by_allocation.values,
                      mode='lines+markers', name='Safety vs Allocation',
                      line=dict(color='red', width=3)),
            row=3, col=2
        )
        
        fig.update_layout(height=1000, showlegend=True,
                         title_text="Human-Cobot Collaboration Optimization Dashboard")
        
        if save_path:
            fig.write_html(save_path)
            print(f"📊 Interactive dashboard saved to: {save_path}")
        
        fig.show()
    
    def save_model(self, filepath):
        """
        Save the trained optimization model.
        
        Args:
            filepath (str): Path to save the model
        """
        import joblib
        
        model_data = {
            'gp_model': self.gp_model,
            'scaler': self.scaler,
            'collaboration_metrics': self.collaboration_metrics,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Collaboration optimizer saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained optimization model.
        
        Args:
            filepath (str): Path to the saved model
        """
        import joblib
        
        model_data = joblib.load(filepath)
        self.gp_model = model_data['gp_model']
        self.scaler = model_data['scaler']
        self.collaboration_metrics = model_data['collaboration_metrics']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"📁 Collaboration optimizer loaded from: {filepath}")

def main():
    """
    Example usage of the Human-Cobot Collaboration Optimizer.
    """
    print("🤝 Human-Cobot Collaboration Optimizer Demo")
    print("=" * 55)
    
    # Initialize system
    optimizer = HumanCobotOptimizer()
    
    # Generate sample data
    print("\n📊 Generating collaboration data...")
    data = optimizer.generate_collaboration_data(n_samples=3000)
    
    # Train optimization model
    print("\n🔧 Training optimization model...")
    metrics = optimizer.train_optimization_model(data)
    
    # Optimize specific task
    print("\n🎯 Optimizing collaboration for specific task...")
    task_result = optimizer.optimize_collaboration(
        human_skill=0.85,
        cobot_accuracy=0.92,
        complexity=0.7,
        urgency=0.4,
        precision_required=0.8,
        human_fatigue=0.2
    )
    
    print(f"   Optimal Human Allocation: {task_result['optimal_human_allocation']:.1%}")
    print(f"   Cobot Allocation: {task_result['cobot_allocation']:.1%}")
    print(f"   Predicted Completion Time: {task_result['predicted_completion_time']:.1f} min")
    print(f"   Predicted Quality Score: {task_result['predicted_quality_score']:.3f}")
    print(f"   Recommendation: {task_result['recommendation']}")
    
    # Simulate scenarios
    print("\n🎭 Simulating collaboration scenarios...")
    scenarios = optimizer.simulate_scenarios(data, n_scenarios=200)
    
    # Generate insights
    print("\n📈 Generating collaboration insights...")
    insights = optimizer.get_collaboration_insights(data)
    
    print(f"   High human allocation tasks: {insights['high_human_allocation_tasks']}")
    print(f"   High cobot tasks: {insights['high_cobot_tasks']}")
    print(f"   Quality correlation with allocation: {insights['quality_by_allocation']}")
    
    # Create dashboard
    print("\n📊 Creating optimization dashboard...")
    optimizer.plot_optimization_dashboard(data, save_path='data/processed/collaboration_dashboard.html')
    
    # Save model
    print("\n💾 Saving optimization model...")
    optimizer.save_model('data/processed/collaboration_optimizer.pkl')
    
    print("\n✅ Human-Cobot collaboration optimization demo completed!")

if __name__ == "__main__":
    main()