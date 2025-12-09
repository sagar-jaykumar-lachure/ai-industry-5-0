"""
Digital Twin for Manufacturing Line
===================================

Production line simulation with real-time anomaly detection using
Isolation Forest and statistical process control.


"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DigitalTwin:
    """
    Digital Twin System for Manufacturing Line Monitoring
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize the Digital Twin system.
        
        Args:
            contamination (float): Expected proportion of anomalies
            random_state (int): Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = IsolationForest(
            contamination=contamination, 
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.is_trained = False
        self.baseline_metrics = {}
        
    def generate_production_data(self, n_samples=5000, n_machines=4, anomaly_rate=0.05):
        """
        Generate synthetic production line data.
        
        Args:
            n_samples (int): Number of data points to generate
            n_machines (int): Number of machines in production line
            anomaly_rate (float): Proportion of anomalous samples
            
        Returns:
            pd.DataFrame: Generated production data
        """
        np.random.seed(self.random_state)
        
        # Machine baseline characteristics
        machine_bases = {
            f'machine_{i}': {
                'cycle_time': np.random.normal(25, 3),
                'defect_rate': np.random.exponential(0.02),
                'throughput': np.random.normal(120, 15),
                'power_consumption': np.random.normal(45, 8),
                'temperature': np.random.normal(35, 5)
            } for i in range(n_machines)
        }
        
        data = []
        for i in range(n_samples):
            timestamp = datetime.now() - timedelta(minutes=(n_samples-i))
            
            # Generate normal operation data
            sample = {
                'timestamp': timestamp,
                'sample_id': i,
                'ambient_temp': np.random.normal(22, 2),
                'humidity': np.random.uniform(40, 70),
                'pressure': np.random.normal(1013, 5),
            }
            
            # Add machine-specific data
            for machine_id, base in machine_bases.items():
                sample[f'{machine_id}_cycle_time'] = np.random.normal(
                    base['cycle_time'], base['cycle_time'] * 0.1
                )
                sample[f'{machine_id}_defect_rate'] = np.random.exponential(
                    base['defect_rate']
                )
                sample[f'{machine_id}_throughput'] = np.random.normal(
                    base['throughput'], base['throughput'] * 0.1
                )
                sample[f'{machine_id}_power'] = np.random.normal(
                    base['power_consumption'], base['power_consumption'] * 0.15
                )
                sample[f'{machine_id}_temp'] = np.random.normal(
                    base['temperature'], base['temperature'] * 0.1
                )
            
            # Calculate derived metrics
            cycle_times = [sample[f'machine_{j}_cycle_time'] for j in range(n_machines)]
            defect_rates = [sample[f'machine_{j}_defect_rate'] for j in range(n_machines)]
            throughputs = [sample[f'machine_{j}_throughput'] for j in range(n_machines)]
            
            sample.update({
                'line_cycle_time': np.mean(cycle_times),
                'line_defect_rate': np.mean(defect_rates),
                'line_throughput': np.mean(throughputs),
                'line_efficiency': np.mean(throughputs) / (np.mean(cycle_times) + 1),
                'bottleneck_machine': np.argmax(cycle_times),
                'max_cycle_time': np.max(cycle_times),
                'defect_variance': np.var(defect_rates)
            })
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        # Inject anomalies
        n_anomalies = int(n_samples * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['machine_failure', 'quality_issue', 'performance_degradation'])
            
            if anomaly_type == 'machine_failure':
                failed_machine = np.random.randint(0, n_machines)
                df.loc[idx, f'machine_{failed_machine}_cycle_time'] *= np.random.uniform(1.5, 2.5)
                df.loc[idx, f'machine_{failed_machine}_defect_rate'] *= np.random.uniform(2, 5)
                df.loc[idx, f'machine_{failed_machine}_throughput'] *= np.random.uniform(0.3, 0.7)
                
            elif anomaly_type == 'quality_issue':
                # Increase defect rates across all machines
                for j in range(n_machines):
                    df.loc[idx, f'machine_{j}_defect_rate'] *= np.random.uniform(2, 4)
                    
            elif anomaly_type == 'performance_degradation':
                # General performance degradation
                df.loc[idx, 'line_cycle_time'] *= np.random.uniform(1.2, 1.8)
                df.loc[idx, 'line_throughput'] *= np.random.uniform(0.6, 0.8)
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for anomaly detection.
        
        Args:
            df (pd.DataFrame): Input production data
            
        Returns:
            np.array: Prepared features
        """
        # Select relevant features for anomaly detection
        feature_columns = [
            'line_cycle_time', 'line_defect_rate', 'line_throughput', 
            'line_efficiency', 'max_cycle_time', 'defect_variance'
        ]
        
        # Add machine-specific features
        machine_features = []
        for col in df.columns:
            if any(machine in col for machine in ['machine_']) and any(metric in col for metric in ['cycle_time', 'defect_rate', 'throughput']):
                machine_features.append(col)
        
        feature_columns.extend(machine_features)
        
        features = df[feature_columns].copy()
        
        # Feature engineering
        features['temp_ambient_ratio'] = features['line_throughput'] / (df['ambient_temp'] + 1)
        features['efficiency_normalized'] = features['line_efficiency'] / features['line_throughput']
        
        return features.values
    
    def train_baseline(self, df):
        """
        Train the digital twin baseline model.
        
        Args:
            df (pd.DataFrame): Training production data
            
        Returns:
            dict: Training metrics
        """
        print("🔧 Preparing features...")
        X = self.prepare_features(df)
        
        print("📊 Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print("🌲 Training Isolation Forest...")
        anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
        
        print("🎯 Clustering normal operation patterns...")
        normal_indices = anomaly_labels == 1
        X_normal = X_scaled[normal_indices]
        if len(X_normal) > 0:
            cluster_labels = self.dbscan.fit_predict(X_normal)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        else:
            n_clusters = 0
        
        # Calculate baseline metrics
        normal_data = df[anomaly_labels == 1]
        if len(normal_data) > 0:
            self.baseline_metrics = {
                'mean_cycle_time': normal_data['line_cycle_time'].mean(),
                'std_cycle_time': normal_data['line_cycle_time'].std(),
                'mean_defect_rate': normal_data['line_defect_rate'].mean(),
                'mean_throughput': normal_data['line_throughput'].mean(),
                'mean_efficiency': normal_data['line_efficiency'].mean(),
            }
        
        # Anomaly statistics
        n_anomalies = sum(anomaly_labels == -1)
        anomaly_rate = n_anomalies / len(df)
        
        metrics = {
            'n_samples': len(df),
            'n_features': X.shape[1],
            'n_anomalies': n_anomalies,
            'anomaly_rate': anomaly_rate,
            'expected_anomaly_rate': self.contamination,
            'n_clusters': n_clusters,
            'baseline_established': len(self.baseline_metrics) > 0
        }
        
        self.is_trained = True
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        print(f"✅ Training completed!")
        print(f"   Anomalies detected: {n_anomalies} ({anomaly_rate:.1%})")
        print(f"   Normal operation clusters: {n_clusters}")
        print(f"   Baseline metrics established: {len(self.baseline_metrics) > 0}")
        
        return metrics
    
    def detect_anomalies(self, df, return_scores=False):
        """
        Detect anomalies in new production data.
        
        Args:
            df (pd.DataFrame): Production data for analysis
            return_scores (bool): Whether to return anomaly scores
            
        Returns:
            pd.DataFrame: Data with anomaly predictions and scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before anomaly detection")
            
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest predictions
        anomaly_labels = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        # Statistical process control
        df_result = df.copy()
        df_result['anomaly'] = anomaly_labels
        df_result['anomaly_score'] = anomaly_scores
        
        # Add anomaly categories
        df_result['anomaly_type'] = 'Normal'
        
        # Categorize anomalies based on deviation from baseline
        if self.baseline_metrics:
            cycle_time_threshold = self.baseline_metrics['mean_cycle_time'] + 2 * self.baseline_metrics['std_cycle_time']
            
            anomaly_mask = anomaly_labels == -1
            if anomaly_mask.any():
                # Machine failure detection
                machine_cycle_times = [col for col in df.columns if 'machine_' in col and 'cycle_time' in col]
                for col in machine_cycle_times:
                    if col in df.columns:
                        machine_threshold = df[col].mean() + 2 * df[col].std()
                        machine_failure_mask = (anomaly_mask) & (df[col] > machine_threshold)
                        df_result.loc[machine_failure_mask, 'anomaly_type'] = 'Machine Failure'
                
                # Quality issue detection
                quality_mask = (anomaly_mask) & (df_result['line_defect_rate'] > 2 * self.baseline_metrics['mean_defect_rate'])
                df_result.loc[quality_mask, 'anomaly_type'] = 'Quality Issue'
                
                # Performance degradation
                performance_mask = (anomaly_mask) & (df_result['line_cycle_time'] > cycle_time_threshold)
                df_result.loc[performance_mask, 'anomaly_type'] = 'Performance Degradation'
        
        # Risk assessment
        df_result['risk_level'] = 'Low'
        df_result.loc[df_result['anomaly_score'] < -0.1, 'risk_level'] = 'Medium'
        df_result.loc[df_result['anomaly_score'] < -0.2, 'risk_level'] = 'High'
        df_result.loc[df_result['anomaly_score'] < -0.3, 'risk_level'] = 'Critical'
        
        return df_result
    
    def simulate_scenario(self, df, scenario_type, intensity=1.0):
        """
        Simulate different operational scenarios.
        
        Args:
            df (pd.DataFrame): Base production data
            scenario_type (str): Type of scenario ('machine_maintenance', 'quality_boost', 'peak_load')
            intensity (float): Intensity of the scenario effect
            
        Returns:
            pd.DataFrame: Data with simulated scenario
        """
        df_scenario = df.copy()
        
        if scenario_type == 'machine_maintenance':
            # Simulate machine under maintenance
            machine_idx = np.random.randint(0, 4)  # Assume 4 machines
            machine_col = f'machine_{machine_idx}'
            
            df_scenario[f'{machine_col}_cycle_time'] *= (1 + intensity * 0.5)
            df_scenario[f'{machine_col}_throughput'] *= (1 - intensity * 0.4)
            df_scenario[f'{machine_col}_defect_rate'] *= (1 + intensity * 0.3)
            
        elif scenario_type == 'quality_boost':
            # Simulate quality improvement
            for machine_idx in range(4):
                machine_col = f'machine_{machine_idx}'
                df_scenario[f'{machine_col}_defect_rate'] *= (1 - intensity * 0.6)
                
        elif scenario_type == 'peak_load':
            # Simulate peak production load
            for col in df_scenario.columns:
                if 'cycle_time' in col:
                    df_scenario[col] *= (1 + intensity * 0.3)
                elif 'throughput' in col:
                    df_scenario[col] *= (1 + intensity * 0.2)
        
        # Recalculate derived metrics
        cycle_times = [df_scenario[f'machine_{i}_cycle_time'].iloc[j] 
                      for i in range(4) for j in range(len(df_scenario))]
        # ... (simplified recalculation)
        
        return df_scenario
    
    def get_performance_metrics(self, df):
        """
        Calculate comprehensive performance metrics.
        
        Args:
            df (pd.DataFrame): Production data
            
        Returns:
            dict: Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        metrics = {
            'avg_cycle_time': df['line_cycle_time'].mean(),
            'avg_defect_rate': df['line_defect_rate'].mean(),
            'avg_throughput': df['line_throughput'].mean(),
            'avg_efficiency': df['line_efficiency'].mean(),
            'anomaly_rate': (df['anomaly'] == -1).mean(),
            'critical_anomalies': (df['risk_level'] == 'Critical').sum(),
            'uptime_percentage': (1 - (df['anomaly'] == -1).mean()) * 100
        }
        
        # Compare to baseline
        if self.baseline_metrics:
            metrics['cycle_time_vs_baseline'] = (
                (metrics['avg_cycle_time'] - self.baseline_metrics['mean_cycle_time']) / 
                self.baseline_metrics['mean_cycle_time']
            ) * 100
            metrics['efficiency_vs_baseline'] = (
                (metrics['avg_efficiency'] - self.baseline_metrics['mean_efficiency']) / 
                self.baseline_metrics['mean_efficiency']
            ) * 100
        
        return metrics
    
    def plot_dashboard(self, df, save_path=None):
        """
        Create comprehensive dashboard visualization.
        
        Args:
            df (pd.DataFrame): Production data with anomaly detection
            save_path (str): Path to save the plot
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Production Metrics Over Time', 'Anomaly Distribution',
                          'Machine Performance Comparison', 'Defect Rate Analysis',
                          'Throughput vs Cycle Time', 'Anomaly Types'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # 1. Production metrics over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['line_cycle_time'], 
                      name='Cycle Time', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['line_throughput'], 
                      name='Throughput', yaxis='y2', line=dict(color='green')),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Anomaly distribution pie chart
        anomaly_counts = df['anomaly'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Normal', 'Anomaly'], values=anomaly_counts.values,
                  name="Anomaly Distribution"),
            row=1, col=2
        )
        
        # 3. Machine performance comparison
        machine_cols = [col for col in df.columns if 'machine_' in col and 'throughput' in col]
        machine_data = []
        for col in machine_cols:
            machine_id = col.split('_')[1]
            machine_data.append({
                'Machine': f'Machine {machine_id}',
                'Avg Throughput': df[col].mean(),
                'Avg Defect Rate': df[col.replace('throughput', 'defect_rate')].mean()
            })
        
        machine_df = pd.DataFrame(machine_data)
        fig.add_trace(
            go.Bar(x=machine_df['Machine'], y=machine_df['Avg Throughput'],
                  name='Avg Throughput', marker_color='lightblue'),
            row=2, col=1
        )
        
        # 4. Defect rate analysis
        fig.add_trace(
            go.Scatter(x=df['line_cycle_time'], y=df['line_defect_rate'],
                      mode='markers', marker=dict(
                          color=df['anomaly_score'],
                          colorscale='RdYlBu_r',
                          colorbar=dict(title="Anomaly Score")
                      ),
                      name='Defect Rate vs Cycle Time'),
            row=2, col=2
        )
        
        # 5. Throughput vs Cycle Time with anomalies
        normal_data = df[df['anomaly'] == 1]
        anomaly_data = df[df['anomaly'] == -1]
        
        fig.add_trace(
            go.Scatter(x=normal_data['line_cycle_time'], y=normal_data['line_throughput'],
                      mode='markers', name='Normal Operations', 
                      marker=dict(color='green', size=5, opacity=0.6)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=anomaly_data['line_cycle_time'], y=anomaly_data['line_throughput'],
                      mode='markers', name='Anomalies', 
                      marker=dict(color='red', size=8, opacity=0.8)),
            row=3, col=1
        )
        
        # 6. Anomaly types pie chart
        if 'anomaly_type' in df.columns:
            anomaly_type_counts = df[df['anomaly'] == -1]['anomaly_type'].value_counts()
            fig.add_trace(
                go.Pie(labels=anomaly_type_counts.index, values=anomaly_type_counts.values,
                      name="Anomaly Types"),
                row=3, col=2
            )
        
        fig.update_layout(height=900, showlegend=True, 
                         title_text="Digital Twin Manufacturing Dashboard")
        
        if save_path:
            fig.write_html(save_path)
            print(f"📊 Interactive dashboard saved to: {save_path}")
        
        fig.show()
    
    def save_model(self, filepath):
        """
        Save the trained digital twin model.
        
        Args:
            filepath (str): Path to save the model
        """
        import joblib
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'dbscan': self.dbscan,
            'baseline_metrics': self.baseline_metrics,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Digital Twin model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained digital twin model.
        
        Args:
            filepath (str): Path to the saved model
        """
        import joblib
        
        model_data = joblib.load(filepath)
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.dbscan = model_data['dbscan']
        self.baseline_metrics = model_data['baseline_metrics']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"📁 Digital Twin model loaded from: {filepath}")

def main():
    """
    Example usage of the Digital Twin system.
    """
    print("🏭 Digital Twin Manufacturing Line Demo")
    print("=" * 50)
    
    # Initialize system
    digital_twin = DigitalTwin(contamination=0.05)
    
    # Generate sample data
    print("\n📊 Generating production line data...")
    data = digital_twin.generate_production_data(n_samples=3000, n_machines=4, anomaly_rate=0.08)
    
    # Train baseline model
    print("\n🔧 Training digital twin baseline...")
    metrics = digital_twin.train_baseline(data)
    
    # Detect anomalies
    print("\n🔍 Detecting anomalies...")
    results = digital_twin.detect_anomalies(data[-500:])  # Last 500 samples
    
    anomaly_count = sum(results['anomaly'] == -1)
    print(f"⚠️  Detected {anomaly_count} anomalies in test data ({anomaly_count/len(results):.1%})")
    
    # Simulate scenario
    print("\n🎭 Simulating maintenance scenario...")
    scenario_data = digital_twin.simulate_scenario(data[-100:], 'machine_maintenance', intensity=1.0)
    scenario_results = digital_twin.detect_anomalies(scenario_data)
    
    # Get performance metrics
    print("\n📈 Calculating performance metrics...")
    performance = digital_twin.get_performance_metrics(results)
    
    print(f"   Average Cycle Time: {performance['avg_cycle_time']:.2f}s")
    print(f"   Average Throughput: {performance['avg_throughput']:.1f} units/hr")
    print(f"   Uptime: {performance['uptime_percentage']:.1f}%")
    print(f"   Anomaly Rate: {performance['anomaly_rate']:.1%}")
    
    # Create dashboard
    print("\n📊 Creating interactive dashboard...")
    digital_twin.plot_dashboard(results, save_path='data/processed/digital_twin_dashboard.html')
    
    # Save model
    print("\n💾 Saving digital twin model...")
    digital_twin.save_model('data/processed/digital_twin_model.pkl')
    
    print("\n✅ Digital Twin demo completed successfully!")

if __name__ == "__main__":
    main()
