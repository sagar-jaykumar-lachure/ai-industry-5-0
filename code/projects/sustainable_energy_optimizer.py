"""
Sustainable Energy Optimizer
============================

AI-driven energy consumption minimization in smart factories using
predictive analytics and optimization algorithms.

Author: MiniMax Agent
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class SustainableEnergyOptimizer:
    """
    Sustainable Energy Optimization System for Smart Factories
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the Sustainable Energy Optimizer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.energy_model = None
        self.renewable_model = None
        self.scaler = StandardScaler()
        self.optimization_results = {}
        self.is_trained = False
        
    def generate_energy_data(self, n_samples=5000, n_factories=3):
        """
        Generate synthetic factory energy consumption data.
        
        Args:
            n_samples (int): Number of data points
            n_factories (int): Number of factories
            
        Returns:
            pd.DataFrame: Energy consumption data
        """
        np.random.seed(self.random_state)
        
        # Factory base characteristics
        factory_profiles = {
            f'factory_{i}': {
                'base_consumption': np.random.uniform(80, 150),  # Base kWh
                'efficiency_rating': np.random.uniform(0.7, 0.95),  # Energy efficiency
                'renewable_capacity': np.random.uniform(20, 60),  # Solar/Wind capacity
                'production_capacity': np.random.uniform(100, 200),  # Max production units
                'operating_hours': np.random.uniform(16, 24),  # Hours per day
            } for i in range(n_factories)
        }
        
        data = []
        for i in range(n_samples):
            factory_id = np.random.randint(0, n_factories)
            factory = factory_profiles[f'factory_{factory_id}']
            
            # Time-based features
            timestamp = datetime.now() - timedelta(days=np.random.randint(0, 365))
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            is_weekend = day_of_week >= 5
            
            # Environmental conditions
            temperature = np.random.normal(22, 8)  # Ambient temperature
            humidity = np.random.uniform(30, 80)  # Relative humidity
            solar_irradiance = max(0, np.random.normal(400, 200) * np.sin((hour - 6) * np.pi / 12))  # Solar power
            wind_speed = np.random.exponential(3)  # Wind speed for renewable energy
            grid_demand = np.random.uniform(0.7, 1.3)  # Regional grid demand factor
            
            # Operational parameters
            production_rate = np.random.uniform(0.4, 1.0) * factory['production_capacity']
            machine_utilization = np.random.uniform(0.6, 1.0)
            maintenance_status = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% chance of maintenance
            
            # Energy-consuming processes
            hvac_load = max(0, (temperature - 20) * 2 + (1 - machine_utilization) * 10)
            lighting_load = 15 + 5 * (1 if hour < 8 or hour > 18 else 0)  # Higher during non-daylight
            process_heat = production_rate * 0.8 * (1 - factory['efficiency_rating'])
            motor_load = machine_utilization * 30
            
            # Calculate renewable energy generation
            solar_generation = min(
                factory['renewable_capacity'] * (solar_irradiance / 1000) * 0.8,  # 80% efficiency
                factory['renewable_capacity']
            )
            wind_generation = min(
                factory['renewable_capacity'] * 0.3 * (wind_speed / 10) ** 0.5,  # Wind power curve
                factory['renewable_capacity'] * 0.3
            )
            total_renewable = solar_generation + wind_generation
            
            # Calculate total energy consumption
            base_consumption = factory['base_consumption'] * machine_utilization
            
            # Efficiency improvements from optimization
            optimization_factor = np.random.uniform(0.85, 1.0)  # Simulate existing optimizations
            
            total_consumption = (
                base_consumption +
                hvac_load +
                lighting_load +
                process_heat +
                motor_load +
                maintenance_status * 50  # Maintenance increases consumption
            ) * optimization_factor
            
            # Grid interaction
            net_grid_consumption = max(0, total_consumption - total_renewable)
            self_consumption_ratio = min(1.0, total_renewable / total_consumption) if total_consumption > 0 else 0
            
            # Cost calculations
            grid_price = 0.12 + 0.05 * grid_demand + 0.02 * (1 if hour > 17 or hour < 7 else 0)  # Peak pricing
            renewable_credit = total_renewable * 0.08  # Renewable energy credits
            total_cost = net_grid_consumption * grid_price - renewable_credit
            
            # Sustainability metrics
            carbon_intensity = 0.5 - 0.3 * self_consumption_ratio  # Lower with more renewables
            energy_efficiency_score = factory['efficiency_rating'] * (1 - (total_consumption - base_consumption) / total_consumption)
            
            data.append({
                'sample_id': i,
                'factory_id': factory_id,
                'timestamp': timestamp,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': is_weekend,
                'temperature': temperature,
                'humidity': humidity,
                'solar_irradiance': solar_irradiance,
                'wind_speed': wind_speed,
                'grid_demand': grid_demand,
                'production_rate': production_rate,
                'machine_utilization': machine_utilization,
                'maintenance_status': maintenance_status,
                'hvac_load': hvac_load,
                'lighting_load': lighting_load,
                'process_heat': process_heat,
                'motor_load': motor_load,
                'solar_generation': solar_generation,
                'wind_generation': wind_generation,
                'total_renewable': total_renewable,
                'total_consumption': total_consumption,
                'net_grid_consumption': net_grid_consumption,
                'self_consumption_ratio': self_consumption_ratio,
                'grid_price': grid_price,
                'renewable_credit': renewable_credit,
                'total_cost': total_cost,
                'carbon_intensity': carbon_intensity,
                'energy_efficiency_score': energy_efficiency_score
            })
        
        df = pd.DataFrame(data)
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for energy prediction model.
        
        Args:
            df (pd.DataFrame): Input energy data
            
        Returns:
            tuple: (features, targets)
        """
        # Select features for energy prediction
        feature_columns = [
            'temperature', 'humidity', 'solar_irradiance', 'wind_speed', 'grid_demand',
            'production_rate', 'machine_utilization', 'maintenance_status',
            'hour', 'day_of_week', 'month', 'is_weekend'
        ]
        
        # Add factory-specific features
        factory_dummies = pd.get_dummies(df['factory_id'], prefix='factory')
        
        features = pd.concat([df[feature_columns], factory_dummies], axis=1)
        
        # Feature engineering
        features['temp_production_interaction'] = features['temperature'] * features['production_rate']
        features['solar_utilization_ratio'] = features['solar_irradiance'] / (features['production_rate'] + 1)
        features['peak_hour'] = ((features['hour'] >= 17) | (features['hour'] <= 7)).astype(int)
        features['weekend_factor'] = features['is_weekend'] * features['production_rate']
        features['renewable_potential'] = (features['solar_irradiance'] + features['wind_speed']) / 2
        
        # Targets
        energy_target = df['total_consumption']
        cost_target = df['total_cost']
        renewable_target = df['total_renewable']
        
        return features.values, energy_target.values, cost_target.values, renewable_target.values, features.columns.tolist()
    
    def train_energy_models(self, df):
        """
        Train energy consumption and optimization models.
        
        Args:
            df (pd.DataFrame): Training energy data
            
        Returns:
            dict: Training metrics
        """
        print("🔧 Preparing features...")
        X, y_energy, y_cost, y_renewable, feature_names = self.prepare_features(df)
        
        print("📊 Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print("🌲 Training energy consumption model...")
        # Train energy consumption model
        X_train, X_test, y_energy_train, y_energy_test = train_test_split(
            X_scaled, y_energy, test_size=0.2, random_state=self.random_state
        )
        
        # Grid search for best model
        models = {
            'RandomForest': RandomForestRegressor(random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1)
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            if hasattr(model, 'get_params'):
                # Simple grid search for tree-based models
                if name == 'RandomForest':
                    params = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
                elif name == 'GradientBoosting':
                    params = {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]}
                else:
                    params = {'alpha': [0.1, 1.0, 10.0]}
                
                grid_search = GridSearchCV(model, params, cv=3, scoring='neg_mean_absolute_error')
                grid_search.fit(X_train, y_energy_train)
                
                if -grid_search.best_score_ < best_score:
                    best_score = -grid_search.best_score_
                    best_model = grid_search.best_estimator_
            else:
                model.fit(X_train, y_energy_train)
                y_pred = model.predict(X_test)
                score = mean_absolute_error(y_energy_test, y_pred)
                
                if score < best_score:
                    best_score = score
                    best_model = model
        
        self.energy_model = best_model
        
        # Train renewable energy model
        print("☀️ Training renewable energy model...")
        _, _, y_renewable_train, y_renewable_test = train_test_split(
            X_scaled, y_renewable, test_size=0.2, random_state=self.random_state
        )
        
        self.renewable_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.renewable_model.fit(X_scaled, y_renewable)
        
        # Calculate metrics
        y_energy_pred = self.energy_model.predict(X_test)
        y_renewable_pred = self.renewable_model.predict(X_test)
        
        energy_metrics = {
            'mae': mean_absolute_error(y_energy_test, y_energy_pred),
            'rmse': np.sqrt(mean_squared_error(y_energy_test, y_energy_pred)),
            'r2': r2_score(y_energy_test, y_energy_pred)
        }
        
        renewable_metrics = {
            'mae': mean_absolute_error(y_renewable_test, y_renewable_pred),
            'rmse': np.sqrt(mean_squared_error(y_renewable_test, y_renewable_pred)),
            'r2': r2_score(y_renewable_test, y_renewable_pred)
        }
        
        self.is_trained = True
        self.feature_names = feature_names
        
        print(f"✅ Training completed!")
        print(f"   Energy Model - MAE: {energy_metrics['mae']:.2f} kWh, R²: {energy_metrics['r2']:.3f}")
        print(f"   Renewable Model - MAE: {renewable_metrics['mae']:.2f} kWh, R²: {renewable_metrics['r2']:.3f}")
        
        return {
            'energy_metrics': energy_metrics,
            'renewable_metrics': renewable_metrics,
            'n_features': X.shape[1],
            'n_samples': len(X)
        }
    
    def optimize_energy_consumption(self, production_params, optimization_horizon=24):
        """
        Optimize energy consumption for given production parameters.
        
        Args:
            production_params (dict): Production and operational parameters
            optimization_horizon (int): Hours to optimize ahead
            
        Returns:
            dict: Optimization results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before optimization")
        
        # Create optimization scenarios
        scenarios = []
        
        for hour_offset in range(optimization_horizon):
            # Simulate different operating conditions
            temp_adjustment = np.random.normal(0, 2)
            utilization_adjustment = np.random.uniform(-0.1, 0.1)
            
            scenario_params = production_params.copy()
            scenario_params['temperature'] += temp_adjustment
            scenario_params['machine_utilization'] += utilization_adjustment
            scenario_params['hour'] = (scenario_params.get('hour', 12) + hour_offset) % 24
            
            # Predict energy consumption
            X_scenario = self._prepare_single_scenario(scenario_params)
            X_scenario_scaled = self.scaler.transform(X_scenario)
            
            predicted_consumption = self.energy_model.predict(X_scenario_scaled)[0]
            predicted_renewable = self.renewable_model.predict(X_scenario_scaled)[0]
            
            scenarios.append({
                'hour_offset': hour_offset,
                'predicted_consumption': predicted_consumption,
                'predicted_renewable': predicted_renewable,
                'net_consumption': max(0, predicted_consumption - predicted_renewable),
                'self_consumption_ratio': min(1.0, predicted_renewable / predicted_consumption) if predicted_consumption > 0 else 0
            })
        
        # Calculate optimization opportunities
        total_consumption = sum(s['predicted_consumption'] for s in scenarios)
        total_renewable = sum(s['predicted_renewable'] for s in scenarios)
        total_net = sum(s['net_consumption'] for s in scenarios)
        
        # Identify optimization potential
        optimization_potential = {
            'peak_shaving_opportunities': self._identify_peak_shaving(scenarios),
            'renewable_utilization_boost': self._calculate_renewable_boost(scenarios),
            'efficiency_improvements': self._suggest_efficiency_improvements(scenarios),
            'demand_response_potential': self._calculate_demand_response(scenarios)
        }
        
        results = {
            'optimization_scenarios': scenarios,
            'total_consumption_kwh': total_consumption,
            'total_renewable_kwh': total_renewable,
            'total_net_consumption_kwh': total_net,
            'overall_self_consumption_ratio': total_renewable / total_consumption if total_consumption > 0 else 0,
            'optimization_potential': optimization_potential,
            'cost_savings_estimate': self._calculate_cost_savings(scenarios),
            'carbon_reduction_estimate': self._calculate_carbon_reduction(scenarios)
        }
        
        return results
    
    def _prepare_single_scenario(self, params):
        """Prepare single scenario for prediction."""
        default_params = {
            'temperature': 22,
            'humidity': 50,
            'solar_irradiance': 400,
            'wind_speed': 3,
            'grid_demand': 1.0,
            'production_rate': 100,
            'machine_utilization': 0.8,
            'maintenance_status': 0,
            'hour': 12,
            'day_of_week': 1,
            'month': 6,
            'is_weekend': 0,
            'factory_id': 0
        }
        
        # Update with provided parameters
        scenario_params = {**default_params, **params}
        
        # Create feature vector
        features = [
            scenario_params['temperature'], scenario_params['humidity'],
            scenario_params['solar_irradiance'], scenario_params['wind_speed'],
            scenario_params['grid_demand'], scenario_params['production_rate'],
            scenario_params['machine_utilization'], scenario_params['maintenance_status'],
            scenario_params['hour'], scenario_params['day_of_week'],
            scenario_params['month'], scenario_params['is_weekend']
        ]
        
        # Add factory dummy (assuming factory 0)
        features.extend([1, 0, 0])  # One-hot encoding for factory 0
        
        # Add engineered features
        features.append(scenario_params['temperature'] * scenario_params['production_rate'])
        features.append(scenario_params['solar_irradiance'] / (scenario_params['production_rate'] + 1))
        features.append(1 if scenario_params['hour'] >= 17 or scenario_params['hour'] <= 7 else 0)
        features.append(scenario_params['is_weekend'] * scenario_params['production_rate'])
        features.append((scenario_params['solar_irradiance'] + scenario_params['wind_speed']) / 2)
        
        return np.array(features).reshape(1, -1)
    
    def _identify_peak_shaving(self, scenarios):
        """Identify peak shaving opportunities."""
        consumptions = [s['predicted_consumption'] for s in scenarios]
        threshold = np.percentile(consumptions, 80)  # Top 20% as peak
        
        peak_hours = [i for i, c in enumerate(consumptions) if c > threshold]
        potential_reduction = sum(c - threshold for c in consumptions if c > threshold)
        
        return {
            'peak_hours': peak_hours,
            'potential_reduction_kwh': potential_reduction,
            'peak_threshold_kwh': threshold
        }
    
    def _calculate_renewable_boost(self, scenarios):
        """Calculate renewable energy utilization boost potential."""
        current_self_consumption = sum(s['self_consumption_ratio'] for s in scenarios) / len(scenarios)
        
        # Assume we can optimize renewable utilization by 15%
        boost_potential = current_self_consumption * 0.15
        
        return {
            'current_self_consumption': current_self_consumption,
            'boost_potential': boost_potential,
            'target_self_consumption': current_self_consumption + boost_potential
        }
    
    def _suggest_efficiency_improvements(self, scenarios):
        """Suggest energy efficiency improvements."""
        avg_consumption = sum(s['predicted_consumption'] for s in scenarios) / len(scenarios)
        
        # Typical efficiency improvements
        improvements = {
            'hvac_optimization': 0.08,  # 8% reduction
            'motor_efficiency': 0.06,   # 6% reduction
            'lighting_upgrade': 0.04,   # 4% reduction
            'process_optimization': 0.10, # 10% reduction
            'maintenance_impact': 0.05  # 5% reduction
        }
        
        total_efficiency_potential = sum(improvements.values())
        total_savings_kwh = avg_consumption * total_efficiency_potential
        
        return {
            'improvement_categories': improvements,
            'total_efficiency_potential': total_efficiency_potential,
            'total_savings_kwh': total_savings_kwh
        }
    
    def _calculate_demand_response(self, scenarios):
        """Calculate demand response potential."""
        # Demand response can shift 10-20% of peak consumption
        consumptions = [s['predicted_consumption'] for s in scenarios]
        peak_consumption = max(consumptions)
        peak_hour = consumptions.index(peak_consumption)
        
        # Assume we can shift 15% of peak to off-peak hours
        shiftable_amount = peak_consumption * 0.15
        
        return {
            'peak_hour': peak_hour,
            'peak_consumption_kwh': peak_consumption,
            'shiftable_amount_kwh': shiftable_amount,
            'potential_savings': shiftable_amount * 0.03  # Assume 3 cents/kWh peak-off-peak difference
        }
    
    def _calculate_cost_savings(self, scenarios):
        """Calculate potential cost savings."""
        # Simplified cost calculation
        base_cost_per_kwh = 0.12
        peak_premium = 0.05
        
        total_savings = 0
        for scenario in scenarios:
            hour = scenario['hour_offset'] % 24
            if hour >= 17 or hour <= 7:  # Peak hours
                total_savings += scenario['net_consumption'] * peak_premium
        
        # Add efficiency savings
        avg_consumption = sum(s['predicted_consumption'] for s in scenarios) / len(scenarios)
        efficiency_savings = avg_consumption * 0.25 * base_cost_per_kwh  # 25% efficiency gain
        
        return {
            'peak_pricing_savings': total_savings,
            'efficiency_savings': efficiency_savings,
            'total_annual_savings': (total_savings + efficiency_savings) * 365
        }
    
    def _calculate_carbon_reduction(self, scenarios):
        """Calculate potential carbon footprint reduction."""
        # Assume grid carbon intensity of 0.5 kg CO2/kWh
        grid_carbon_intensity = 0.5
        
        total_reduction = 0
        for scenario in scenarios:
            # Reduction from renewable increase
            renewable_increase = scenario['predicted_renewable'] * 0.1  # 10% increase
            carbon_reduction_renewable = renewable_increase * grid_carbon_intensity
            
            # Reduction from efficiency
            efficiency_reduction = scenario['predicted_consumption'] * 0.25 * 0.5  # 25% efficiency, 50% carbon intensity reduction
            carbon_reduction_efficiency = efficiency_reduction * grid_carbon_intensity
            
            total_reduction += carbon_reduction_renewable + carbon_reduction_efficiency
        
        return {
            'hourly_reduction_kg_co2': total_reduction,
            'annual_reduction_kg_co2': total_reduction * 365,
            'equivalent_trees': (total_reduction * 365) / 21.8  # 1 tree absorbs ~21.8 kg CO2/year
        }
    
    def get_energy_insights(self, df):
        """
        Generate comprehensive energy insights.
        
        Args:
            df (pd.DataFrame): Energy consumption data
            
        Returns:
            dict: Key energy insights
        """
        insights = {
            'consumption_patterns': {
                'peak_consumption_hour': df.groupby('hour')['total_consumption'].mean().idxmax(),
                'low_consumption_hour': df.groupby('hour')['total_consumption'].mean().idxmin(),
                'weekend_vs_weekday': df.groupby('is_weekend')['total_consumption'].mean().to_dict(),
                'seasonal_variation': df.groupby('month')['total_consumption'].mean().to_dict()
            },
            'renewable_performance': {
                'avg_self_consumption': df['self_consumption_ratio'].mean(),
                'renewable_utilization': (df['total_renewable'] / (df['total_renewable'] + df['net_grid_consumption'])).mean(),
                'peak_renewable_generation': df['total_renewable'].max(),
                'renewable_by_weather': df.groupby(pd.cut(df['solar_irradiance'], bins=5))['total_renewable'].mean().to_dict()
            },
            'efficiency_metrics': {
                'energy_per_production_unit': (df['total_consumption'] / df['production_rate']).mean(),
                'efficiency_score_distribution': df['energy_efficiency_score'].describe().to_dict(),
                'machine_utilization_impact': df.groupby(pd.cut(df['machine_utilization'], bins=5))['total_consumption'].mean().to_dict(),
                'maintenance_impact': df.groupby('maintenance_status')['total_consumption'].mean().to_dict()
            },
            'cost_analysis': {
                'avg_cost_per_kwh': (df['total_cost'] / df['total_consumption']).mean(),
                'cost_by_hour': df.groupby('hour')['total_cost'].mean().to_dict(),
                'renewable_savings': df['renewable_credit'].sum(),
                'peak_vs_offpeak_cost': df[df['hour'].isin([2, 3, 4, 5])]['total_cost'].mean() / df[df['hour'].isin([14, 15, 16, 17])]['total_cost'].mean()
            }
        }
        
        return insights
    
    def plot_energy_dashboard(self, df, save_path=None):
        """
        Create comprehensive energy optimization dashboard.
        
        Args:
            df (pd.DataFrame): Energy data
            save_path (str): Path to save the plot
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Energy Consumption Patterns', 'Renewable vs Grid Energy',
                          'Efficiency Metrics', 'Cost Analysis',
                          'Production vs Energy', 'Carbon Footprint'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Energy consumption patterns over time
        hourly_consumption = df.groupby('hour')['total_consumption'].mean()
        fig.add_trace(
            go.Scatter(x=hourly_consumption.index, y=hourly_consumption.values,
                      mode='lines+markers', name='Avg Consumption',
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # 2. Renewable vs Grid energy
        fig.add_trace(
            go.Scatter(x=df['total_renewable'], y=df['net_grid_consumption'],
                      mode='markers', marker=dict(
                          color=df['self_consumption_ratio'],
                          colorscale='Greens',
                          colorbar=dict(title="Self-Consumption Ratio")
                      ),
                      name='Renewable vs Grid'),
            row=1, col=2
        )
        
        # 3. Efficiency metrics by factory
        factory_efficiency = df.groupby('factory_id')['energy_efficiency_score'].mean()
        fig.add_trace(
            go.Bar(x=[f'Factory {i}' for i in factory_efficiency.index], 
                  y=factory_efficiency.values,
                  name='Efficiency Score', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # 4. Cost analysis
        fig.add_trace(
            go.Scatter(x=df['production_rate'], y=df['total_cost'],
                      mode='markers', marker=dict(
                          color=df['hour'],
                          colorscale='Viridis',
                          colorbar=dict(title="Hour")
                      ),
                      name='Cost vs Production'),
            row=2, col=2
        )
        
        # 5. Production vs Energy correlation
        fig.add_trace(
            go.Scatter(x=df['machine_utilization'], y=df['total_consumption'],
                      mode='markers', marker=dict(size=6, opacity=0.6),
                      name='Utilization vs Consumption'),
            row=3, col=1
        )
        
        # 6. Carbon footprint by factory
        factory_carbon = df.groupby('factory_id')['carbon_intensity'].mean()
        fig.add_trace(
            go.Bar(x=[f'Factory {i}' for i in factory_carbon.index],
                  y=factory_carbon.values,
                  name='Carbon Intensity', marker_color='orange'),
            row=3, col=2
        )
        
        fig.update_layout(height=1000, showlegend=True,
                         title_text="Sustainable Energy Optimization Dashboard")
        
        if save_path:
            fig.write_html(save_path)
            print(f"📊 Interactive dashboard saved to: {save_path}")
        
        fig.show()
    
    def save_models(self, filepath_prefix):
        """
        Save the trained models.
        
        Args:
            filepath_prefix (str): Prefix for model file paths
        """
        import joblib
        
        energy_model_data = {
            'energy_model': self.energy_model,
            'renewable_model': self.renewable_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'optimization_results': self.optimization_results,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        joblib.dump(energy_model_data, f"{filepath_prefix}_energy_models.pkl")
        print(f"💾 Energy models saved to: {filepath_prefix}_energy_models.pkl")
    
    def load_models(self, filepath_prefix):
        """
        Load the trained models.
        
        Args:
            filepath_prefix (str): Prefix for model file paths
        """
        import joblib
        
        model_data = joblib.load(f"{filepath_prefix}_energy_models.pkl")
        self.energy_model = model_data['energy_model']
        self.renewable_model = model_data['renewable_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.optimization_results = model_data['optimization_results']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"📁 Energy models loaded from: {filepath_prefix}_energy_models.pkl")

def main():
    """
    Example usage of the Sustainable Energy Optimizer.
    """
    print("⚡ Sustainable Energy Optimizer Demo")
    print("=" * 45)
    
    # Initialize system
    energy_optimizer = SustainableEnergyOptimizer()
    
    # Generate sample data
    print("\n📊 Generating energy consumption data...")
    data = energy_optimizer.generate_energy_data(n_samples=3000, n_factories=3)
    
    # Train models
    print("\n🔧 Training energy models...")
    metrics = energy_optimizer.train_energy_models(data)
    
    # Optimize energy consumption
    print("\n🎯 Optimizing energy consumption...")
    production_params = {
        'production_rate': 120,
        'machine_utilization': 0.85,
        'temperature': 20,
        'solar_irradiance': 500,
        'factory_id': 0
    }
    
    optimization_results = energy_optimizer.optimize_energy_consumption(production_params)
    
    print(f"   Total Consumption: {optimization_results['total_consumption_kwh']:.1f} kWh")
    print(f"   Total Renewable: {optimization_results['total_renewable_kwh']:.1f} kWh")
    print(f"   Self-Consumption Ratio: {optimization_results['overall_self_consumption_ratio']:.1%}")
    print(f"   Annual Cost Savings: ${optimization_results['cost_savings_estimate']['total_annual_savings']:.0f}")
    print(f"   Annual CO2 Reduction: {optimization_results['carbon_reduction_estimate']['annual_reduction_kg_co2']:.0f} kg")
    
    # Generate insights
    print("\n📈 Generating energy insights...")
    insights = energy_optimizer.get_energy_insights(data)
    
    peak_hour = insights['consumption_patterns']['peak_consumption_hour']
    avg_self_consumption = insights['renewable_performance']['avg_self_consumption']
    
    print(f"   Peak consumption hour: {peak_hour}:00")
    print(f"   Average self-consumption: {avg_self_consumption:.1%}")
    print(f"   Energy per production unit: {insights['efficiency_metrics']['energy_per_production_unit']:.2f} kWh/unit")
    
    # Create dashboard
    print("\n📊 Creating energy dashboard...")
    energy_optimizer.plot_energy_dashboard(data, save_path='data/processed/energy_dashboard.html')
    
    # Save models
    print("\n💾 Saving energy models...")
    energy_optimizer.save_models('data/processed/energy_optimizer')
    
    print("\n✅ Sustainable Energy Optimizer demo completed!")

if __name__ == "__main__":
    main()