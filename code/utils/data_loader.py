"""
Data Loading Utilities
======================

Utility functions for loading and preprocessing data across all Industry 5.0 modules.

Author: MiniMax Agent
Date: December 2025
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Centralized data loading and preprocessing utilities for Industry 5.0 systems.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir (str): Path to data directory
        """
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, "raw_data")
        self.datasets_dir = os.path.join(data_dir, "datasets")
        self.processed_data_dir = os.path.join(data_dir, "processed_data")
        
        # Ensure directories exist
        for dir_path in [self.raw_data_dir, self.datasets_dir, self.processed_data_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_sensor_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load manufacturing sensor data.
        
        Args:
            filename (str, optional): Specific filename to load
            
        Returns:
            pd.DataFrame: Sensor data with standardized columns
        """
        if filename is None:
            filename = "sample_sensor_data.csv"
        
        filepath = os.path.join(self.datasets_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Sensor data file not found: {filepath}")
            return self._generate_sample_sensor_data()
        
        df = pd.read_csv(filepath)
        
        # Standardize column names
        column_mapping = {
            'timestamp': 'timestamp',
            'equipment_id': 'equipment_id',
            'vibration': 'vibration',
            'temperature': 'temperature',
            'pressure': 'pressure',
            'runtime_hours': 'runtime_hours',
            'maintenance_count': 'maintenance_count',
            'operating_load': 'operating_load',
            'ambient_temp': 'ambient_temp',
            'humidity': 'humidity'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded sensor data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_production_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load production line data.
        
        Args:
            filename (str, optional): Specific filename to load
            
        Returns:
            pd.DataFrame: Production data with standardized columns
        """
        if filename is None:
            filename = "production_metrics.csv"
        
        filepath = os.path.join(self.datasets_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Production data file not found: {filepath}")
            return self._generate_sample_production_data()
        
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded production data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_energy_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load energy consumption data.
        
        Args:
            filename (str, optional): Specific filename to load
            
        Returns:
            pd.DataFrame: Energy data with standardized columns
        """
        if filename is None:
            filename = "energy_consumption.csv"
        
        filepath = os.path.join(self.datasets_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Energy data file not found: {filepath}")
            return self._generate_sample_energy_data()
        
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded energy data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_collaboration_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load human-cobot collaboration data.
        
        Args:
            filename (str, optional): Specific filename to load
            
        Returns:
            pd.DataFrame: Collaboration data with standardized columns
        """
        if filename is None:
            # Generate collaboration data if not available
            return self._generate_sample_collaboration_data()
        
        filepath = os.path.join(self.datasets_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Collaboration data file not found: {filepath}")
            return self._generate_sample_collaboration_data()
        
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded collaboration data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str, category: str = "datasets") -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Output filename
            category (str): Data category ('raw_data', 'datasets', 'processed_data')
            
        Returns:
            str: Path to saved file
        """
        if category == "raw_data":
            save_dir = self.raw_data_dir
        elif category == "processed_data":
            save_dir = self.processed_data_dir
        else:
            save_dir = self.datasets_dir
        
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to: {filepath}")
        return filepath
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data summary.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            dict: Data summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Add categorical column statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['categorical_summary'] = {}
            for col in categorical_cols:
                summary['categorical_summary'][col] = df[col].value_counts().head(10).to_dict()
        
        return summary
    
    def _generate_sample_sensor_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate sample sensor data for testing."""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            data.append({
                'timestamp': pd.Timestamp('2025-12-01') + pd.Timedelta(minutes=i*15),
                'equipment_id': f'EQUIP_{i % 5:02d}_{i:04d}',
                'vibration': np.random.normal(2.5, 0.5),
                'temperature': np.random.normal(75, 5),
                'pressure': np.random.normal(100, 10),
                'runtime_hours': np.random.randint(0, 5000),
                'maintenance_count': np.random.randint(0, 10),
                'operating_load': np.random.uniform(0.6, 1.0),
                'ambient_temp': np.random.normal(22, 3),
                'humidity': np.random.uniform(30, 80)
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_production_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate sample production data for testing."""
        np.random.seed(42)
        
        data = []
        machines = ['M1', 'M2', 'M3', 'M4']
        
        for i in range(n_samples):
            machine_id = machines[i % len(machines)]
            data.append({
                'timestamp': pd.Timestamp('2025-12-01') + pd.Timedelta(minutes=i*15),
                'machine_id': machine_id,
                'cycle_time': np.random.normal(25, 3),
                'defect_rate': np.random.exponential(0.02),
                'throughput': np.random.normal(120, 15),
                'power_consumption': np.random.normal(45, 8),
                'temperature': np.random.normal(35, 5),
                'status': 'Running' if np.random.random() > 0.05 else 'Maintenance',
                'quality_score': np.random.normal(93, 5),
                'efficiency': np.random.uniform(0.8, 0.95)
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_energy_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate sample energy data for testing."""
        np.random.seed(42)
        
        data = []
        factories = ['F1', 'F2', 'F3']
        
        for i in range(n_samples):
            factory_id = factories[i % len(factories)]
            data.append({
                'timestamp': pd.Timestamp('2025-12-01') + pd.Timedelta(minutes=i*15),
                'factory_id': factory_id,
                'production_rate': np.random.uniform(80, 180),
                'machine_utilization': np.random.uniform(0.6, 1.0),
                'temperature': np.random.normal(22, 3),
                'solar_irradiance': max(0, np.random.normal(400, 200)),
                'wind_speed': np.random.exponential(3),
                'total_consumption': np.random.normal(150, 30),
                'solar_generation': np.random.uniform(20, 40),
                'wind_generation': np.random.uniform(5, 15),
                'total_renewable': np.random.uniform(25, 55),
                'net_grid_consumption': np.random.uniform(90, 130),
                'self_consumption_ratio': np.random.uniform(0.2, 0.4),
                'total_cost': np.random.uniform(15, 25),
                'carbon_intensity': np.random.uniform(0.4, 0.5)
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_collaboration_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate sample collaboration data for testing."""
        np.random.seed(42)
        
        data = []
        task_types = ['assembly', 'quality_control', 'packaging', 'material_handling', 'inspection']
        
        for i in range(n_samples):
            data.append({
                'timestamp': pd.Timestamp('2025-12-01') + pd.Timedelta(minutes=i*30),
                'task_id': f'TASK_{i:04d}',
                'task_type': np.random.choice(task_types),
                'complexity': np.random.uniform(0.3, 1.0),
                'urgency': np.random.uniform(0.1, 1.0),
                'precision_required': np.random.uniform(0.4, 1.0),
                'human_skill': np.random.choice([0.6, 0.75, 0.9], p=[0.3, 0.5, 0.2]),
                'human_fatigue': np.random.uniform(0.0, 0.8),
                'human_experience': np.random.uniform(0.2, 1.0),
                'cobot_accuracy': np.random.uniform(0.85, 0.98),
                'cobot_speed': np.random.uniform(0.7, 1.0),
                'cobot_payload': np.random.uniform(5, 50),
                'workspace_layout': np.random.uniform(0.5, 1.0),
                'safety_requirements': np.random.uniform(0.3, 1.0),
                'lighting_conditions': np.random.uniform(0.6, 1.0),
                'optimal_human_allocation': np.random.uniform(0.1, 0.9),
                'task_completion_time': np.random.uniform(15, 45),
                'quality_score': np.random.uniform(0.8, 1.0),
                'safety_incidents': np.random.poisson(0.1),
                'human_workload': np.random.uniform(0.2, 0.9),
                'cobot_utilization': np.random.uniform(0.3, 0.9)
            })
        
        return pd.DataFrame(data)

def main():
    """Example usage of DataLoader."""
    print("🔄 DataLoader Demo")
    print("=" * 30)
    
    # Initialize loader
    loader = DataLoader()
    
    # Load different types of data
    sensor_data = loader.load_sensor_data()
    production_data = loader.load_production_data()
    energy_data = loader.load_energy_data()
    collaboration_data = loader.load_collaboration_data()
    
    # Generate summaries
    print("\n📊 Data Summaries:")
    print(f"Sensor data: {sensor_data.shape}")
    print(f"Production data: {production_data.shape}")
    print(f"Energy data: {energy_data.shape}")
    print(f"Collaboration data: {collaboration_data.shape}")
    
    # Get detailed summary for one dataset
    summary = loader.get_data_summary(sensor_data)
    print(f"\n📈 Sensor Data Summary:")
    print(f"Columns: {len(summary['columns'])}")
    print(f"Missing values: {sum(summary['missing_values'].values())}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    
    # Save sample data
    print("\n💾 Saving sample data...")
    loader.save_data(sensor_data, "generated_sensor_data.csv", "processed_data")
    
    print("\n✅ DataLoader demo completed!")

if __name__ == "__main__":
    main()