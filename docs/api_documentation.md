# API Documentation - AI Industry 5.0

## Overview

This document provides comprehensive API reference for all Python implementations in the AI Industry 5.0 repository. The modules are designed for easy integration into existing manufacturing systems and can be used independently or as part of a larger Industry 5.0 solution.

## Table of Contents

1. [Predictive Maintenance](#predictive-maintenance)
2. [Digital Twin](#digital-twin)
3. [Human-Cobot Collaboration](#human-cobot-collaboration)
4. [Sustainable Energy Optimizer](#sustainable-energy-optimizer)
5. [Common Utilities](#common-utilities)

---

## Predictive Maintenance

### Class: `PredictiveMaintenance`

**Location**: `code/projects/predictive_maintenance.py`

#### Constructor

```python
PredictiveMaintenance(n_estimators=100, random_state=42)
```

**Parameters:**
- `n_estimators` (int): Number of trees in Random Forest (default: 100)
- `random_state` (int): Random seed for reproducibility (default: 42)

#### Methods

##### `generate_sample_data(n_samples=1000, equipment_types=5)`

Generate synthetic manufacturing sensor data for training and testing.

**Parameters:**
- `n_samples` (int): Number of data points to generate (default: 1000)
- `equipment_types` (int): Number of different equipment types (default: 5)

**Returns:** `pd.DataFrame` with sensor data including:
- `equipment_id`: Unique equipment identifier
- `vibration`, `temperature`, `pressure`: Sensor readings
- `runtime_hours`: Equipment operating hours
- `rul`: Remaining Useful Life (target variable)
- `failure_probability`: Calculated failure probability

**Example:**
```python
pm_system = PredictiveMaintenance()
data = pm_system.generate_sample_data(n_samples=2000, equipment_types=3)
print(f"Generated {len(data)} samples")
```

##### `train(df, test_size=0.2)`

Train the predictive maintenance model using sensor data.

**Parameters:**
- `df` (pd.DataFrame): Training data from `generate_sample_data()`
- `test_size` (float): Proportion of data for testing (default: 0.2)

**Returns:** `dict` containing training metrics:
- `train_mae`, `test_mae`: Mean Absolute Error
- `train_rmse`, `test_rmse`: Root Mean Square Error
- `train_r2`, `test_r2`: R-squared scores
- `n_features`, `n_samples`: Dataset statistics

**Example:**
```python
metrics = pm_system.train(data)
print(f"Test MAE: {metrics['test_mae']:.2f} hours")
```

##### `predict(df)`

Make RUL predictions on new equipment data.

**Parameters:**
- `df` (pd.DataFrame): Equipment data with same structure as training data

**Returns:** `np.array` of RUL predictions in hours

**Example:**
```python
predictions = pm_system.predict(new_equipment_data)
high_risk_equipment = new_equipment_data[predictions < 500]
```

##### `predict_failure_risk(df, threshold=500)`

Predict failure risk for equipment with risk categorization.

**Parameters:**
- `df` (pd.DataFrame): Equipment data
- `threshold` (int): RUL threshold for high risk classification (default: 500)

**Returns:** `pd.DataFrame` with additional columns:
- `predicted_rul`: RUL predictions
- `failure_risk`: Boolean indicating high risk
- `risk_level`: Categorical risk level ('Critical', 'High', 'Medium', 'Low')

**Example:**
```python
risk_assessment = pm_system.predict_failure_risk(equipment_data)
critical_equipment = risk_assessment[risk_assessment['risk_level'] == 'Critical']
```

##### `generate_maintenance_schedule(df, planning_horizon=30)`

Generate maintenance schedule based on RUL predictions.

**Parameters:**
- `df` (pd.DataFrame): Equipment data with predictions
- `planning_horizon` (int): Days to plan ahead (default: 30)

**Returns:** `pd.DataFrame` with maintenance schedule:
- `equipment_id`: Equipment identifier
- `current_rul`: Predicted remaining useful life
- `days_to_failure`: Days until predicted failure
- `recommended_maintenance`: Recommended maintenance date
- `priority`: Maintenance priority ('High', 'Medium')
- `risk_level`: Equipment risk assessment

**Example:**
```python
schedule = pm_system.generate_maintenance_schedule(equipment_data, planning_horizon=14)
print(f"Equipment needing maintenance: {len(schedule)}")
```

##### `plot_predictions(df, save_path=None)`

Create comprehensive visualization dashboard.

**Parameters:**
- `df` (pd.DataFrame): Data for visualization
- `save_path` (str, optional): Path to save the plot

**Returns:** Displays matplotlib dashboard with:
- Actual vs Predicted RUL scatter plot
- Feature importance ranking
- RUL distribution histogram
- Equipment health status pie chart

**Example:**
```python
pm_system.plot_predictions(test_data, save_path='maintenance_dashboard.png')
```

---

## Digital Twin

### Class: `DigitalTwin`

**Location**: `code/projects/digital_twin.py`

#### Constructor

```python
DigitalTwin(contamination=0.05, random_state=42)
```

**Parameters:**
- `contamination` (float): Expected proportion of anomalies (default: 0.05)
- `random_state` (int): Random seed for reproducibility (default: 42)

#### Methods

##### `generate_production_data(n_samples=5000, n_machines=4, anomaly_rate=0.05)`

Generate synthetic production line data with realistic manufacturing patterns.

**Parameters:**
- `n_samples` (int): Number of data points (default: 5000)
- `n_machines` (int): Number of machines in production line (default: 4)
- `anomaly_rate` (float): Proportion of anomalous samples (default: 0.05)

**Returns:** `pd.DataFrame` with production data including:
- `timestamp`: Data collection timestamp
- Machine-specific metrics: `machine_{i}_cycle_time`, `machine_{i}_defect_rate`, etc.
- Line-level metrics: `line_cycle_time`, `line_defect_rate`, `line_throughput`
- Environmental factors: `ambient_temp`, `humidity`, `pressure`

**Example:**
```python
twin = DigitalTwin()
production_data = twin.generate_production_data(n_samples=3000, n_machines=4)
```

##### `train_baseline(df)`

Train the digital twin baseline model using normal operation data.

**Parameters:**
- `df` (pd.DataFrame): Production data from `generate_production_data()`

**Returns:** `dict` with training metrics:
- `n_samples`, `n_features`: Dataset statistics
- `n_anomalies`, `anomaly_rate`: Anomaly detection results
- `n_clusters`: Number of normal operation clusters
- `baseline_established`: Boolean indicating successful baseline creation

**Example:**
```python
metrics = twin.train_baseline(production_data)
print(f"Detected {metrics['n_anomalies']} anomalies")
```

##### `detect_anomalies(df, return_scores=False)`

Detect anomalies in new production data using trained baseline.

**Parameters:**
- `df` (pd.DataFrame): Production data for analysis
- `return_scores` (bool): Whether to return anomaly confidence scores (default: False)

**Returns:** `pd.DataFrame` with additional columns:
- `anomaly`: -1 for anomaly, 1 for normal
- `anomaly_score`: Confidence score (lower = more anomalous)
- `anomaly_type`: Categorized anomaly type ('Machine Failure', 'Quality Issue', etc.)
- `risk_level`: Risk assessment ('Low', 'Medium', 'High', 'Critical')

**Example:**
```python
results = twin.detect_anomalies(new_production_data)
anomalies = results[results['anomaly'] == -1]
print(f"Found {len(anomalies)} anomalies")
```

##### `simulate_scenario(df, scenario_type, intensity=1.0)`

Simulate different operational scenarios for what-if analysis.

**Parameters:**
- `df` (pd.DataFrame): Base production data
- `scenario_type` (str): Type of scenario ('machine_maintenance', 'quality_boost', 'peak_load')
- `intensity` (float): Intensity of scenario effect (default: 1.0)

**Returns:** `pd.DataFrame` with simulated scenario data

**Example:**
```python
maintenance_scenario = twin.simulate_scenario(data, 'machine_maintenance', intensity=1.5)
```

##### `get_performance_metrics(df)`

Calculate comprehensive performance metrics for production data.

**Parameters:**
- `df` (pd.DataFrame): Production data with anomaly detection

**Returns:** `dict` with performance metrics:
- `avg_cycle_time`, `avg_defect_rate`, `avg_throughput`, `avg_efficiency`: Average metrics
- `anomaly_rate`: Proportion of anomalous operations
- `critical_anomalies`: Count of critical anomalies
- `uptime_percentage`: System availability percentage
- `cycle_time_vs_baseline`, `efficiency_vs_baseline`: Performance vs baseline

**Example:**
```python
performance = twin.get_performance_metrics(results)
print(f"System uptime: {performance['uptime_percentage']:.1f}%")
```

##### `plot_dashboard(df, save_path=None)`

Create comprehensive interactive dashboard using Plotly.

**Parameters:**
- `df` (pd.DataFrame): Production data with anomaly detection
- `save_path` (str, optional): Path to save HTML dashboard

**Returns:** Displays interactive dashboard with multiple subplots:
- Production metrics over time
- Anomaly distribution analysis
- Machine performance comparison
- Defect rate analysis
- Throughput vs cycle time correlation
- Anomaly type breakdown

**Example:**
```python
twin.plot_dashboard(results, save_path='digital_twin_dashboard.html')
```

---

## Human-Cobot Collaboration

### Class: `HumanCobotOptimizer`

**Location**: `code/projects/human_cobot_collaboration.py`

#### Constructor

```python
HumanCobotOptimizer(random_state=42)
```

**Parameters:**
- `random_state` (int): Random seed for reproducibility (default: 42)

#### Methods

##### `generate_collaboration_data(n_samples=2000)`

Generate synthetic human-cobot collaboration data.

**Parameters:**
- `n_samples` (int): Number of collaboration scenarios (default: 2000)

**Returns:** `pd.DataFrame` with collaboration data including:
- Task characteristics: `complexity`, `urgency`, `precision_required`
- Human factors: `human_skill`, `human_fatigue`, `human_experience`
- Cobot characteristics: `cobot_accuracy`, `cobot_speed`, `cobot_payload`
- Environmental factors: `workspace_layout`, `safety_requirements`
- Performance metrics: `optimal_human_allocation`, `task_completion_time`, `quality_score`

**Example:**
```python
optimizer = HumanCobotOptimizer()
collaboration_data = optimizer.generate_collaboration_data(n_samples=3000)
```

##### `train_optimization_model(df)`

Train the Gaussian Process optimization model.

**Parameters:**
- `df` (pd.DataFrame): Collaboration data from `generate_collaboration_data()`

**Returns:** `dict` with training metrics:
- `train_mae`, `test_mae`: Mean Absolute Error
- `train_r2`, `test_r2`: R-squared scores
- `n_features`, `n_samples`: Dataset statistics
- `quality_correlation`: Correlation between allocation and quality
- `safety_incident_rate`: Average safety incident rate

**Example:**
```python
metrics = optimizer.train_optimization_model(collaboration_data)
print(f"Model R²: {metrics['test_r2']:.3f}")
```

##### `optimize_collaboration(human_skill, cobot_accuracy, complexity, urgency=0.5, precision_required=0.7, **kwargs)`

Optimize human-cobot collaboration for specific task parameters.

**Parameters:**
- `human_skill` (float): Human worker skill level (0-1)
- `cobot_accuracy` (float): Cobot accuracy (0-1)
- `complexity` (float): Task complexity (0-1)
- `urgency` (float): Task urgency (0-1, default: 0.5)
- `precision_required` (float): Precision requirement (0-1, default: 0.7)
- `**kwargs`: Additional parameters (human_fatigue, human_experience, cobot_speed, etc.)

**Returns:** `dict` with optimization results:
- `optimal_human_allocation`: Recommended human allocation ratio (0-1)
- `allocation_uncertainty`: Confidence interval for allocation
- `cobot_allocation`: Complementary cobot allocation
- `predicted_completion_time`: Estimated task completion time
- `predicted_quality_score`: Expected quality outcome (0-1)
- `predicted_safety_risk`: Safety risk assessment (0-1)
- `human_workload`: Predicted human workload (0-1)
- `cobot_utilization`: Predicted cobot utilization (0-1)
- `recommendation`: Human-readable optimization recommendation

**Example:**
```python
result = optimizer.optimize_collaboration(
    human_skill=0.85,
    cobot_accuracy=0.92,
    complexity=0.7,
    urgency=0.4,
    precision_required=0.8
)
print(f"Optimal human allocation: {result['optimal_human_allocation']:.1%}")
```

##### `simulate_scenarios(df, n_scenarios=100)`

Simulate various collaboration scenarios for analysis.

**Parameters:**
- `df` (pd.DataFrame): Base collaboration data
- `n_scenarios` (int): Number of scenarios to simulate (default: 100)

**Returns:** `pd.DataFrame` with simulation results including:
- Scenario parameters and optimization results for each simulated scenario

**Example:**
```python
scenarios = optimizer.simulate_scenarios(collaboration_data, n_scenarios=200)
```

##### `get_collaboration_insights(df)`

Generate comprehensive insights about collaboration patterns.

**Parameters:**
- `df` (pd.DataFrame): Collaboration data

**Returns:** `dict` with key insights:
- `high_human_allocation_tasks`: Task types favoring human workers
- `high_cobot_tasks`: Task types suitable for automation
- `skill_impact`: Impact of human skill on allocation
- `complexity_impact`: Impact of task complexity on allocation
- `fatigue_threshold`: Allocation adjustment for fatigue
- Various statistical analyses of collaboration patterns

**Example:**
```python
insights = optimizer.get_collaboration_insights(collaboration_data)
print(f"High human allocation tasks: {insights['high_human_allocation_tasks']}")
```

##### `plot_optimization_dashboard(df, save_path=None)`

Create comprehensive optimization dashboard.

**Parameters:**
- `df` (pd.DataFrame): Collaboration data
- `save_path` (str, optional): Path to save HTML dashboard

**Returns:** Displays interactive dashboard with:
- Human allocation vs task complexity scatter plot
- Skill level impact box plots
- Task type performance analysis
- Quality vs allocation correlation
- Workload distribution histograms
- Safety analysis by allocation

**Example:**
```python
optimizer.plot_optimization_dashboard(collaboration_data, save_path='collaboration_dashboard.html')
```

---

## Sustainable Energy Optimizer

### Class: `SustainableEnergyOptimizer`

**Location**: `code/projects/sustainable_energy_optimizer.py`

#### Constructor

```python
SustainableEnergyOptimizer(random_state=42)
```

**Parameters:**
- `random_state` (int): Random seed for reproducibility (default: 42)

#### Methods

##### `generate_energy_data(n_samples=5000, n_factories=3)`

Generate synthetic factory energy consumption data.

**Parameters:**
- `n_samples` (int): Number of data points (default: 5000)
- `n_factories` (int): Number of factories (default: 3)

**Returns:** `pd.DataFrame` with energy data including:
- Time features: `timestamp`, `hour`, `day_of_week`, `month`, `is_weekend`
- Environmental conditions: `temperature`, `humidity`, `solar_irradiance`, `wind_speed`
- Operational parameters: `production_rate`, `machine_utilization`, `maintenance_status`
- Energy components: `hvac_load`, `lighting_load`, `process_heat`, `motor_load`
- Renewable generation: `solar_generation`, `wind_generation`, `total_renewable`
- Consumption metrics: `total_consumption`, `net_grid_consumption`, `self_consumption_ratio`
- Cost and sustainability: `total_cost`, `carbon_intensity`, `energy_efficiency_score`

**Example:**
```python
energy_optimizer = SustainableEnergyOptimizer()
energy_data = energy_optimizer.generate_energy_data(n_samples=4000, n_factories=3)
```

##### `train_energy_models(df)`

Train energy consumption and renewable energy models.

**Parameters:**
- `df` (pd.DataFrame): Energy data from `generate_energy_data()`

**Returns:** `dict` with training metrics:
- `energy_metrics`: Metrics for energy consumption model (mae, rmse, r2)
- `renewable_metrics`: Metrics for renewable energy model (mae, rmse, r2)
- `n_features`, `n_samples`: Dataset statistics

**Example:**
```python
metrics = energy_optimizer.train_energy_models(energy_data)
print(f"Energy model R²: {metrics['energy_metrics']['r2']:.3f}")
```

##### `optimize_energy_consumption(production_params, optimization_horizon=24)`

Optimize energy consumption for given production parameters.

**Parameters:**
- `production_params` (dict): Production and operational parameters
- `optimization_horizon` (int): Hours to optimize ahead (default: 24)

**Required production_params:**
- `production_rate`: Production rate
- `machine_utilization`: Machine utilization ratio
- `temperature`: Ambient temperature
- `factory_id`: Factory identifier

**Returns:** `dict` with optimization results:
- `optimization_scenarios`: Hourly optimization scenarios
- `total_consumption_kwh`: Total energy consumption
- `total_renewable_kwh`: Total renewable generation
- `total_net_consumption_kwh`: Net grid consumption
- `overall_self_consumption_ratio`: Overall renewable utilization
- `optimization_potential`: Dictionary with optimization opportunities:
  - `peak_shaving_opportunities`: Peak demand reduction potential
  - `renewable_utilization_boost`: Renewable integration improvements
  - `efficiency_improvements`: Energy efficiency suggestions
  - `demand_response_potential`: Demand response opportunities
- `cost_savings_estimate`: Cost savings projections
- `carbon_reduction_estimate`: Carbon footprint reduction projections

**Example:**
```python
optimization = energy_optimizer.optimize_energy_consumption({
    'production_rate': 120,
    'machine_utilization': 0.85,
    'temperature': 20,
    'solar_irradiance': 500,
    'factory_id': 0
})
print(f"Annual cost savings: ${optimization['cost_savings_estimate']['total_annual_savings']:.0f}")
```

##### `get_energy_insights(df)`

Generate comprehensive energy insights from historical data.

**Parameters:**
- `df` (pd.DataFrame): Energy consumption data

**Returns:** `dict` with insights:
- `consumption_patterns`: Peak/low consumption analysis
- `renewable_performance`: Renewable energy utilization analysis
- `efficiency_metrics`: Energy efficiency analysis
- `cost_analysis`: Cost analysis and optimization opportunities

**Example:**
```python
insights = energy_optimizer.get_energy_insights(energy_data)
peak_hour = insights['consumption_patterns']['peak_consumption_hour']
print(f"Peak consumption occurs at {peak_hour}:00")
```

##### `plot_energy_dashboard(df, save_path=None)`

Create comprehensive energy optimization dashboard.

**Parameters:**
- `df` (pd.DataFrame): Energy data
- `save_path` (str, optional): Path to save HTML dashboard

**Returns:** Displays interactive dashboard with:
- Energy consumption patterns over time
- Renewable vs grid energy analysis
- Efficiency metrics by factory
- Cost analysis visualizations
- Production vs energy correlation
- Carbon footprint analysis

**Example:**
```python
energy_optimizer.plot_energy_dashboard(energy_data, save_path='energy_dashboard.html')
```

---

## Common Utilities

### Model Persistence

All classes provide `save_model()` and `load_model()` methods for persistence:

```python
# Save models
pm_system.save_model('path/to/model.pkl')
twin.save_model('path/to/twin.pkl')
optimizer.save_model('path/to/optimizer.pkl')
energy_optimizer.save_models('path/to/energy_')

# Load models
pm_system.load_model('path/to/model.pkl')
twin.load_model('path/to/twin.pkl')
optimizer.load_model('path/to/optimizer.pkl')
energy_optimizer.load_models('path/to/energy_')
```

### Data Export/Import

```python
# Export data to CSV
df.to_csv('data/processed/exported_data.csv', index=False)

# Import data from CSV
imported_df = pd.read_csv('data/processed/exported_data.csv')
```

### Configuration

```python
# Example configuration for production deployment
CONFIG = {
    'predictive_maintenance': {
        'model_path': 'models/pm_model.pkl',
        'alert_threshold': 500,  # hours
        'training_frequency': 'weekly'
    },
    'digital_twin': {
        'model_path': 'models/dt_model.pkl',
        'contamination': 0.05,
        'real_time_monitoring': True
    },
    'energy_optimizer': {
        'model_path': 'models/energy_model.pkl',
        'optimization_horizon': 24,
        'renewable_focus': True
    }
}
```

### Error Handling

```python
try:
    predictions = pm_system.predict(new_data)
except ValueError as e:
    print(f"Model not trained: {e}")
    # Train model first
    pm_system.train(training_data)
    predictions = pm_system.predict(new_data)
```

### Performance Monitoring

```python
import time
import psutil

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent
        
        print(f"Execution time: {end_time - start_time:.2f}s")
        print(f"Memory usage: {end_memory - start_memory:.1f}%")
        
        return result
    return wrapper

# Apply monitoring to predictions
pm_system.predict = monitor_performance(pm_system.predict)
```

---

## Integration Examples

### Complete Workflow

```python
# Initialize all systems
pm_system = PredictiveMaintenance()
twin = DigitalTwin()
optimizer = HumanCobotOptimizer()
energy_optimizer = SustainableEnergyOptimizer()

# Generate data
pm_data = pm_system.generate_sample_data(1000)
twin_data = twin.generate_production_data(2000)
collaboration_data = optimizer.generate_collaboration_data(1500)
energy_data = energy_optimizer.generate_energy_data(2500)

# Train models
pm_metrics = pm_system.train(pm_data)
twin_metrics = twin.train_baseline(twin_data)
optimizer_metrics = optimizer.train_optimization_model(collaboration_data)
energy_metrics = energy_optimizer.train_energy_models(energy_data)

# Make predictions
pm_predictions = pm_system.predict(pm_data[:100])
twin_anomalies = twin.detect_anomalies(twin_data[:100])
collaboration_optimization = optimizer.optimize_collaboration(
    human_skill=0.8, cobot_accuracy=0.9, complexity=0.6
)
energy_optimization = energy_optimizer.optimize_energy_consumption({
    'production_rate': 100, 'machine_utilization': 0.8, 'factory_id': 0
})

# Generate reports
pm_system.plot_predictions(pm_data[:500])
twin.plot_dashboard(twin_data[:500])
optimizer.plot_optimization_dashboard(collaboration_data[:500])
energy_optimizer.plot_energy_dashboard(energy_data[:500])
```

---

## Support and Contributing

For issues, feature requests, or contributions:
1. Check existing documentation and examples
2. Create detailed issue reports with code samples
3. Follow the established code style and documentation standards
4. Include unit tests for new functionality

**Author**: MiniMax Agent  
**Version**: 1.0.0  
**Last Updated**: December 2025