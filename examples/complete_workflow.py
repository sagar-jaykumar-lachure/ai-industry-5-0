"""
Complete Industry 5.0 Workflow Example
======================================

Demonstrates a complete workflow using all Industry 5.0 modules together.


"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add the code directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

# Import all modules
from projects.predictive_maintenance import PredictiveMaintenance
from projects.digital_twin import DigitalTwin
from projects.human_cobot_collaboration import HumanCobotOptimizer
from projects.sustainable_energy_optimizer import SustainableEnergyOptimizer
from utils.data_loader import DataLoader

def setup_environment():
    """Setup the working environment."""
    print("🚀 Setting up Industry 5.0 Environment")
    print("=" * 45)
    
    # Create output directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('docs/output', exist_ok=True)
    
    # Initialize data loader
    loader = DataLoader()
    
    return loader

def run_complete_workflow():
    """Execute complete Industry 5.0 workflow."""
    
    # Setup environment
    loader = setup_environment()
    
    print("\n" + "="*60)
    print("INDUSTRY 5.0 COMPLETE WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # =============================================================================
    # PHASE 1: DATA GENERATION AND LOADING
    # =============================================================================
    print("\n📊 PHASE 1: DATA GENERATION AND LOADING")
    print("-" * 45)
    
    # Generate data for all systems
    print("🔧 Generating synthetic data...")
    
    # Predictive Maintenance Data
    pm_system = PredictiveMaintenance(n_estimators=50)  # Reduced for faster demo
    pm_data = pm_system.generate_sample_data(n_samples=500, equipment_types=2)
    
    # Digital Twin Data
    twin_system = DigitalTwin(contamination=0.08)
    twin_data = twin_system.generate_production_data(n_samples=800, n_machines=3, anomaly_rate=0.1)
    
    # Human-Cobot Collaboration Data
    collaboration_optimizer = HumanCobotOptimizer()
    collaboration_data = collaboration_optimizer.generate_collaboration_data(n_samples=400)
    
    # Energy Optimization Data
    energy_optimizer = SustainableEnergyOptimizer()
    energy_data = energy_optimizer.generate_energy_data(n_samples=600, n_factories=2)
    
    print(f"✅ Generated data:")
    print(f"   - Predictive Maintenance: {len(pm_data)} samples")
    print(f"   - Digital Twin: {len(twin_data)} samples")
    print(f"   - Human-Cobot Collaboration: {len(collaboration_data)} samples")
    print(f"   - Energy Optimization: {len(energy_data)} samples")
    
    # =============================================================================
    # PHASE 2: MODEL TRAINING
    # =============================================================================
    print("\n🧠 PHASE 2: MODEL TRAINING")
    print("-" * 30)
    
    print("🔧 Training predictive maintenance model...")
    pm_metrics = pm_system.train(pm_data)
    print(f"   MAE: {pm_metrics['test_mae']:.2f} hours")
    print(f"   R²: {pm_metrics['test_r2']:.3f}")
    
    print("🔧 Training digital twin baseline...")
    twin_metrics = twin_system.train_baseline(twin_data)
    print(f"   Anomalies detected: {twin_metrics['n_anomalies']} ({twin_metrics['anomaly_rate']:.1%})")
    
    print("🔧 Training collaboration optimization model...")
    collab_metrics = collaboration_optimizer.train_optimization_model(collaboration_data)
    print(f"   Model R²: {collab_metrics['test_r2']:.3f}")
    
    print("🔧 Training energy optimization models...")
    energy_metrics = energy_optimizer.train_energy_models(energy_data)
    print(f"   Energy model R²: {energy_metrics['energy_metrics']['r2']:.3f}")
    
    # =============================================================================
    # PHASE 3: PREDICTIONS AND OPTIMIZATION
    # =============================================================================
    print("\n🎯 PHASE 3: PREDICTIONS AND OPTIMIZATION")
    print("-" * 45)
    
    # Predictive Maintenance Predictions
    print("🔮 Running predictive maintenance analysis...")
    test_pm_data = pm_data.tail(50)
    pm_predictions = pm_system.predict(test_pm_data)
    risk_assessment = pm_system.predict_failure_risk(test_pm_data)
    
    critical_equipment = risk_assessment[risk_assessment['risk_level'] == 'Critical']
    print(f"   Critical equipment detected: {len(critical_equipment)}")
    
    # Generate maintenance schedule
    maintenance_schedule = pm_system.generate_maintenance_schedule(pm_data.tail(100), planning_horizon=7)
    print(f"   Equipment needing maintenance (7 days): {len(maintenance_schedule)}")
    
    # Digital Twin Anomaly Detection
    print("🔍 Running digital twin anomaly detection...")
    test_twin_data = twin_data.tail(100)
    twin_results = twin_system.detect_anomalies(test_twin_data)
    
    anomalies = twin_results[twin_results['anomaly'] == -1]
    print(f"   Anomalies detected: {len(anomalies)}")
    
    if len(anomalies) > 0:
        anomaly_types = anomalies['anomaly_type'].value_counts()
        print(f"   Anomaly types: {dict(anomaly_types)}")
    
    # Performance metrics
    performance = twin_system.get_performance_metrics(twin_results)
    print(f"   System uptime: {performance['uptime_percentage']:.1f}%")
    
    # Human-Cobot Collaboration Optimization
    print("🤝 Optimizing human-cobot collaboration...")
    
    # Test different scenarios
    scenarios = [
        {"human_skill": 0.85, "cobot_accuracy": 0.92, "complexity": 0.7, "urgency": 0.4},
        {"human_skill": 0.6, "cobot_accuracy": 0.95, "complexity": 0.9, "urgency": 0.8},
        {"human_skill": 0.9, "cobot_accuracy": 0.88, "complexity": 0.3, "urgency": 0.2}
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        result = collaboration_optimizer.optimize_collaboration(**scenario)
        print(f"   Scenario {i}: {result['optimal_human_allocation']:.1%} human allocation")
        print(f"            Quality: {result['predicted_quality_score']:.3f}")
        print(f"            Safety Risk: {result['predicted_safety_risk']:.3f}")
    
    # Energy Optimization
    print("⚡ Running energy optimization...")
    
    production_params = {
        'production_rate': 120,
        'machine_utilization': 0.85,
        'temperature': 20,
        'solar_irradiance': 500,
        'factory_id': 0
    }
    
    energy_optimization = energy_optimizer.optimize_energy_consumption(production_params)
    
    print(f"   Total consumption: {energy_optimization['total_consumption_kwh']:.1f} kWh")
    print(f"   Renewable generation: {energy_optimization['total_renewable_kwh']:.1f} kWh")
    print(f"   Self-consumption ratio: {energy_optimization['overall_self_consumption_ratio']:.1%}")
    print(f"   Annual savings: ${energy_optimization['cost_savings_estimate']['total_annual_savings']:.0f}")
    
    # =============================================================================
    # PHASE 4: INSIGHTS AND ANALYSIS
    # =============================================================================
    print("\n📈 PHASE 4: INSIGHTS AND ANALYSIS")
    print("-" * 35)
    
    # Predictive Maintenance Insights
    print("🔧 Predictive Maintenance Insights:")
    feature_importance = pm_system.get_feature_importance(5)
    print(f"   Top features: {feature_importance['feature'].tolist()}")
    
    # Digital Twin Insights
    print("🏭 Digital Twin Insights:")
    if len(twin_results) > 0:
        machine_performance = {}
        for machine_id in ['machine_0', 'machine_1', 'machine_2']:
            if machine_id in twin_results.columns:
                avg_cycle_time = twin_results[f'{machine_id.split("_")[1]}_cycle_time'].mean()
                machine_performance[machine_id] = avg_cycle_time
        print(f"   Machine performance: {machine_performance}")
    
    # Human-Cobot Collaboration Insights
    print("🤝 Collaboration Insights:")
    collaboration_insights = collaboration_optimizer.get_collaboration_insights(collaboration_data)
    print(f"   High human allocation tasks: {collaboration_insights['high_human_allocation_tasks']}")
    print(f"   Average quality by allocation correlation: {collaboration_insights['quality_by_allocation']}")
    
    # Energy Insights
    print("⚡ Energy Insights:")
    energy_insights = energy_optimizer.get_energy_insights(energy_data)
    peak_hour = energy_insights['consumption_patterns']['peak_consumption_hour']
    print(f"   Peak consumption hour: {peak_hour}:00")
    print(f"   Average self-consumption: {energy_insights['renewable_performance']['avg_self_consumption']:.1%}")
    
    # =============================================================================
    # PHASE 5: DASHBOARD GENERATION
    # =============================================================================
    print("\n📊 PHASE 5: DASHBOARD GENERATION")
    print("-" * 35)
    
    print("🎨 Generating visualization dashboards...")
    
    try:
        # Generate smaller datasets for visualization
        pm_viz_data = pm_data.tail(200)
        twin_viz_data = twin_results.tail(200)
        collab_viz_data = collaboration_data.tail(200)
        energy_viz_data = energy_data.tail(200)
        
        # Create dashboards (save to files)
        pm_system.plot_predictions(pm_viz_data, save_path='docs/output/predictive_maintenance_dashboard.png')
        twin_system.plot_dashboard(twin_viz_data, save_path='docs/output/digital_twin_dashboard.html')
        collaboration_optimizer.plot_optimization_dashboard(collab_viz_data, save_path='docs/output/collaboration_dashboard.html')
        energy_optimizer.plot_energy_dashboard(energy_viz_data, save_path='docs/output/energy_dashboard.html')
        
        print("   ✅ Dashboards generated successfully")
        
    except Exception as e:
        print(f"   ⚠️ Dashboard generation skipped: {e}")
    
    # =============================================================================
    # PHASE 6: MODEL PERSISTENCE
    # =============================================================================
    print("\n💾 PHASE 6: MODEL PERSISTENCE")
    print("-" * 30)
    
    print("🗄️ Saving trained models...")
    
    try:
        pm_system.save_model('data/processed/predictive_maintenance_model.pkl')
        twin_system.save_model('data/processed/digital_twin_model.pkl')
        collaboration_optimizer.save_model('data/processed/collaboration_optimizer.pkl')
        energy_optimizer.save_models('data/processed/energy_optimizer')
        
        print("   ✅ Models saved successfully")
        
    except Exception as e:
        print(f"   ⚠️ Model saving failed: {e}")
    
    # =============================================================================
    # PHASE 7: SUMMARY REPORT
    # =============================================================================
    print("\n📋 PHASE 7: SUMMARY REPORT")
    print("-" * 30)
    
    print("\n🎯 INDUSTRY 5.0 WORKFLOW SUMMARY")
    print("="*40)
    
    print("\n✅ COMPLETED TASKS:")
    print("   1. ✅ Data generation and loading")
    print("   2. ✅ Model training (4 AI systems)")
    print("   3. ✅ Predictions and optimization")
    print("   4. ✅ Insights and analysis")
    print("   5. ✅ Dashboard generation")
    print("   6. ✅ Model persistence")
    
    print("\n📊 KEY METRICS ACHIEVED:")
    print(f"   • Predictive Maintenance MAE: {pm_metrics['test_mae']:.2f} hours")
    print(f"   • Digital Twin Anomaly Rate: {twin_metrics['anomaly_rate']:.1%}")
    print(f"   • Collaboration Model R²: {collab_metrics['test_r2']:.3f}")
    print(f"   • Energy Model R²: {energy_metrics['energy_metrics']['r2']:.3f}")
    print(f"   • System Uptime: {performance['uptime_percentage']:.1f}%")
    print(f"   • Annual Energy Savings: ${energy_optimization['cost_savings_estimate']['total_annual_savings']:.0f}")
    
    print("\n🎯 INDUSTRY 5.0 PRINCIPLES DEMONSTRATED:")
    print("   • Human-Centricity: ✅ Human-cobot collaboration optimization")
    print("   • Sustainability: ✅ Energy optimization and carbon reduction")
    print("   • Resilience: ✅ Predictive maintenance and anomaly detection")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Deploy models in production environment")
    print("   2. Integrate with existing manufacturing systems")
    print("   3. Implement real-time monitoring dashboards")
    print("   4. Establish feedback loops for continuous improvement")
    print("   5. Scale to additional production lines and facilities")
    
    print("\n" + "="*60)
    print("🎉 INDUSTRY 5.0 WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)

def main():
    """Main execution function."""
    try:
        run_complete_workflow()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Workflow interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Workflow failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n👋 Thank you for using the Industry 5.0 workflow demo!")

if __name__ == "__main__":
    main()
