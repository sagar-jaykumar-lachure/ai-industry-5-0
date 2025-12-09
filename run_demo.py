#!/usr/bin/env python3
"""
AI Industry 5.0 Demo Runner
============================

Main script to run demonstrations of all Industry 5.0 components.

Usage:
    python run_demo.py                    # Run complete workflow
    python run_demo.py --pm               # Run only predictive maintenance
    python run_demo.py --twin             # Run only digital twin
    python run_demo.py --collab           # Run only collaboration optimizer
    python run_demo.py --energy           # Run only energy optimizer
    python run_demo.py --data             # Run only data generation


"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def run_complete_demo():
    """Run the complete Industry 5.0 workflow demo."""
    print("🚀 Starting Complete Industry 5.0 Demo")
    print("="*50)
    
    try:
        from examples.complete_workflow import main as run_workflow
        run_workflow()
        
    except ImportError as e:
        print(f"❌ Failed to import workflow modules: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def run_predictive_maintenance():
    """Run predictive maintenance demo."""
    print("🔧 Predictive Maintenance Demo")
    print("="*35)
    
    try:
        from code.projects.predictive_maintenance import main as run_pm
        run_pm()
        
    except Exception as e:
        print(f"❌ Predictive maintenance demo failed: {e}")

def run_digital_twin():
    """Run digital twin demo."""
    print("🏭 Digital Twin Demo")
    print("="*25)
    
    try:
        from code.projects.digital_twin import main as run_twin
        run_twin()
        
    except Exception as e:
        print(f"❌ Digital twin demo failed: {e}")

def run_collaboration():
    """Run human-cobot collaboration demo."""
    print("🤝 Human-Cobot Collaboration Demo")
    print("="*40)
    
    try:
        from code.projects.human_cobot_collaboration import main as run_collab
        run_collab()
        
    except Exception as e:
        print(f"❌ Collaboration demo failed: {e}")

def run_energy():
    """Run sustainable energy optimizer demo."""
    print("⚡ Sustainable Energy Optimizer Demo")
    print("="*45)
    
    try:
        from code.projects.sustainable_energy_optimizer import main as run_energy
        run_energy()
        
    except Exception as e:
        print(f"❌ Energy optimizer demo failed: {e}")

def run_data_demo():
    """Run data loading and generation demo."""
    print("📊 Data Loading Demo")
    print("="*25)
    
    try:
        from code.utils.data_loader import main as run_data
        run_data()
        
    except Exception as e:
        print(f"❌ Data demo failed: {e}")

def show_help():
    """Show help information."""
    help_text = """
🤖 AI Industry 5.0 Demo Runner
==============================

Available commands:
    python run_demo.py              # Run complete workflow demo
    python run_demo.py --pm         # Predictive maintenance demo only
    python run_demo.py --twin       # Digital twin demo only
    python run_demo.py --collab     # Human-cobot collaboration demo only
    python run_demo.py --energy     # Energy optimization demo only
    python run_demo.py --data       # Data loading demo only
    python run_demo.py --help       # Show this help

Repository Structure:
├── code/
│   ├── projects/           # Main project implementations
│   ├── examples/           # Complete workflow examples
│   └── utils/              # Utility functions
├── data/
│   ├── datasets/           # Sample datasets
│   ├── raw_data/           # Raw data files
│   └── processed_data/     # Processed data and models
├── docs/                   # Documentation
└── presentation/           # PowerPoint presentation

Quick Start:
1. Install dependencies: pip install -r requirements.txt
2. Run complete demo: python run_demo.py
3. Explore individual modules with specific flags
4. Check generated dashboards in docs/output/

For more information, see README.md
"""
    print(help_text)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Industry 5.0 Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pm', action='store_true', help='Run predictive maintenance demo')
    parser.add_argument('--twin', action='store_true', help='Run digital twin demo')
    parser.add_argument('--collab', action='store_true', help='Run collaboration optimizer demo')
    parser.add_argument('--energy', action='store_true', help='Run energy optimizer demo')
    parser.add_argument('--data', action='store_true', help='Run data loading demo')
    parser.add_argument('--help', action='store_true', help='Show help information')
    
    args = parser.parse_args()
    
    if args.help or len(sys.argv) == 1:
        show_help()
        return
    
    # Run specific demos
    if args.pm:
        run_predictive_maintenance()
    elif args.twin:
        run_digital_twin()
    elif args.collab:
        run_collaboration()
    elif args.energy:
        run_energy()
    elif args.data:
        run_data_demo()
    else:
        # Default: run complete demo
        run_complete_demo()

if __name__ == "__main__":
    main()
