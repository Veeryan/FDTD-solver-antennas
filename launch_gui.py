#!/usr/bin/env python3
"""
Launcher script for Patch Antenna Simulator GUI
Handles dependencies and provides user-friendly startup
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy',
        'matplotlib',
        'pydantic',
        'tkinter'  # Usually comes with Python
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages"""
    if packages:
        print(f"Installing missing packages: {', '.join(packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
            return True
        except subprocess.CalledProcessError:
            return False
    return True

def main():
    """Main launcher function"""
    print("🛰️ Patch Antenna Simulator - Desktop GUI")
    print("=" * 50)
    
    # Add current directory to Python path for antenna_sim package (do this first!)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"📁 Added {current_dir} to Python path")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("🐍 Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected")
        venv_path = Path(".venv")
        if venv_path.exists():
            print("💡 Tip: Activate virtual environment with: .venv\\Scripts\\activate")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        input("Press Enter to exit...")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check dependencies
    print("📦 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        if not in_venv:
            print("💡 Consider using a virtual environment to avoid conflicts")
        
        response = input("Would you like to install them automatically? (y/n): ")
        
        if response.lower().startswith('y'):
            if not install_missing_packages(missing):
                print("❌ Failed to install packages. Please install manually:")
                print(f"   pip install {' '.join(missing)}")
                input("Press Enter to exit...")
                return
        else:
            print("Please install missing packages manually and try again.")
            input("Press Enter to exit...")
            return
    
    print("✅ All dependencies satisfied")
    
    # Check if antenna_sim package is available (after path is set)
    try:
        import antenna_sim
        print("✅ Antenna simulation package found")
    except ImportError as e:
        print("❌ Error: antenna_sim package not found")
        print(f"Import error: {e}")
        print("Make sure you're running this from the correct directory")
        print(f"Current directory: {current_dir}")
        print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
        input("Press Enter to exit...")
        return
    
    # Check for openEMS directory
    openems_dir = Path("openEMS")
    if not openems_dir.exists():
        print("⚠️  Warning: openEMS directory not found")
        print("You'll need to specify the correct path in the GUI")
    else:
        print("✅ openEMS directory found")
    
    print("\n🚀 Starting GUI application...")
    
    try:
        # Import and run the GUI
        from gui_app import AntennaSimulatorGUI
        
        app = AntennaSimulatorGUI()
        app.run()
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed")
        print("2. Check that openEMS is properly installed")
        print("3. Run from the project root directory")
        print("4. Try activating virtual environment first")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
