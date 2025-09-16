# 🛰️ Patch Antenna Simulator - Desktop GUI

A modern, native Windows desktop application for patch antenna electromagnetic simulation and visualization.

![GUI Preview](https://img.shields.io/badge/Platform-Windows-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

### 🎯 **Clean, Modern Interface**
- **Dark theme** with professional styling
- **Tabbed visualization** (Geometry, 2D Patterns, 3D Pattern)
- **Real-time parameter updates**
- **Intuitive controls** and layout

### 📐 **Antenna Design**
- **Interactive parameter input** (frequency, εr, thickness, loss tangent)
- **Material selection** (copper, aluminum, gold, silver)
- **Auto-calculated patch dimensions**
- **Real-time 3D geometry visualization**

### ⚡ **FDTD Simulation**
- **Tutorial-based openEMS solver** (reliable and accurate)
- **Background simulation** (non-blocking UI)
- **Progress feedback** and status updates
- **Error handling** with user-friendly messages

### 📊 **Advanced Visualization**
- **3D radiation patterns** with proper dBi scaling
- **2D polar plots** (E-plane and H-plane)
- **Full 360° pattern display**
- **Interactive matplotlib plots** with zoom/pan
- **Professional antenna engineering appearance**

## 🚀 Quick Start

### **Option 1: Double-click Launch (Recommended)**
1. **Double-click** `launch_gui.bat`
2. The launcher will:
   - Check Python installation
   - Activate virtual environment (if available)
   - Install missing dependencies
   - Start the GUI application

### **Option 2: Manual Launch**
```bash
# From project root directory
python launch_gui.py
```

### **Option 3: Direct Launch**
```bash
# If all dependencies are installed
python gui_app.py
```

## 📋 Requirements

### **System Requirements**
- **Windows 10/11** (tested)
- **Python 3.8+**
- **4GB+ RAM** (for FDTD simulations)

### **Python Dependencies**
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `pydantic` - Data validation
- `tkinter` - GUI framework (usually included with Python)

### **External Dependencies**
- **openEMS** - FDTD electromagnetic solver
  - Download from: https://www.openems.de/start/download.html
  - Extract to project folder or specify custom path in GUI

## 🎮 Usage Guide

### **1. Set Parameters**
- **Frequency**: Operating frequency in GHz (e.g., 2.45 for WiFi)
- **Dielectric εr**: Substrate relative permittivity (e.g., 4.3 for FR-4)
- **Thickness h**: Substrate thickness in mm (e.g., 1.6 for standard PCB)
- **Loss tangent**: Material loss factor (e.g., 0.02 for FR-4)
- **Metal**: Conductor material (copper recommended)
- **openEMS DLL**: Path to openEMS installation

### **2. Update Geometry**
- Click **"📐 Update Geometry"** to visualize antenna structure
- View 3D geometry in the **Geometry** tab
- Patch dimensions are auto-calculated for resonance

### **3. Run Simulation**
- Click **"⚡ Run FDTD Simulation"** to start analysis
- Monitor progress in status bar
- Simulation runs in background (UI remains responsive)

### **4. View Results**
- **2D Patterns tab**: E-plane and H-plane polar plots
- **3D Pattern tab**: Complete radiation pattern visualization
- **Interactive plots**: Zoom, pan, and explore results

## 🔧 Troubleshooting

### **Common Issues**

#### **"Python not found"**
- Install Python 3.8+ from python.org
- Make sure Python is added to system PATH

#### **"openEMS probe failed"**
- Check openEMS DLL path in GUI
- Download openEMS from official website
- Extract to project folder or update path

#### **"Module not found" errors**
- Run: `pip install numpy matplotlib pydantic`
- Or use the automatic installer in `launch_gui.py`

#### **Simulation fails**
- Check parameter values (realistic ranges)
- Ensure openEMS path is correct
- Try smaller frequency or simpler geometry first

#### **GUI looks broken**
- Update to Python 3.8+
- Try: `pip install --upgrade tkinter matplotlib`
- Restart application

### **Performance Tips**
- **Close other applications** during FDTD simulation
- **Use SSD storage** for faster file I/O
- **8GB+ RAM recommended** for complex simulations

## 🏗️ Architecture

### **Code Organization**
```
gui_app.py              # Main GUI application
├── ModernStyle         # UI styling and theming
├── ParameterFrame      # Parameter input controls
├── ControlFrame        # Simulation control buttons
├── PlotFrame           # Matplotlib visualization
└── AntennaSimulatorGUI # Main application class

launch_gui.py           # Dependency checker and launcher
launch_gui.bat          # Windows batch launcher
```

### **Integration with Existing Code**
The GUI **reuses all existing working code** without modification:
- `antenna_sim.models` - Parameter definitions
- `antenna_sim.solver_approx` - Analytical solver
- `antenna_sim.plotting` - Visualization functions
- `antenna_sim.solver_fdtd_openems_fixed` - FDTD solver

## 🎨 UI Design

### **Color Scheme**
- **Background**: Dark gray (#2b2b2b)
- **Cards**: Medium gray (#3c3c3c)
- **Accents**: Blue (#0078d4), Green (#107c10), Orange (#ff8c00)
- **Text**: White/light gray for readability

### **Layout**
- **Left Panel**: Parameters and controls (fixed width)
- **Right Panel**: Tabbed visualization (expandable)
- **Modern styling** with rounded corners and shadows

## 🔮 Future Enhancements

### **Planned Features**
- [ ] **Parameter sweeps** (frequency, thickness, εr)
- [ ] **Export results** (images, data files)
- [ ] **Batch simulations** with progress tracking
- [ ] **Advanced materials** database
- [ ] **Smith chart** visualization
- [ ] **Animation** of field distributions

### **UI Improvements**
- [ ] **Resizable panels** with splitters
- [ ] **Customizable themes** (light/dark modes)
- [ ] **Keyboard shortcuts** for common actions
- [ ] **Recent parameters** history
- [ ] **Tooltips** with parameter explanations

## 📄 License

MIT License - see main project for details.

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch
3. **Test** with GUI application
4. **Submit** pull request

## 📞 Support

- **Issues**: Create GitHub issue with GUI tag
- **Documentation**: Check main project README
- **Examples**: See `test_openems.py` for solver usage

---

**🚀 Ready to simulate? Double-click `launch_gui.bat` to get started!**
