#!/usr/bin/env python3
"""
Debug script to test geometry plotting functionality
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

# Import our modules
from antenna_sim.models import PatchAntennaParams, Metal, metal_defaults
from antenna_sim.physics import design_patch_for_frequency
from antenna_sim.plotting import draw_patch_3d_geometry

def test_parameter_creation():
    """Test parameter creation like the GUI does"""
    print("=== Testing Parameter Creation ===")
    
    try:
        metal_name = "copper"
        metal_enum = Metal(metal_name)
        metal_props = metal_defaults[metal_enum]
        
        params = PatchAntennaParams(
            frequency_hz=2.45e9,
            eps_r=4.3,
            h_m=1.6e-3,
            loss_tangent=0.02,
            metal=metal_props,
            patch_length_m=None,
            patch_width_m=None
        )
        
        print("✅ Parameters created successfully!")
        print(f"   Frequency: {params.frequency_hz/1e9:.2f} GHz")
        print(f"   εr: {params.eps_r}")
        print(f"   h: {params.h_m*1e3:.1f} mm")
        print(f"   Metal: {params.metal.name}")
        
        return params
        
    except Exception as e:
        print(f"❌ Parameter creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dimension_calculation(params):
    """Test patch dimension calculation"""
    print("\n=== Testing Dimension Calculation ===")
    
    try:
        if params.patch_length_m and params.patch_width_m:
            L_m = params.patch_length_m
            W_m = params.patch_width_m
            print("Using provided dimensions:")
        else:
            L_m, W_m, eps_eff = design_patch_for_frequency(params.frequency_hz, params.eps_r, params.h_m)
            print("Calculated dimensions:")
        
        print(f"   Length (L): {L_m*1e3:.1f} mm")
        print(f"   Width (W): {W_m*1e3:.1f} mm") 
        print(f"   Height (h): {params.h_m*1e3:.1f} mm")
        
        return L_m, W_m
        
    except Exception as e:
        print(f"❌ Dimension calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_geometry_plot(params, L_m, W_m):
    """Test geometry plotting"""
    print("\n=== Testing Geometry Plot ===")
    
    try:
        # Create plot exactly like GUI
        fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#2b2b2b')
        
        print(f"Calling draw_patch_3d_geometry with:")
        print(f"   L_m = {L_m}")
        print(f"   W_m = {W_m}")
        print(f"   h_m = {params.h_m}")
        
        # Draw geometry
        draw_patch_3d_geometry(ax, L_m, W_m, params.h_m)
        
        # Style the plot
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        
        # Set title
        ax.set_title(f'Patch Antenna Geometry\n{params.frequency_hz/1e9:.2f} GHz, εr={params.eps_r}', 
                    color='white', fontsize=12, pad=20)
        
        fig.tight_layout()
        
        # Save plot to file for inspection
        fig.savefig('debug_geometry.png', facecolor='#2b2b2b', dpi=100)
        print("✅ Geometry plot created successfully!")
        print("   Saved as 'debug_geometry.png'")
        
        # Also show with regular matplotlib
        plt.figure(figsize=(10, 8))
        ax2 = plt.gca(projection='3d')
        draw_patch_3d_geometry(ax2, L_m, W_m, params.h_m)
        ax2.set_title(f'Patch Antenna Geometry\n{params.frequency_hz/1e9:.2f} GHz, εr={params.eps_r}')
        plt.savefig('debug_geometry_light.png', dpi=100)
        print("   Also saved light version as 'debug_geometry_light.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Geometry plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("🔧 DEBUG: Testing Geometry Plotting")
    print("=" * 50)
    
    # Test 1: Parameter creation
    params = test_parameter_creation()
    if not params:
        return
    
    # Test 2: Dimension calculation  
    L_m, W_m = test_dimension_calculation(params)
    if L_m is None or W_m is None:
        return
    
    # Test 3: Geometry plotting
    success = test_geometry_plot(params, L_m, W_m)
    
    if success:
        print("\n🎉 All tests passed! Geometry plotting should work.")
        print("Check the generated PNG files to see if the geometry looks correct.")
    else:
        print("\n💥 Geometry plotting failed. Check the error messages above.")

if __name__ == "__main__":
    main()
