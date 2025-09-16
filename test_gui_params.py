#!/usr/bin/env python3
"""
Quick test script to verify GUI parameter creation works correctly
"""

from antenna_sim.models import PatchAntennaParams, Metal, metal_defaults

def test_parameter_creation():
    """Test that we can create parameters like the GUI does"""
    
    print("Testing parameter creation...")
    
    # Simulate GUI parameter creation
    metal_name = "copper"  # This is what comes from the GUI
    metal_enum = Metal(metal_name)
    metal_props = metal_defaults[metal_enum]
    
    print(f"Metal name: {metal_name}")
    print(f"Metal enum: {metal_enum}")
    print(f"Metal props: {metal_props}")
    
    try:
        params = PatchAntennaParams(
            frequency_hz=2.45e9,
            eps_r=4.3,
            h_m=1.6e-3,
            loss_tangent=0.02,
            metal=metal_props,  # Pass MetalProperties object
            patch_length_m=None,
            patch_width_m=None
        )
        
        print("✅ SUCCESS: Parameters created successfully!")
        print(f"   Frequency: {params.frequency_hz/1e9:.2f} GHz")
        print(f"   Dielectric: εr = {params.eps_r}")
        print(f"   Thickness: h = {params.h_m*1e3:.1f} mm")
        print(f"   Metal: {params.metal.name}")
        print(f"   Conductivity: {params.metal.conductivity_s_per_m:.1e} S/m")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_parameter_creation()
    if success:
        print("\n🎉 GUI parameter creation should work now!")
    else:
        print("\n💥 There's still an issue with parameter creation")
