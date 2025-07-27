import numpy as np

from et_simul.core.eye import Eye

print("=== STEP-BY-STEP ANGLE KAPPA CALCULATION ===\n")

# Create Eye instance with default values
e = Eye()

print(f"Created Eye with default parameters:")
print(f"- Fovea horizontal displacement (α) = {e.fovea_alpha_deg}°")
print(f"- Fovea vertical displacement (β) = {e.fovea_beta_deg}°")
print(f"- Eye axial length = {e.axial_length*1000:.1f} mm")

# STEP 1: Calculate retina distance
retina_distance = e.axial_length / 2
print(f"\nSTEP 1: Calculate retina distance from rotation center")
print(f"retina_distance = axial_length / 2 = {retina_distance*1000:.1f} mm")

# STEP 2: Convert angles to radians
alpha_rad = e.fovea_alpha_deg * np.pi / 180.0
beta_rad = e.fovea_beta_deg * np.pi / 180.0
print(f"\nSTEP 2: Convert angles to radians")
print(f"α = {e.fovea_alpha_deg}° = {alpha_rad:.4f} rad")
print(f"β = {e.fovea_beta_deg}° = {beta_rad:.4f} rad")

# STEP 3: Get actual fovea position from Eye instance
print(f"\nSTEP 3: Get 3D fovea position from Eye.fovea_position property")
fovea_pos = e.fovea_position
print(
    f"Fovea position: [{fovea_pos[0]*1000:.3f}, {fovea_pos[1]*1000:.3f}, {fovea_pos[2]*1000:.3f}] mm"
)

print(f"\nCalculation details (from eye.py:763-765):")
print(
    f"x = R × sin(α) × cos(β) = {retina_distance*1000:.1f} × sin({e.fovea_alpha_deg}°) × cos({e.fovea_beta_deg}°) = {fovea_pos[0]*1000:.3f} mm"
)
print(
    f"y = R × sin(β)          = {retina_distance*1000:.1f} × sin({e.fovea_beta_deg}°)                    = {fovea_pos[1]*1000:.3f} mm"
)
print(
    f"z = R × cos(α) × cos(β) = {retina_distance*1000:.1f} × cos({e.fovea_alpha_deg}°) × cos({e.fovea_beta_deg}°) = {fovea_pos[2]*1000:.3f} mm"
)

# STEP 4: Calculate angle kappa using the new Eye.angle_kappa property
print(f"\nSTEP 4: Calculate angle kappa using Eye.angle_kappa property")
angle_kappa_eye_property = e.angle_kappa
print(f"e.angle_kappa = {angle_kappa_eye_property:.3f}°")

# Show the manual calculation for comparison
print(f"\nManual calculation (same as in Eye.angle_kappa):")
visual_axis = fovea_pos / np.linalg.norm(fovea_pos)
optical_axis = np.array([0, 0, -1])  # Eye looks along -Z axis

print(
    f"Visual axis (normalized): [{visual_axis[0]:.4f}, {visual_axis[1]:.4f}, {visual_axis[2]:.4f}]"
)
print(f"Optical axis: [{optical_axis[0]}, {optical_axis[1]}, {optical_axis[2]}]")

dot_product = np.dot(visual_axis, optical_axis)
angle_kappa_3d = np.arccos(abs(dot_product)) * 180.0 / np.pi

print(f"Dot product = {dot_product:.6f}")
print(f"Angle kappa = arccos(|{dot_product:.6f}|) = {angle_kappa_3d:.3f}°")

# STEP 5: Simple formula calculation
print(f"\nSTEP 5: Simple formula calculation")
angle_kappa_simple = np.sqrt(e.fovea_alpha_deg**2 + e.fovea_beta_deg**2)
print(f"angle_kappa = sqrt(α² + β²)")
print(f"            = sqrt({e.fovea_alpha_deg}² + {e.fovea_beta_deg}²)")
print(f"            = sqrt({e.fovea_alpha_deg**2} + {e.fovea_beta_deg**2})")
print(f"            = sqrt({e.fovea_alpha_deg**2 + e.fovea_beta_deg**2})")
print(f"            = {angle_kappa_simple:.3f}°")

# STEP 6: Why they're the same
print(f"\nSTEP 6: Why the simple formula works")
print(f"For small angles (< 10°), the small angle approximation applies:")
print(
    f"- sin(α) ≈ α (in radians) → sin({e.fovea_alpha_deg}°) = {np.sin(alpha_rad):.6f} ≈ {alpha_rad:.6f}"
)
print(f"- cos(α) ≈ 1 → cos({e.fovea_alpha_deg}°) = {np.cos(alpha_rad):.6f} ≈ 1")
print(f"- cos(β) ≈ 1 → cos({e.fovea_beta_deg}°) = {np.cos(beta_rad):.6f} ≈ 1")

print(f"\nTherefore:")
print(f"Distance from optical axis ≈ R × sqrt(α² + β²)")
print(f"Angle kappa ≈ arctan(distance/R) ≈ sqrt(α² + β²) (for small angles)")

print(f"\nCOMPARISON:")
print(f"Eye.angle_kappa property: {angle_kappa_eye_property:.3f}°")
print(f"Manual 3D method:         {angle_kappa_3d:.3f}°")
print(f"Simple method:            {angle_kappa_simple:.3f}°")
print(f"Difference (3D vs Simple): {abs(angle_kappa_3d - angle_kappa_simple):.6f}°")
print(
    f"Error:                     {abs(angle_kappa_3d - angle_kappa_simple)/angle_kappa_simple*100:.3f}%"
)

print(f"\nCONCLUSION:")
print(f"For practical eye tracking with typical fovea displacements,")
print(f"angle_kappa = sqrt(fovea_alpha_deg² + fovea_beta_deg²) is excellent!")
