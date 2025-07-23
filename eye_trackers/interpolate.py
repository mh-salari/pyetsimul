from interpolate.et_setup import InterpolationTracker
from et_simul.performance_analysis import (
    accuracy_over_gaze_points,
    accuracy_over_observer_positions,
)
from et_simul.visualization.draw import draw_setup
from et_simul.core import Light
import numpy as np


def main():
    print("=== Python Interpolate Test (System Integration) ===\n")

    # 1D polynomials
    et = InterpolationTracker.setup("cerrolaza_2008")  # Default
    # et = InterpolationTracker.setup('hennessey_2008')
    # et = InterpolationTracker.setup('second_order')

    # 2D polynomials
    # et = InterpolationTracker.setup('zhu_ji_2005')
    # et = InterpolationTracker.setup('blignaut_wium_2013')

    # et.lights[0] = Light(position=np.array([200e-3, 0, 0, 1]))
    print("Visualizing eye tracker setup...")
    draw_setup(et)

    print("1. Testing over screen (fixed observer, sweep gaze positions):")
    print("-" * 60)
    screen_results = accuracy_over_gaze_points(et)

    print(f"\nScreen Test Summary:")
    print(f"  Error statistics (mm):")
    print(f"    Max:    {screen_results['mtr']['max']*1e3:.4f}")
    print(f"    Mean:   {screen_results['mtr']['mean']*1e3:.4f}")
    print(f"    Std:    {screen_results['mtr']['std']*1e3:.4f}")
    print(f"    Median: {screen_results['mtr']['median']*1e3:.4f}")
    print(f"  Error statistics (degrees):")
    print(f"    Max:    {screen_results['deg']['max']:.4f}")
    print(f"    Mean:   {screen_results['deg']['mean']:.4f}")
    print(f"    Std:    {screen_results['deg']['std']:.4f}")
    print(f"    Median: {screen_results['deg']['median']:.4f}")

    print(f"\n2. Testing over observer (fixed gaze, sweep observer positions):")
    print("-" * 60)
    observer_results = accuracy_over_observer_positions(et)

    print(f"\nScreen Test Summary:")
    print(f"  Error statistics (mm):")
    print(f"    Max:    {observer_results['mtr']['max']*1e3:.4f}")
    print(f"    Mean:   {observer_results['mtr']['mean']*1e3:.4f}")
    print(f"    Std:    {observer_results['mtr']['std']*1e3:.4f}")
    print(f"    Median: {observer_results['mtr']['median']*1e3:.4f}")
    print(f"  Error statistics (degrees):")
    print(f"    Max:    {observer_results['deg']['max']:.4f}")
    print(f"    Mean:   {observer_results['deg']['mean']:.4f}")
    print(f"    Std:    {observer_results['deg']['std']:.4f}")
    print(f"    Median: {observer_results['deg']['median']:.4f}")


if __name__ == "__main__":
    main()
