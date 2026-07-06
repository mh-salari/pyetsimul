"""Generate a labelled dataset: render an eye across many gaze targets and save it to disk.

The simulator can produce a whole dataset in one call. DataGenerationStrategy renders the eye through the camera
at every point of a variation and records the pupil and glint observations together with the ground-truth gaze.
A TargetPositionVariation sweeps the gaze target over a grid on the screen (an EyePositionVariation would instead
sweep the head/eye position). The result is written to JSON, and load_experiment_data reads it back: the
labelled data a gaze-mapping or evaluation step would consume.
"""

from pathlib import Path

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.simulation import DataGenerationStrategy, TargetPositionVariation
from pyetsimul.simulation.data_loading import load_experiment_data
from pyetsimul.types import Position3D

# execute() renders in parallel with multiprocessing; the __main__ guard keeps spawned workers from re-running this
# file on import (the default start method is spawn on macOS and Windows).
if __name__ == "__main__":
    # A generic screen-based eye-tracking setup. Distances in millimetres; frame centred on the screen:
    # +x right, +y from the screen toward the eye, +z up.
    SCREEN_W, SCREEN_H = 380.0, 300.0  # the display the eye looks at
    OUTPUT_DIR = Path(__file__).parent / "outputs"

    base = get_eye_model("PyEtSimul")
    eye = Eye(model=base)
    eye.position = Position3D(0.0, 700.0, 50.0)  # in front of the screen (+y), a little above its centre

    camera = Camera()
    camera.position = Position3D(0.0, 350.0, -150.0)  # between the screen and the eye, below it, looking up
    camera.point_at(eye.position)  # aim the camera at the eye

    light = Light(position=Position3D(70.0, 350.0, -140.0))  # its corneal reflection is the glint

    # The strategy renders the scene at every point of a variation; save_to_file writes it to outputs/<name>.json.
    data_gen = DataGenerationStrategy(
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=Position3D(0.0, 0.0, 0.0),  # nominal fixation (screen centre); each variation point overrides it
        experiment_name="gaze_grid",
        output_dir=OUTPUT_DIR,
    )

    # Sweep the gaze target over a grid on the screen plane (y = 0); grid_size is [nx, ny, nz].
    gaze_grid = TargetPositionVariation(
        grid_center=Position3D(0.0, 0.0, 0.0),
        dx=[-SCREEN_W / 2, SCREEN_W / 2],
        dy=[0.0, 0.0],
        dz=[-SCREEN_H / 2, SCREEN_H / 2],
        grid_size=[9, 1, 7],
    )
    result = data_gen.execute(gaze_grid)
    print(f"generated {result['total_measurements']} measurements, saved to {result['saved_files']}")

    # Read the saved dataset back: the labelled pupil/glint observations a later step would consume.
    loaded = load_experiment_data("gaze_grid", OUTPUT_DIR)
    print(f"loaded {loaded['total_measurements']} measurements from {OUTPUT_DIR}")
