import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pde import PDE, FieldCollection, ScalarField, CartesianGrid, MemoryStorage
import logging
from multiprocessing import Pool, cpu_count

def setup_logging(render_dir):
    """Set up logging to file."""
    log_file = os.path.join(render_dir, 'processing.log')

    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add formatter to handler
    file_handler.setFormatter(file_formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

def write_settings_to_file(settings, render_dir):
    """Write settings to a text file."""
    settings_path = os.path.join(render_dir, 'settings.txt')
    with open(settings_path, 'w') as f:
        f.write("Settings:\n")
        json.dump(settings, f, indent=4)

def check_for_invalid_values(state_data, title, time_point):
    """Check for invalid values in the state data."""
    if np.isnan(state_data).any() or np.isinf(state_data).any():
        logging.error(f"Invalid values encountered in mode {title} at time {time_point}.")
        return True
    return False

def process_frame(frame_data):
    frame_idx, state, title, render_dir, circular_mask, settings, RADIUS, description, a, b, d0, d1 = frame_data

    frames_dir = os.path.join(render_dir, f'frames_{title.replace(" ", "_").lower()}')
    os.makedirs(frames_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))

    u_data = np.ma.masked_where(~circular_mask, state[0].data)
    v_data = np.ma.masked_where(~circular_mask, state[1].data)

    u_plot = ax.imshow(u_data, cmap=settings["u_color"], alpha=0.6, vmin=settings["color_vmin"], vmax=settings["color_vmax"], extent=[-RADIUS, RADIUS, -RADIUS, RADIUS])

    v_plot = ax.imshow(v_data, cmap=settings["v_color"], alpha=0.6, vmin=settings["color_vmin"], vmax=settings["color_vmax"], extent=[-RADIUS, RADIUS, -RADIUS, RADIUS])

    cbar_u = plt.colorbar(u_plot, ax=ax, fraction=0.046, pad=0.12)
    cbar_u.ax.set_ylabel('Compound X', labelpad=10)

    cbar_v = plt.colorbar(v_plot, ax=ax, fraction=0.046, pad=0.22)
    cbar_v.ax.set_ylabel('Compound Y', labelpad=10)

    plt.title(title, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    params_text = f'a = {a}\nb = {b}\nd0 = {d0}\nd1 = {d1}'
    ax.text(-RADIUS + 0.05, -RADIUS + 0.05, params_text, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.figtext(0.5, 0.06, description, ha="center", fontsize=10, wrap=True, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
    plt.savefig(frame_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    logging.info(f"Frame {frame_idx} saved for mode {title} at {frame_path}")

    return frame_path

def process_mode(mode, render_dir, settings):
    title = mode["title"]
    a = mode["a"]
    b = mode["b"]
    d0 = mode["d0"]
    d1 = mode["d1"]
    filename = mode["filename"]
    description = mode["description"]

    logging.info(f"Starting mode {title}")

    # Define the PDE
    eq = PDE(
        {
            "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
            "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
        }
    )

    # Initialize state with reflective boundary conditions
    RADIUS = 1 / settings["zoom_factor"]
    grid = CartesianGrid([[-RADIUS, RADIUS], [-RADIUS, RADIUS]], [settings["resolution"], settings["resolution"]], periodic=not settings["fixed_boundary"])

    u = ScalarField(grid, a, label="Field $u$")
    v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")

    center = (grid.shape[0] // 2, grid.shape[1] // 2)
    Y, X = np.ogrid[:grid.shape[0], :grid.shape[1]]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    circular_mask = dist_from_center <= (RADIUS * (settings["resolution"] / (2 * RADIUS)))

    if settings["fixed_boundary"]:
        u.data[~circular_mask] = 0
        v.data[~circular_mask] = 0

    state = FieldCollection([u, v])

    storage = MemoryStorage()

    try:
        sol = eq.solve(state, t_range=settings["t_max"], dt=settings["dt"], tracker=storage.tracker(interval=1))
        for time_point, state_data in storage.items():
            if check_for_invalid_values(state_data.data, title, time_point):
                raise ValueError(f"Invalid values encountered in mode {title} at time {time_point}.")
    except (RuntimeWarning, ValueError) as e:
        logging.error(f"Warning or error encountered in mode {title}: {e}")
        return []

    storage_dict = list(storage.items())

    frame_data_list = [
        (frame_idx, state, title, render_dir, circular_mask, settings, RADIUS, description, a, b, d0, d1)
        for frame_idx, (time, state) in enumerate(storage_dict)
    ]

    with Pool(cpu_count()) as pool:
        frame_paths = pool.map(process_frame, frame_data_list)

    logging.info(f"Finished processing mode {title}")

    video_path = os.path.join(render_dir, filename)

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError("First frame not found. Check if the frames are being saved correctly.")
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), settings["frame_rate"], frame_size)

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Error reading frame {frame_path}. Skipping.")
            continue
        out.write(frame)

    out.release()
    logging.info(f"Video saved to {video_path}")

    return frame_paths

def create_overview_video(modes, frame_paths, render_dir, settings):
    logging.info("Creating overview video")
    
    if not all(len(paths) == len(frame_paths[0]) for paths in frame_paths):
        raise ValueError("All modes must have the same number of frames")

    frame_count = len(frame_paths[0])
    first_frame = cv2.imread(frame_paths[0][0])
    height, width, _ = first_frame.shape

    grid_size = (2 * height, 2 * width)
    video_path = os.path.join(render_dir, 'overview_phases.mp4')
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), settings["frame_rate"], grid_size)

    for frame_idx in range(frame_count):
        frame = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)

        for i, paths in enumerate(frame_paths):
            mode_frame_path = paths[frame_idx]
            logging.info(f"Reading frame {frame_idx} for mode {modes[i]['title']} from {mode_frame_path}")
            mode_frame = cv2.imread(mode_frame_path)
            if mode_frame is None:
                logging.warning(f"Error reading frame {frame_idx} of mode {modes[i]['title']}. Skipping.")
                continue
            
            row = i // 2
            col = i % 2
            frame[row * height:(row + 1) * height, col * width:(col + 1) * width] = mode_frame
        
        out.write(frame)

    out.release()
    logging.info(f"Overview video saved to {video_path}")

def main():
    # Load settings from external JSON file
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    # Required keys
    required_keys = ["resolution", "frame_rate", "t_max", "dt", "color_vmin", "color_vmax", "u_color", "v_color", "fixed_boundary", "zoom_factor", "modes"]
    
    # Check for missing keys
    missing_keys = [key for key in required_keys if key not in settings]
    if missing_keys:
        raise KeyError(f"Missing required settings: {', '.join(missing_keys)}")
    
    # Extract constants from settings
    RESOLUTION = settings["resolution"]
    FRAME_RATE = settings["frame_rate"]
    T_MAX = settings["t_max"]
    DT = settings["dt"] / 100  # Further reduce the time step to improve stability
    COLOR_VMIN = settings["color_vmin"]
    COLOR_VMAX = settings["color_vmax"]
    U_COLOR = settings["u_color"]
    V_COLOR = settings["v_color"]
    FIXED_BOUNDARY = settings["fixed_boundary"]
    ZOOM_FACTOR = settings["zoom_factor"]

    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Determine the next render number
    render_numbers = [int(name) for name in os.listdir(results_dir) if name.isdigit()]
    render_number = max(render_numbers, default=0) + 1
    render_dir = os.path.join(results_dir, str(render_number))
    os.makedirs(render_dir, exist_ok=True)

    # Set up logging
    setup_logging(render_dir)

    # Save settings to file before starting any processing
    write_settings_to_file(settings, render_dir)

    # Process modes sequentially
    all_frame_paths = []
    for mode in settings["modes"]:
        frame_paths = process_mode(mode, render_dir, settings)
        all_frame_paths.append(frame_paths)

    # Create overview video if there are exactly 4 modes
    if len(settings["modes"]) == 4:
        create_overview_video(settings["modes"], all_frame_paths, render_dir, settings)

if __name__ == "__main__":
    main()
