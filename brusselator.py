import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
import cv2
from pde import CartesianGrid, ScalarField
from multiprocessing import Pool, cpu_count

def T(q, a, b, d0, d1):
    return b - 1 - np.linalg.norm(q) ** 2 * d0 - a ** 2 - np.linalg.norm(q) ** 2 * d1

def Delta(q, a, b, d0, d1):
    term1 = (b - 1 - np.linalg.norm(q) ** 2 * d0)
    term2 = (-a ** 2 - np.linalg.norm(q) ** 2 * d1)
    return term1 * term2 + b * a ** 2

def omega(q, a, b, d0, d1):
    T_q = T(q, a, b, d0, d1)
    Delta_q = Delta(q, a, b, d0, d1)
    discriminant = np.sqrt(complex(T_q ** 2 - 4 * Delta_q))
    omega_plus = 0.5 * (T_q + discriminant)
    omega_minus = 0.5 * (T_q - discriminant)
    return omega_plus, omega_minus

def c_plus(q):
    return 0.01*np.random.normal()

def c_minus(q):
    return 0.01*np.random.normal()

def process_frame(frame_data):
    try:
        frame_idx, state, title, render_dir, circular_mask, settings, RADIUS, description, a, b, d0, d1 = frame_data

        frames_dir = os.path.join(render_dir, f'frames_{title.replace(" ", "_").lower()}')
        os.makedirs(frames_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 8))

        u_data = np.ma.masked_where(~circular_mask, state[0])
        v_data = np.ma.masked_where(~circular_mask, state[1])

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
        plt.close(fig)  # Ensure the figure is closed after saving
        
        logging.info(f"Frame {frame_idx} saved for mode {title} at {frame_path}")

        return frame_path
    except Exception as e:
        logging.error(f"Error processing frame {frame_idx} for mode {title}: {e}")
        return None

def process_mode(mode, render_dir, settings):
    title = mode["title"]
    a = mode["a"]
    b = mode["b"]
    d0 = mode["d0"]
    d1 = mode["d1"]
    filename = mode["filename"]
    description = mode["description"]

    logging.info(f"Starting mode {title}")

    RADIUS = 1 / settings["zoom_factor"]
    grid = CartesianGrid([[-RADIUS, RADIUS], [-RADIUS, RADIUS]], [settings["resolution"], settings["resolution"]], periodic=not settings["fixed_boundary"])
    center = (grid.shape[0] // 2, grid.shape[1] // 2)
    Y, X = np.ogrid[:grid.shape[0], :grid.shape[1]]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    circular_mask = dist_from_center <= (RADIUS * (settings["resolution"] / (2 * RADIUS)))

    freqs = np.fft.fftfreq(grid.shape[0], d=(2 * RADIUS) / grid.shape[0])
    qx, qy = np.meshgrid(freqs, freqs)
    q_vectors = np.vstack([qx.ravel(), qy.ravel()]).T
    q_vectors = q_vectors[np.where(circular_mask.ravel())]

    state_storage = []

    t_values = np.arange(0, settings["t_max"], settings["dt"])
    for t in t_values:
        u_data = np.ones(grid.shape)*a
        v_data = np.ones(grid.shape)*b/a # Steady state condition
        for q in q_vectors:
            omega_plus, omega_minus = omega(q, a, b, d0, d1)
            u_phase_plus = np.exp(omega_plus * t - 1j *d0* (q[0] * X + q[1] * Y))
            v_phase_plus = np.exp(omega_plus * t - 1j *d1* (q[0] * X + q[1] * Y))
            u_phase_minus = np.exp(omega_minus * t - 1j *d0* (q[0] * X + q[1] * Y))
            v_phase_minus = np.exp(omega_minus * t - 1j *d1* (q[0] * X + q[1] * Y))
            u_data += np.real(c_plus(q) * u_phase_plus + c_minus(q) *u_phase_minus)
            v_data += np.real(c_plus(q) *v_phase_plus + c_minus(q) *v_phase_minus)
        state_storage.append((t, [u_data, v_data]))

    frame_data_list = [
        (frame_idx, state, title, render_dir, circular_mask, settings, RADIUS, description, a, b, d0, d1)
        for frame_idx, (time, state) in enumerate(state_storage)
    ]

    frame_paths = []
    for frame_data in frame_data_list:
        frame_path = process_frame(frame_data)
        if frame_path:
            frame_paths.append(frame_path)

    logging.info(f"Finished frame processing for mode {title}")

    video_path = os.path.join(render_dir, filename)

    if not frame_paths:
        logging.error(f"No frames processed for mode {title}. Skipping video creation.")
        return []

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

def main():
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    required_keys = ["resolution", "frame_rate", "t_max", "dt", "color_vmin", "color_vmax", "u_color", "v_color", "fixed_boundary", "zoom_factor", "modes"]
    missing_keys = [key for key in required_keys if key not in settings]
    if missing_keys:
        raise KeyError(f"Missing required settings: {', '.join(missing_keys)}")

    RESOLUTION = settings["resolution"]
    FRAME_RATE = settings["frame_rate"]
    T_MAX = settings["t_max"]
    DT = settings["dt"]
    COLOR_VMIN = settings["color_vmin"]
    COLOR_VMAX = settings["color_vmax"]
    U_COLOR = settings["u_color"]
    V_COLOR = settings["v_color"]
    FIXED_BOUNDARY = settings["fixed_boundary"]
    ZOOM_FACTOR = settings["zoom_factor"]

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    for mode in settings["modes"]:
        frame_paths = process_mode(mode, results_dir, settings)
        logging.info(f"Processed {len(frame_paths)} frames for mode {mode['title']}")

if __name__ == "__main__":
    main()

