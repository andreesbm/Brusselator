import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pde import PDE, FieldCollection, ScalarField, CartesianGrid, MemoryStorage

# Load settings from external JSON file
with open('c:/Users/Ott/Brusselator/settings.json', 'r') as f:
    settings = json.load(f)

# Constants from settings
RESOLUTION = settings["resolution"]
FRAME_RATE = settings["frame_rate"]
OSCILLATION_PERIOD = settings["oscillation_period"]
NUM_OSCILLATIONS = settings["num_oscillations"]
COLOR_VMIN = settings["color_vmin"]
COLOR_VMAX = settings["color_vmax"]
U_COLOR = settings["u_color"]
V_COLOR = settings["v_color"]

# Modes from settings
modes = settings["modes"]

# Calculate total simulation time and time step
T_MAX = NUM_OSCILLATIONS * OSCILLATION_PERIOD
DT = OSCILLATION_PERIOD / (FRAME_RATE * 2)  # Time step to get smooth frames for video

# Create results directory
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Determine the next render number
render_numbers = [int(name) for name in os.listdir(results_dir) if name.isdigit()]
render_number = max(render_numbers, default=0) + 1
render_dir = os.path.join(results_dir, str(render_number))
os.makedirs(render_dir, exist_ok=True)

storage_dict = {}  # Store the MemoryStorage objects for each mode

for mode in modes:
    # Extract parameters
    title = mode["title"]
    a = mode["a"]
    b = mode["b"]
    d0 = mode["d0"]
    d1 = mode["d1"]
    filename = mode["filename"]
    description = mode["description"]

    # Define the PDE
    eq = PDE(
        {
            "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
            "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
        }
    )

    # Initialize state with reflective boundary conditions
    RADIUS = RESOLUTION // 2
    grid = CartesianGrid([[-RADIUS, RADIUS], [-RADIUS, RADIUS]], [RESOLUTION, RESOLUTION], periodic=False)

    # Create initial fields
    u = ScalarField(grid, a, label="Field $u$")
    v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")

    # Create a circular mask for the Dirichlet boundary condition
    center = (grid.shape[0] // 2, grid.shape[1] // 2)
    Y, X = np.ogrid[:grid.shape[0], :grid.shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    circular_mask = dist_from_center <= RADIUS

    # Apply the mask to enforce the Dirichlet boundary conditions
    u.data[~circular_mask] = 0
    v.data[~circular_mask] = 0

    # Create a state collection
    state = FieldCollection([u, v])

    # Directory to save frames
    frames_dir = os.path.join(render_dir, f'frames_{title.replace(" ", "_").lower()}')
    os.makedirs(frames_dir, exist_ok=True)

    # Create a MemoryStorage object to store the simulation state
    storage = MemoryStorage()

    # Simulate the PDE with storage tracker
    sol = eq.solve(state, t_range=T_MAX, dt=DT, tracker=storage.tracker(interval=1))
    storage_dict[title] = list(storage.items())  # Store the items in a list

    # Save each state as an image
    for frame_idx, (time, state) in enumerate(storage_dict[title]):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Apply the circular mask to the data
        u_data = np.ma.masked_where(~circular_mask, state[0].data)
        v_data = np.ma.masked_where(~circular_mask, state[1].data)
        
        # Plot the u field with the specified colormap
        u_plot = ax.imshow(u_data, cmap=U_COLOR, alpha=0.6, vmin=COLOR_VMIN, vmax=COLOR_VMAX, extent=[-RADIUS, RADIUS, -RADIUS, RADIUS])
        
        # Plot the v field with the specified colormap
        v_plot = ax.imshow(v_data, cmap=V_COLOR, alpha=0.6, vmin=COLOR_VMIN, vmax=COLOR_VMAX, extent=[-RADIUS, RADIUS, -RADIUS, RADIUS])
        
        # Add colorbars with labels below
        cbar_u = plt.colorbar(u_plot, ax=ax, fraction=0.046, pad=0.08)
        cbar_u.ax.set_ylabel('Compound X', labelpad=10)
        
        cbar_v = plt.colorbar(v_plot, ax=ax, fraction=0.046, pad=0.14)
        cbar_v.ax.set_ylabel('Compound Y', labelpad=10)
        
        # Set title
        plt.title(title)
        
        # Set axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Add parameters text in a box at the bottom left within the plot with distance from axes
        params_text = f'a = {a}\nb = {b}\nd0 = {d0}\nd1 = {d1}'
        ax.text(-RADIUS + 5, -RADIUS + 5, params_text, ha='left', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        
        # Add description below the plot
        plt.figtext(-RADIUS - 5, -RADIUS + 5, description, ha="center", fontsize=10, wrap=True)

        # Save the frame
        frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
        plt.savefig(frame_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved frame {frame_idx} at time {time}")

    # Custom directory to save the video file
    video_path = os.path.join(render_dir, filename)

    # Compile frames into a video
    first_frame = cv2.imread(os.path.join(frames_dir, 'frame_0000.png'))
    if first_frame is None:
        raise ValueError("First frame not found. Check if the frames are being saved correctly.")
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Initialize video writer
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), FRAME_RATE, frame_size)

    # Write frames to video
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        if frame is None:
            print(f"Error reading frame {frame_file}. Skipping.")
            continue
        out.write(frame)

    out.release()

    print(f"Video saved to {video_path}")

# Create a video with a 2x2 grid of the four modes
grid_video_path = os.path.join(render_dir, 'overview_phases.avi')
out_grid = cv2.VideoWriter(grid_video_path, cv2.VideoWriter_fourcc(*'XVID'), FRAME_RATE, (2 * width, 2 * height))

num_frames = len(storage_dict[modes[0]["title"]])

for frame_idx in range(num_frames):
    combined_frame = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
    
    for i, mode in enumerate(modes):
        frames_dir = os.path.join(render_dir, f'frames_{mode["title"].replace(" ", "_").lower()}')
        frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Error reading frame {frame_path}. Skipping.")
            continue
        
        if i == 0:
            combined_frame[0:height, 0:width] = frame
        elif i == 1:
            combined_frame[0:height, width:2 * width] = frame
        elif i == 2:
            combined_frame[height:2 * height, 0:width] = frame
        elif i == 3:
            combined_frame[height:2 * height, width:2 * width] = frame

    out_grid.write(combined_frame)

out_grid.release()

print(f"Overview video saved to {grid_video_path}")