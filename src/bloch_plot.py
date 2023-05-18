import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_bloch_sphere(states):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw the sphere, axes, and labels
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="b", alpha=0.1)
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color="k", linewidth=1)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color="k", linewidth=1)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color="k", linewidth=1)
    ax.text(1.3, 0, 0, r"$X$", fontsize=14)
    ax.text(0, 1.3, 0, r"$Y$", fontsize=14)
    ax.text(0, 0, 1.3, r"$Z$", fontsize=14)

    # Initialize the state vector plot
    (state_line,) = ax.plot([], [], [], color="r", linewidth=2)

    def init():
        state_line.set_data([], [])
        state_line.set_3d_properties([])
        return (state_line,)

    def update(frame):
        # Check if it is the first frame
        if frame == 0:
            # Hold the first state for 2 seconds
            time.sleep(2)

        # Calculate the coordinates of the state vector on the Bloch sphere for the current frame
        state = states[frame]
        theta = 2 * np.arccos(state[0])
        phi = 2 * np.pi * state[1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        state_line.set_data([0, x], [0, y])
        state_line.set_3d_properties([0, z])

        # Check if it is the last frame
        if frame == len(states) - 1:
            animation_obj.event_source.stop()  # Stop the animation

        return (state_line,)

    animation_obj = animation.FuncAnimation(
        fig, update, frames=len(states), init_func=init, blit=True
    )

    ax.set_xlim3d([-1.2, 1.2])
    ax.set_ylim3d([-1.2, 1.2])
    ax.set_zlim3d([-1.2, 1.2])

    plt.show()


states = [
    (1, 0),  # Pure state along the Z-axis
    (0, 1),  # Pure state along the X-axis
    (1 / np.sqrt(2), 1 / np.sqrt(2)),  # Equatorial state
    (1 / np.sqrt(3), np.sqrt(2 / 3)),  # Arbitrary state
    (-1 / np.sqrt(2), 1 / np.sqrt(2)),  # Equatorial state (opposite direction)
    (0, -1),  # Pure state along the negative X-axis
    (0, 0),  # Null state at the origin
]

plot_bloch_sphere(states)
