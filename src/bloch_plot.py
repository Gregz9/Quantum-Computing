import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_bloch_sphere(state):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw the sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="b", alpha=0.1)

    # Draw the x, y, and z axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color="k", linewidth=1)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color="k", linewidth=1)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color="k", linewidth=1)

    # Label the x, y, and z axes
    ax.text(1.3, 0, 0, r"$X$", fontsize=14)
    ax.text(0, 1.3, 0, r"$Y$", fontsize=14)
    ax.text(0, 0, 1.3, r"$Z$", fontsize=14)

    # Calculate the coordinates of the state vector on the Bloch sphere
    theta = 2 * np.arccos(state[0])
    phi = 2 * np.pi * state[1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Draw the state vector
    ax.plot([0, x], [0, y], [0, z], color="r", linewidth=2)

    # Set the limits of the axes
    ax.set_xlim3d([-1.2, 1.2])
    ax.set_ylim3d([-1.2, 1.2])
    ax.set_zlim3d([-1.2, 1.2])

    plt.show()


# Define the state vector
state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

# Plot the state vector on the Bloch sphere
plot_bloch_sphere(state)
