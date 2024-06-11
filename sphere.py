import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def generate_3d_matrix(edge_vertex_coordinates, voxel_size):
    # Calculate the dimensions of the 3D matrix
    max_coords = np.ceil(np.max(edge_vertex_coordinates, axis=0) / voxel_size).astype(int)
    min_coords = np.floor(np.min(edge_vertex_coordinates, axis=0) / voxel_size).astype(int)
    dims = max_coords - min_coords + 1  # Add 1 to include the last coordinate

    # Create an empty 3D matrix filled with zeros
    matrix = np.zeros(dims, dtype=bool)

    # Convert edge vertex coordinates to indices in the matrix
    indices = ((edge_vertex_coordinates - min_coords * voxel_size) / voxel_size).astype(int)

    # Set the corresponding indices in the matrix to True to represent the object
    for index in indices:
        matrix[tuple(index)] = True

    return matrix

# Function to create a sphere
def create_sphere(radius, center, num_points=100):
    theta = np.linspace(0, np.pi, num_points)
    phi = np.linspace(0, 2*np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)
    return np.array([x.flatten(), y.flatten(), z.flatten()]).T

# Define parameters
sphere_radius = 100  # Radius of the sphere in mm
voxel_size = 5  # Size of the voxel in mm
sphere_center = [100, 100, 100]  # Center of the sphere

# Generate sphere edge vertices
sphere_edge_vertices = create_sphere(sphere_radius, sphere_center)

# Generate 3D matrix
ml_3d_matrix = generate_3d_matrix(sphere_edge_vertices, voxel_size)

# Plot 3D matrix
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*sphere_edge_vertices.T, color='blue', label='Sphere Edge Vertices')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sphere Edge Vertices')
#plt.show()

# Plot 3D matrix as binary volume
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(ml_3d_matrix, edgecolor='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Binary Volume')
plt.show()