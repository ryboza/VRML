import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to calculate normalized weighted voxel occupancy
def calculate_weighted_voxel_occupancy(edge_vertex_coordinates, voxel_size):
    # Calculate the dimensions of the 3D matrix
    max_coords = np.ceil(np.max(edge_vertex_coordinates, axis=0) / voxel_size).astype(int)
    min_coords = np.floor(np.min(edge_vertex_coordinates, axis=0) / voxel_size).astype(int)
    dims = max_coords - min_coords + 1  # Add 1 to include the last coordinate

    # Create an empty 3D matrix filled with zeros
    matrix = np.zeros(dims)

    # Iterate over each voxel
    for i in range(min_coords[0], max_coords[0] + 1):
        for j in range(min_coords[1], max_coords[1] + 1):
            for k in range(min_coords[2], max_coords[2] + 1):
                # Calculate centroid of the voxel
                centroid = np.array([i, j, k]) * voxel_size + voxel_size / 2
                
                # Iterate over each point
                for point in edge_vertex_coordinates:
                    # Check if the point is within the voxel
                    if (point[0] >= i * voxel_size and point[0] < (i + 1) * voxel_size and
                        point[1] >= j * voxel_size and point[1] < (j + 1) * voxel_size and
                        point[2] >= k * voxel_size and point[2] < (k + 1) * voxel_size):
                        # Calculate distance from centroid
                        distance = np.linalg.norm(point - centroid)
                        # Calculate normalized weight
                        normalized_weight = 1 - np.clip(distance / (np.sqrt(3) * voxel_size / 2), 0, 1)  # Normalize weight
                        matrix[i - min_coords[0], j - min_coords[1], k - min_coords[2]] += normalized_weight

    return matrix

# Define parameters
sphere_radius = 100  # Radius of the sphere in mm
voxel_size = 20  # Size of the voxel in mm
sphere_center1 = [100, 100, 100]  # Center of the sphere
sphere_center2 = [100 + sphere_radius, 100, 100]  # Center of the second sphere
# Function to create a sphere
def create_sphere(radius, center, num_points=50):
    theta = np.linspace(0, np.pi, num_points)
    phi = np.linspace(0, 2*np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)
    return np.array([x.flatten(), y.flatten(), z.flatten()]).T



# Generate edge vertices for the first sphere
sphere_edge_vertices1 = create_sphere(sphere_radius, sphere_center1)

# Generate edge vertices for the second sphere
sphere_edge_vertices2 = create_sphere(sphere_radius, sphere_center2)

# Combine edge vertices of both spheres
combined_edge_vertices = np.vstack((sphere_edge_vertices1, sphere_edge_vertices2))

# Calculate normalized weighted voxel occupancy for both spheres
normalized_weighted_occupancy = calculate_weighted_voxel_occupancy(combined_edge_vertices, voxel_size)

# Plot a 2D slice of normalized weighted voxel occupancy
slice_index = normalized_weighted_occupancy.shape[0] // 2  # Choose a slice index along the z-axis
plt.imshow(normalized_weighted_occupancy[:, :, slice_index], cmap='viridis', origin='lower', extent=[0, normalized_weighted_occupancy.shape[0]*voxel_size, 0, normalized_weighted_occupancy.shape[1]*voxel_size])
plt.colorbar(label='Normalized Weighted Voxel Occupancy')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Slice of Normalized Weighted Voxel Occupancy')
plt.show()
