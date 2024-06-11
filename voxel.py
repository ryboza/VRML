import numpy as np
import pymeshlab

class MeshVoxel:
    def __init__(self, PML_bounding_box : pymeshlab.BoundingBox, oversize = 0.0002):
        """
        Initialize the voxel with vertices.
        
        Parameters:
        """
        self.oversize = oversize
        self.center = PML_bounding_box.center()
        self.diagonal =  PML_bounding_box.diagonal()
        self.dim_x = PML_bounding_box.dim_x()
        self.dim_y = PML_bounding_box.dim_y()
        self.dim_z = PML_bounding_box.dim_z()
        self.dim = np.array([self.dim_x, self.dim_y, self.dim_z])

        self.max = PML_bounding_box.max()
        self.min = PML_bounding_box.min()
        self.max_os = PML_bounding_box.max()+oversize     
        self.min_os = PML_bounding_box.min()-oversize       
        # Generate voxel vertices
        self.vertices = self.generate_vertices()
        self.oversized_vertices = self.generate_oversized_vertices()


    
    def __repr__(self):
        """
        Return a string representation of the voxel.
        """
        return f"Voxel(vertices={self.vertices})"


    def generate_vertices(self):
        vertices = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = self.center[0] + (-1)**i * self.dim_x / 2
                    y = self.center[1] + (-1)**j * self.dim_y / 2
                    z = self.center[2] + (-1)**k * self.dim_z / 2
                    vertices.append([x, y, z])
        return np.array(vertices)

    def generate_oversized_vertices(self):
        """
        Generate oversized vertices of the voxel based on self.oversize factor.
        """
        oversized_vertices = []
        for vertex in self.vertices:
            x = self.center[0] + (vertex[0] - self.center[0]) * self.oversize
            y = self.center[1] + (vertex[1] - self.center[1]) * self.oversize
            z = self.center[2] + (vertex[2] - self.center[2]) * self.oversize
            oversized_vertices.append([x, y, z])
        return np.array(oversized_vertices)
            
    def generate_3d_matrix(edge_vertex_coordinates):
        # Determine the dimensions of the matrix based on the maximum and minimum coordinates
        max_coords = np.max(edge_vertex_coordinates, axis=0)
        min_coords = np.min(edge_vertex_coordinates, axis=0)
        dims = max_coords - min_coords + 1  # Add 1 to include the last coordinate

        # Create an empty 3D matrix filled with zeros
        matrix = np.zeros(dims, dtype=bool)  # Assuming the coordinates are integer

        # Convert edge vertex coordinates to indices in the matrix
        indices = edge_vertex_coordinates - min_coords

        # Set the corresponding indices in the matrix to True to represent the object
        for index in indices:
            matrix[tuple(index)] = True

        return matrix