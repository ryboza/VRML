import numpy as np
import pymeshlab
import pyvista as pv
class MeshVoxel:
    def __init__(self, PML_bounding_box : pymeshlab.BoundingBox, i_oversize = 1.2):
        """
        Initialize the voxel with vertices.
        
        Parameters:
        """
        self.oversize = i_oversize
        self.center = PML_bounding_box.center()
        self.diagonal =  PML_bounding_box.diagonal()
        self.dim_x = PML_bounding_box.dim_x()
        self.dim_y = PML_bounding_box.dim_y()
        self.dim_z = PML_bounding_box.dim_z()
        self.dim = np.array([self.dim_x, self.dim_y, self.dim_z])

        self.max = PML_bounding_box.max()
        self.min = PML_bounding_box.min()
        self.max_os = PML_bounding_box.max()+i_oversize     
        self.min_os = PML_bounding_box.min()-i_oversize      
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
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = self.center[0] + (-1)**i * self.dim_x * self.oversize / 2
                    y = self.center[1] + (-1)**j * self.dim_y * self.oversize / 2
                    z = self.center[2] + (-1)**k * self.dim_z * self.oversize / 2
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
    

    def plot_edges_3d_pyvista(self, plotter=None):
        # Initialize plotter if None
        if plotter is None:
            plotter = pv.Plotter()
            # Create a PolyData object
        polydata = pv.PolyData()
        
         # Define the voxel faces using vertices indices
        faces = np.array([
            [4, 0, 1, 3, 2],  # Bottom face
            [4, 4, 5, 7, 6],  # Top face
            [4, 0, 1, 5, 4],  # Front face
            [4, 2, 3, 7, 6],  # Back face
            [4, 0, 2, 6, 4],  # Left face
            [4, 1, 3, 7, 5]   # Right face
        ])       
        
        # Convert vertices to a pyvista-friendly format
        points = self.oversized_vertices
        
        mesh = pv.PolyData(self.oversized_vertices, faces)
        
        # Add the polyline to the plotter
        plotter.add_mesh(mesh, color='blue', line_width=2, opacity=0.02)

        # Add the vertices to the plotter
        plotter.add_points(points, color='yellow', point_size=20)

        return plotter       
def get_bounding_box(vertices):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    return min_coords, max_coords   
 
def check_overlap(voxel1, voxel2):
    vertices1 = voxel1.oversized_vertices
    vertices2 = voxel2.oversized_vertices
    
    min1, max1 = get_bounding_box(vertices1)
    min2, max2 = get_bounding_box(vertices2)
    
    overlap_x = (min1[0] <= max2[0] and max1[0] >= min2[0])
    overlap_y = (min1[1] <= max2[1] and max1[1] >= min2[1])
    overlap_z = (min1[2] <= max2[2] and max1[2] >= min2[2])
    
    return overlap_x and overlap_y and overlap_z