import pymeshlab
from voxel import MeshVoxel 
import CasSurface
#import matplotlib.pyplot as plt
import numpy as np
class Cas:
    def __init__(self):
        self.DomainCollector = []
        self.adjacency_matrix = []
        self.ms = pymeshlab.MeshSet()   
        self.threshold = 0.001    
        self.ms.load_new_mesh("big.wrl")      
        self.ms.generate_splitting_by_connected_components(delete_source_mesh = True)
        self.ms.get_geometric_measures()
        self.ms.__str__()
        self.ax = None
        for i  in range(1, self.ms.mesh_number()+1):
            self.ms.set_current_mesh(i)
            #set output object
            surface = CasSurface.CasSurface(self.ms, self.ms.current_mesh())            
            self.DomainCollector.append(surface)
            #self.ax = surface.plot_edges_3d(self.ax)
            self.ax = surface.plot_edges_3d_pyvista(self.ax)

    def build_adjacency_matrix(self):
        n_surfaces = len(self.DomainCollector)
        self.adjacency_matrix = np.zeros((n_surfaces, n_surfaces), dtype=int)

        # Measure distances between surfaces using pyvista
        for i, surface1 in enumerate(self.DomainCollector):
            for j, surface2 in enumerate(self.DomainCollector):
                if i != j:  # Avoid comparing a surface to itself
                    
                    closest_cells, closest_points = surface2.PV_mesh.find_closest_cell(surface1.PV_mesh.points, return_closest_point=True)
                    distance = np.linalg.norm(surface1.PV_mesh.points - closest_points, axis=1)
                    if (distance <= self.threshold).any():
                        self.adjacency_matrix[i, j] = 1
                else:
                    self.adjacency_matrix[i, j] = 1 #1 on diagonal
        return self.adjacency_matrix 


            
def measure_distance(point1, point2):
    return np.sum((point1 - point2)**2)
    #return np.linalg.norm(point1 - point2)

def build_adjacency_matrix(casobject, threshold_distance=0.002):
    threshold_distance = threshold_distance**2
    adjacency_matrix = np.zeros((len(casobject.DomainCollector), len(casobject.DomainCollector)))

    for surface1_idx, surface1 in enumerate(casobject.DomainCollector):
        for surface2_idx, surface2 in enumerate(casobject.DomainCollector):
            if surface1_idx == surface2_idx:
                adjacency_matrix[surface1_idx, surface2_idx] = 1
                continue
            
            surface1_points = surface1.ML_edge_vertex_matrix
            surface2_points = surface2.ML_edge_vertex_matrix
            
            for point1 in surface1_points:
                for point2 in surface2_points:
                    distance = measure_distance(point1, point2)
                    if distance < threshold_distance:
                        adjacency_matrix[surface1_idx, surface2_idx] = 1
                        adjacency_matrix[surface2_idx, surface1_idx] = 1
                        break  # Break loop if at least one point meets the threshold
    
    return adjacency_matrix           


def check_voxel_overlap(voxel1, voxel2):
    """
    Check if two voxels overlap.

    Parameters:
        voxel1: MeshVoxel
            First MeshVoxel object.
        voxel2: MeshVoxel
            Second MeshVoxel object.

    Returns:
        bool
            True if the voxels overlap, False otherwise.
    """
    # Calculate the boundaries of the two voxels
    voxel1_min = voxel1.center - voxel1.dim / 2
    voxel1_max = voxel1.center + voxel1.dim / 2
    voxel2_min = voxel2.center - voxel2.dim / 2
    voxel2_max = voxel2.center + voxel2.dim / 2

    # Calculate the boundaries of the intersection
    intersection_min = np.maximum(voxel1.min_os, voxel2.min_os)
    intersection_max = np.minimum(voxel1.max_os, voxel2.max_os)

    # Calculate the dimensions of the intersection
    intersection_dim = np.maximum(0, intersection_max - intersection_min)

    # Calculate the volume of the intersection
    intersection_volume = np.prod(intersection_dim)

    # If the volume of the intersection is greater than 0, the voxels overlap
    return intersection_volume > 0



def build_adjacency_matrix_voxel(casobject):
    """
    Build an adjacency matrix based on voxel overlapping.

    Parameters:
        voxels: list
            List of MeshVoxel objects.
        threshold_overlap: float, optional (default=0.002)
            Threshold for voxel overlapping.

    Returns:
        np.ndarray
            Adjacency matrix indicating the adjacency between voxels.
    """
    num_voxels = len(casobject.DomainCollector)
    adjacency_matrix = np.zeros((len(casobject.DomainCollector), len(casobject.DomainCollector)))

    for voxel1_idx, surface1 in enumerate(casobject.DomainCollector):
        for voxel2_idx, surface2 in enumerate(casobject.DomainCollector):
            if voxel1_idx == voxel2_idx:
                adjacency_matrix[voxel1_idx, voxel2_idx] = 1
                continue        

            voxel1 = surface1.voxel
            voxel2 = surface2.voxel  
                 
            if check_voxel_overlap(voxel1, voxel2):
                adjacency_matrix[voxel1_idx, voxel2_idx] = 1
                adjacency_matrix[voxel2_idx, voxel1_idx] = 1       
            
    return adjacency_matrix                
        

def measure_distance(self, surface1, surface2):
    # Convert surface data to pyvista objects
    mesh1 = surface1.PV_mesh
    mesh2 = surface2.PV_mesh

    # Compute distance between surfaces
    distance = mesh1.distance(mesh2)

    return distance
 

 
 
            
casobject =Cas()
casobject.build_adjacency_matrix()


#old method 1 adjacency_matrix = build_adjacency_matrix_voxel(casobject)
#old method 2 adjacency_matrix = build_adjacency_matrix(casobject)

print("Adjacency Matrix:")
print(casobject.adjacency_matrix)



def dfs(adjacency_matrix, visited, current_vertex, current_group):
    visited[current_vertex] = True
    current_group.append(current_vertex)

    for neighbor, connected in enumerate(adjacency_matrix[current_vertex]):
        if connected == 1 and not visited[neighbor]:
            dfs(adjacency_matrix, visited, neighbor, current_group)

def extract_non_adjacent_groups(adjacency_matrix):
    num_surfaces = len(adjacency_matrix)
    visited = [False] * num_surfaces
    non_adjacent_groups = []

    for vertex in range(num_surfaces):
        if not visited[vertex]:
            current_group = []
            dfs(adjacency_matrix, visited, vertex, current_group)
            non_adjacent_groups.append(current_group)

    return non_adjacent_groups




output = extract_non_adjacent_groups(casobject.adjacency_matrix)
print("Non Adjacent Domains:")
print(output)
casobject.ax.show()
pass





