import numpy as np
from voxel import *
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