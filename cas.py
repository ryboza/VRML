import pymeshlab
from voxel import *
import cas_surface
#import matplotlib.pyplot as plt
import numpy as np
class Cas:
    def __init__(self, i_WRL, i_threshold):
        self.DomainCollector = []
        self.adjacency_matrix = []
        self.ms = pymeshlab.MeshSet()   
        self.threshold = i_threshold   
        self.ms.load_new_mesh(i_WRL)      
        self.ms.generate_splitting_by_connected_components(delete_source_mesh = True)
        self.ms.get_geometric_measures()
        self.ms.__str__()
        self.ax = None
        for i  in range(1, self.ms.mesh_number()+1):
            self.ms.set_current_mesh(i)
            #set output object
            surface = cas_surface.CasSurface(self.ms, self.ms.current_mesh())            
            self.DomainCollector.append(surface)
            #self.ax = surface.plot_edges_3d(self.ax)
            self.ax = surface.plot_edges_3d_pyvista(self.ax)
            self.ax = surface.voxel.plot_edges_3d_pyvista(self.ax)

    def build_adjacency_matrix(self):
        n_surfaces = len(self.DomainCollector)
        self.adjacency_matrix = np.zeros((n_surfaces, n_surfaces), dtype=int)

        # Measure distances between surfaces using pyvista
        for i, surface1 in enumerate(self.DomainCollector):
            for j, surface2 in enumerate(self.DomainCollector):
                if i != j:  # Avoid comparing a surface to itself
                    #STEP1: check if voxels overlap
                    if check_overlap(surface1.voxel, surface2.voxel):
                        #STEP2: check with pyvista if they are adjacent with given threshold
                        closest_cells, closest_points = surface2.PV_mesh.find_closest_cell(surface1.PV_mesh.points, return_closest_point=True)
                        distance = np.linalg.norm(surface1.PV_mesh.points - closest_points, axis=1)
                        if (distance <= self.threshold).any():
                            self.adjacency_matrix[i, j] = 1
                    else:
                        pass
                else:
                    self.adjacency_matrix[i, j] = 1 #1 on diagonal
        return self.adjacency_matrix 
    
    

