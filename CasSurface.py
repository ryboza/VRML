import numpy as np
import pyvista as pv
import pymeshlab
import voxel
class CasSurface:
    def __init__(self, parent_mesh_set : pymeshlab.MeshSet, mesh: pymeshlab.Mesh):
        """
        Initialize the Cas Surface .
        Parameters:
        
        """
        self.PML_parent_mesh_set = parent_mesh_set
        self.PML_mesh = mesh
        self.PML_bounding_box = None

        #set boundary polyline
        self.PML_parent_mesh_set.set_selection_all()
        self.PML_parent_mesh_set.generate_polyline_from_selection_perimeter()
        self.PML_parent_mesh_set.set_selection_none()
        self.ML_edge_matrix = self.PML_parent_mesh_set.current_mesh().edge_matrix() #new mesh was generated and it is always a ms.current mesh
        self.ML_edge_vertex_matrix = self.PML_parent_mesh_set.current_mesh().vertex_matrix() #new mesh was generated and it is always a ms.current mesh
        self.PV_mesh = self.surface_to_pyvista()
        #set bounding box
        self.PML_mesh.update_bounding_box()
        self.PML_bounding_box = self.PML_mesh.bounding_box()

        #set voxel
        self.voxel=voxel.MeshVoxel(self.PML_bounding_box, oversize=0.0000002)


    
    def __repr__(self):
        """
        Return a string representation of the voxel.
        """
        return f"Voxel(vertices={self.vertices})"


    def surface_to_pyvista(self):
        # Convert surface data to pyvista PolyData
        vertices = self.PML_mesh.vertex_matrix()#self.ML_edge_vertex_matrix
        faces = self.PML_mesh.face_matrix()#self.ML_edge_matrix

        # Horizontally stack the reshaped faces
        stacked_faces = np.hstack([np.full((faces.shape[0], 1), 3),faces])
        mesh = pv.PolyData(vertices, stacked_faces)

        return mesh
    def plot_edges(self):
        import matplotlib.pyplot as plt

        from mpl_toolkits.mplot3d import Axes3D
        plt.figure()
        
        # Plot vertices
        plt.scatter(self.ML_edge_vertex_matrix[:, 0], self.ML_edge_vertex_matrix[:, 1], color='red')
        
        # Plot edges
        for edge in self.ML_edge_matrix:
            start_vertex = self.ML_edge_vertex_matrix[edge[0]]
            end_vertex = self.ML_edge_vertex_matrix[edge[1]]
            plt.plot([start_vertex[0], end_vertex[0]], [start_vertex[1], end_vertex[1]], color='blue')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Edge Matrix Plot')
        plt.show()
    import plotly.graph_objects as go    
    def plot_edges_3d(self, ax = None):
        
    # Initialize plot if ax is None
        if ax is None:
            ax = self.go.Figure()
            ax.update_layout(
                title='3D Edge Matrix Plot',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                )
            )    
            
            
        edge_x = []
        edge_y = []
        edge_z = []

        for edge in self.ML_edge_matrix:
            start_vertex = self.ML_edge_vertex_matrix[edge[0]]
            end_vertex = self.ML_edge_vertex_matrix[edge[1]]
            edge_x += [start_vertex[0], end_vertex[0], None]
            edge_y += [start_vertex[1], end_vertex[1], None]
            edge_z += [start_vertex[2], end_vertex[2], None]

        edge_trace =self.go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=2, color='blue'),
            hoverinfo='none',
            mode='lines'
        )

        vertex_trace = self.go.Scatter3d(
            x=self.ML_edge_vertex_matrix[:, 0], y=self.ML_edge_vertex_matrix[:, 1], z=self.ML_edge_vertex_matrix[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            hoverinfo='text'
        )

        ax.add_trace(edge_trace)
        ax.add_trace(vertex_trace)

        return ax            
            
            
    def plot_edges_3d_pyvista(self, plotter=None):
        # Initialize plotter if None
        if plotter is None:
            plotter = pv.Plotter()
            # Create a PolyData object
        polydata = pv.PolyData()
        # Convert vertices to a pyvista-friendly format
        points = self.ML_edge_vertex_matrix
        
        # Create lines from edges
        lines = []
        # Create the lines
        lines = np.hstack([np.array([2, edge[0], edge[1]]) for edge in self.ML_edge_matrix])
        polydata.lines = lines
        # Create the polyline
        polyline = pv.PolyData()
        polyline.points = points
        polyline.lines = lines
        
        # Add the polyline to the plotter
        plotter.add_mesh(polyline, color='blue', line_width=2)

        # Add the vertices to the plotter
        plotter.add_points(points, color='red', point_size=10)

        return plotter            
        