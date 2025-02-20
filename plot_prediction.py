from pathlib import Path
import pandas as pd
import pyvista as pv
import numpy as np
import streamlit as st
import platform


def plot_prediction(mesh_geometry, mesh_topology, prediction):

    prediction_values = prediction.detach().numpy().flatten()
    nodes_3d = mesh_geometry.copy()
    nodes_3d['z'] = 0  # Flat plane mesh
    nodes = nodes_3d[['x', 'y', 'z']].values
    mesh_topology_copy = mesh_topology.copy()
    # The connectivity (elements) with 1-based indexing
    elements = mesh_topology_copy[['n1', 'n2', 'n3']].values - 1  # Convert to 0-based indexing for PyVista

    faces = []
    for row in elements:
        faces.append([3, row[0], row[1], row[2]])  # Each row starts with the number of vertices for that face (3 for triangles)
    faces = np.array(faces).flatten()

    # Create a PyVista mesh
    mesh = pv.PolyData(nodes)  # Create the mesh with node coordinates
    mesh.faces = faces
    
    mesh.point_data['von Mises stress (MPa)'] = prediction_values

    plotter = pv.Plotter(off_screen=True, window_size=[600, 600])
    plotter.add_mesh(mesh, show_edges=True, scalars='von Mises stress (MPa)', cmap='jet', point_size=1.5)
    plotter.add_scalar_bar()
    plotter.view_isometric()
    plotter.background_color = 'white'
    
    return plotter
