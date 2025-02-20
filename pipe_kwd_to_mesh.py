from ansys.dyna.core.lib.deck import Deck
from ansys.dyna.core.keywords import keywords
from pathlib import Path
import pandas as pd
import pyvista as pv
import numpy as np
import streamlit as st
import platform
#pv.start_xvfb()

def process_kwd_to_mesh(keyword_file):
    deck = Deck()
    deck.loads(keyword_file)

    nodes = deck.get(type="NODE", filter=lambda kwd: kwd.subkeyword == "NODE")[0]
    
    mesh_geometry = nodes.cards[0].table[['x', 'y']]

    elements = deck.get(type="ELEMENT", filter=lambda kwd: kwd.subkeyword == "SHELL")[0]

    mesh_topology = elements.cards[0].table[['n1', 'n2', 'n3']]

    return deck, mesh_geometry, mesh_topology

def plot_mesh(mesh_geometry, mesh_topology):
    nodes_3d = mesh_geometry.copy()
    nodes_3d['z'] = 0  # Flat plane mesh
    nodes = nodes_3d[['x', 'y', 'z']].values  # Get coordinates as an array
    mesh_topology_copy = mesh_topology.copy()
    
    # The connectivity (elements) with 1-based indexing
    elements = mesh_topology_copy[['n1', 'n2', 'n3']].values - 1  # Convert to 0-based indexing for PyVista
    
    # Prepare the element connectivity for PyVista: [3, n1, n2, n3, 3, n4, n5, n6, ...]
    faces = []
    for row in elements:
        faces.append([3, row[0], row[1], row[2]])  # Each row starts with the number of vertices for that face (3 for triangles)
    faces = np.array(faces).flatten()  # Flatten the list to match PyVista's format

    # Create a PyVista mesh
    mesh = pv.PolyData(nodes)  # Create the mesh with node coordinates
    mesh.faces = faces  # Assign faces (connectivity)
    
    # Create an array to store the color values for each node (default: 0 - no color)
    colors = np.zeros(mesh.n_points)
    
    # Condition for red color: y < 0.001
    red_condition = nodes_3d['y'] < 0.001
    colors[red_condition] = 1  # Set color value 1 for red nodes

    # Condition for blue color: y > 499.999
    blue_condition = nodes_3d['y'] > 499.999
    colors[blue_condition] = 2  # Set color value 2 for blue nodes

    # Ensure that the color array is properly assigned to the mesh
    mesh.point_arrays["Node Colors"] = colors

    # Check if the point_arrays are properly added
    if "Node Colors" not in mesh.point_arrays:
        print("Error: Node Colors not added to the mesh.")

    # Set up the plotter and show the mesh
    plotter = pv.Plotter(off_screen=True, window_size=[600, 600])
    plotter.add_mesh(mesh, show_edges=True, scalars="Node Colors", cmap="coolwarm", point_size=10)
    plotter.view_isometric()
    plotter.background_color = 'white'
    
    return plotter
