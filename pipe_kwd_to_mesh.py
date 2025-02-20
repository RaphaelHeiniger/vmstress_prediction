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
    

    plotter = pv.Plotter(off_screen=True, window_size=[600, 600])
    plotter.add_mesh(mesh, show_edges=True)
    plotter.view_isometric()
    plotter.background_color = 'white'
    
    return plotter

import pyvista as pv
import numpy as np

def plot_b_l(mesh_geometry, mesh_topology):
    # Copy the mesh geometry to avoid modifying the original data
    nodes_3d = mesh_geometry.copy()
    nodes_3d['z'] = 0  # Flat plane mesh
    nodes = nodes_3d[['x', 'y', 'z']].values  # Get coordinates as an array
    
    # Create the mesh topology (elements), converting to 0-based indexing for PyVista
    mesh_topology_copy = mesh_topology.copy()
    elements = mesh_topology_copy[['n1', 'n2', 'n3']].values - 1  # Convert to 0-based indexing for PyVista
    
    # Prepare the element connectivity for PyVista
    faces = []
    for row in elements:
        faces.append([3, row[0], row[1], row[2]])  # Each row starts with the number of vertices for that face (3 for triangles)
    faces = np.array(faces).flatten()  # Flatten the list to match PyVista's format

    # Create a PyVista mesh
    mesh = pv.PolyData(nodes)  # Create the mesh with node coordinates
    mesh.faces = faces  # Assign faces (connectivity)
    
    # Filter the points where y > 499.999
    filtered_points_y_greater = nodes[nodes[:, 1] > 499.999]
    
    # Create direction vectors that point in the positive y direction and scale them by 20
    directions = np.array([[0, 1, 0]] * filtered_points_y_greater.shape[0])  # Arrows pointing in the positive y-direction
    directions = directions * 20  # Scale the direction vectors to have a length of 20
    
    # Filter the points where y < 0.001 and make them blue and bigger
    filtered_points_y_small = nodes[nodes[:, 1] < 0.001]
    
    # Create the plotter
    plotter = pv.Plotter(off_screen=True, window_size=[600, 600])
    
    # Add the regular mesh with the default color and size
    plotter.add_mesh(mesh, show_edges=True)
    
    # Add arrows to the filtered points with y > 499.999
    plotter.add_arrows(filtered_points_y_greater, directions, color="red", mag=1.3, label='Load')
    
    # Add the points with y < 0.001, make them bigger and blue
    plotter.add_points(filtered_points_y_small, color="blue", point_size=15, label='Boundary')
    
    legend = [
    ['Top pressure', 'blue'],  # no custom glyph
    ['Middle point pressure', 'green', 'circle'],  # Using a defaults glyph
    {'label': 'Lower pressure', 'color': 'red', 'face': pv.Box()},
    ]

    plotter.add_legend(legend)

    plotter.view_isometric()
    plotter.background_color = 'white'
    
    return plotter

