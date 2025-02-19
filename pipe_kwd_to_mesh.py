from ansys.dyna.core.lib.deck import Deck
from ansys.dyna.core.keywords import keywords
from pathlib import Path
import pandas as pd

def process_kwd_to_mesh(keyword_file):
    deck = Deck()
    deck.loads(keyword_file)

    nodes = deck.get(type="NODE", filter=lambda kwd: kwd.subkeyword == "NODE")[0]
    
    mesh_geometry = nodes.cards[0].table[['x', 'y']]

    elements = deck.get(type="ELEMENT", filter=lambda kwd: kwd.subkeyword == "SHELL")[0]

    mesh_topology = elements.cards[0].table[['n1', 'n2', 'n3']]

    return deck, mesh_geometry, mesh_topology

def plot_mesh(mesh_geometry, mesh_topology):
    nodes = mesh_geometry[['x', 'y']].values  # Extract coordinates as a numpy array
    
    # Load the element connectivity (node indices for each element)
    elements = mesh_topology[['n1', 'n2', 'n3']].values - 1  # Convert to zero-based indexing
    
    # Create a PyVista mesh from the nodes and elements
    mesh = pv.PolyData(nodes)  # Initialize mesh with node coordinates
    mesh.faces = elements.flatten('F')  # Flatten elements for PyVista faces format
    
    # Now plot the mesh off-screen (use off-screen rendering mode for cloud environments)
    pv.set_plot_theme("dark")  # Optional: Set a theme for better visualization
    plotter = pv.Plotter(off_screen=True)  # Use off-screen rendering mode
    plotter.add_mesh(mesh, color="cyan", show_edges=True)  # Add mesh to plotter
    plotter.show() 
