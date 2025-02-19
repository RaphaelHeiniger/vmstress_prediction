from ansys.dyna.core.lib.deck import Deck
from ansys.dyna.core.keywords import keywords
from pathlib import Path
import pandas as pd
import pyvista as pv

def process_kwd_to_mesh(keyword_file):
    deck = Deck()
    deck.loads(keyword_file)

    nodes = deck.get(type="NODE", filter=lambda kwd: kwd.subkeyword == "NODE")[0]
    
    mesh_geometry = nodes.cards[0].table[['x', 'y']]

    elements = deck.get(type="ELEMENT", filter=lambda kwd: kwd.subkeyword == "SHELL")[0]

    mesh_topology = elements.cards[0].table[['n1', 'n2', 'n3']]

    return deck, mesh_geometry, mesh_topology

def plot_mesh(mesh_geometry, mesh_topology):
    nodes_3d = mesh_geometry
    nodes_3d['z'] = 0
    nodes = nodes_3d[['x', 'y', 'z']].values
    
    elements = mesh_topology[['n1', 'n2', 'n3']].values - 1
    mesh = pv.PolyData(nodes)
    mesh.faces = elements.flatten('F')
    
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="cyan", show_edges=True)
    plotter.show() 
