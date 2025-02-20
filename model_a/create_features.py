from pathlib import Path
import pandas as pd
from typing import List, Tuple
import numpy as np
import torch

def apply_boundary_conditions(mesh_geometry, boxes: List[Tuple[
    Tuple[int, int],  # bottom left coordinate of box x1,y1
    Tuple[int, int]] # top right coordinate of box x2,y2
    ] = None) -> None:

    mesh = mesh_geometry.copy()

    mesh["fix"] = 0 # Initialise all nodes to have unconstrained boundaries

    # Check if each point is inside or on the edge of any box
    for (x1, y1), (x2, y2) in boxes:
        inside_box = (mesh["x"] >= x1) & (mesh["x"] <= x2) & (mesh["y"] >= y1) & (mesh["y"] <= y2)
        mesh.loc[inside_box, "fix"] = 1

    boundary_conditions = mesh["fix"]

    return boundary_conditions

def apply_external_loads(mesh_geometry, boxes: List[Tuple[
    Tuple[int, int], # bottom left coordinate of box x1,y1
    Tuple[int, int], # top right coordinate of box x2,y2
    Tuple[int, int]] # fx, fy
    ] = None) -> None:


    mesh = mesh_geometry.copy()

    mesh["fx"] = 0 # Initialise all nodes to have unconstrained boundaries
    mesh["fy"] = 0 # Initialise all nodes to have unconstrained boundaries

    # Check if each point is inside or on the edge of any box
    for (x1, y1), (x2, y2) , (fx, fy)in boxes:
        inside_box = (mesh["x"] >= x1) & (mesh["x"] <= x2) & (mesh["y"] >= y1) & (mesh["y"] <= y2)
        mesh.loc[inside_box, "fx"] = fx
        mesh.loc[inside_box, "fy"] = fy

    external_loads = mesh[["fx", "fy"]]

    return external_loads

def create_edge_features(mesh_geometry, mesh_topology):

    nodes = mesh_geometry.copy()
    nodes = nodes.to_numpy()

    elements = mesh_topology.copy()
    elements = elements.to_numpy()

    elements -= 1
    
    edge_list = set()  # Using a set to avoid duplicate edges
    edge_attr_list = []


    for n1, n2, n3 in elements:
        edges = [(n1, n2), (n2, n3), (n3, n1)]
        for start, end in edges:
            if (end, start) not in edge_list:  # Avoid duplicates in bidirectional edges
                edge_list.add((start, end))
                edge_list.add((end, start))

                dx, dy = nodes[end] - nodes[start]
                norm = np.sqrt(dx**2 + dy**2)
                edge_attr_list.append([dx, dy, norm])
                edge_attr_list.append([-dx, -dy, norm])


    edge_index_np = np.array(list(edge_list))
    edge_attr_np = np.array(edge_attr_list)


    edge_index = torch.tensor(edge_index_np.T, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    
    return edge_index, edge_attr
    
