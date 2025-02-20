import io
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader


def create_dataset(mesh_geometry, mesh_topology, boundary, loads, edge_index, edge_attr):
    data_list = []

    df_node_coord = mesh_geometry.copy()
    df_boundary = boundary.copy()
    df_load = loads.copy()

    df_node_features = pd.concat([df_node_coord, df_boundary, df_load], axis=1)

    node_features = torch.tensor(df_node_features.values, dtype=torch.float32)

    df_elements = mesh_topology.copy()
    df_elements = df_elements.astype(np.int64)

    elements = torch.tensor(df_elements.values, dtype=torch.long)

    nodal_coords= torch.tensor(df_node_coord.values, dtype=torch.float32)

    data_list.append(Data(x=node_features,
                          edge_index=edge_index,
                          edge_attr=edge_attr,
                          cells=elements,
                          mesh_pos=nodal_coords))
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    return loader