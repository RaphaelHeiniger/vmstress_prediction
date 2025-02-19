from ansys.dyna.core.lib.deck import Deck
from ansys.dyna.core.keywords import keywords
from pathlib import Path
import pandas as pd

def process_kwd_to_mesh(keyword_file):
    deck = Deck()
    deck.loads(keyword_file)

    nodes = deck.get(type="NODE", filter=lambda kwd: kwd.subkeyword == "NODE")[0]
    mesh_geometry = nodes.cards[0].table[['x', 'y']].to_csv(header=None, index=False)

    elements = deck.get(type="ELEMENT", filter=lambda kwd: kwd.subkeyword == "SHELL")[0]

    mesh_topology = elements.cards[0].table[['n1', 'n2', 'n3']].to_csv(header=None, index=False)

    return deck, mesh_geometry, mesh_topology
