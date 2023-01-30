import sys
import pandas as pd
import itertools
from Bio import Phylo
"""
calculate distance matrix from phylogenetic tree.
taken from ecopredict repo of marco galardini:
    https://github.com/mgalardini/ecopredict/blob/master/src/cluster_tree
"""

t = Phylo.read("biofilm_tree", 'newick')

t.root_at_midpoint()

d = {}
for x, y in itertools.combinations(t.get_terminals(), 2):
    v = t.distance(x, y)
    d[x.name] = d.get(x.name, {})
    d[x.name][y.name] = v
    d[y.name] = d.get(y.name, {})
    d[y.name][x.name] = v
for x in t.get_terminals():
    d[x.name][x.name] = 0

m = pd.DataFrame(d)
m = m.loc[m.columns]
m.to_csv("dmat.csv")
