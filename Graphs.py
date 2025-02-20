import networkx as nx
import pickle
import matplotlib.pyplot as plt
from collections import Counter

with open('out.pkl', 'rb') as f:
    losses, zs, all_visited = pickle.load(f)

n = 1
state_counts = Counter(all_visited)
top_n_states = state_counts.most_common(n)

# G = nx.from_numpy_array(top_n_states[0], create_using=nx.DiGraph)
