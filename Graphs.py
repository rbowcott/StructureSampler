import networkx as nx
import pickle
from collections import Counter

with open('out.pkl', 'rb') as f:
    losses, zs, all_visited = pickle.load(f)

n = 1
state_counts = Counter(all_visited)
top_n_states = state_counts.most_common(n)

for state, count in top_n_states:
    G = nx.from_numpy_array(state, create_using=nx.DiGraph)
