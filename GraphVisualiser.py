import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

def visualise_top_n(all_visited, n_graphs, labels, its, true_likelihood = None):
    state_counts = Counter(tuple(map(tuple, g.tolist())) for g in all_visited)
    top_n_states = state_counts.most_common(n_graphs)

    fig, ax = plt.subplots(n_graphs//3, 3, figsize=(30, 3 * n_graphs))
    axes = ax.flat

    for i in range(n_graphs):
        state, visits = top_n_states[i]
        empirical_likelihood = visits / its
        state = np.array(state)
        idx = int("".join(map(str,state.flatten())), 2)

        G = nx.from_numpy_array(state, create_using=nx.DiGraph) 

        labels_to_nodes = {i: lab for i, lab in enumerate(labels)}
        G = nx.relabel_nodes(G, labels_to_nodes)

        pos = nx.kamada_kawai_layout(G)
        nx.draw(G,
                pos,
                with_labels = True,
                node_size = 2000,
                node_color="tab:orange",
                arrowsize=75,
                ax = axes[i])
        
        if true_likelihood is not None:
            axes[i].set_title(f'Visits: {visits} \n Empirical Likelihood: {empirical_likelihood} \n True Likelihood: {true_likelihood[idx]}')
        else:
            axes[i].set_title(f'Visits: {visits} \n Empirical Likelihood: {empirical_likelihood}')

    for a in axes:
        a.margins(0.2)
    fig.tight_layout()
    plt.savefig('Top_N_Graphs.png')
