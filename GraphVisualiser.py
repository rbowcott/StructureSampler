import networkx as nx
from matplotlib import pyplot as plt
from collections import Counter

def visualise_top_n(all_visited, n, labels):
    state_counts = Counter(all_visited)
    top_n_states = state_counts.most_common(n)

    fig, ax = plt.subplots(n//3, 3, figsize=(30, 20))
    axes = ax.flat

    for i in range(n):
        state, visits = top_n_states[i]
        state = state.cpu().detach().numpy()

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
        
        axes[i].set_title(f'Visits: {visits}')

    for a in axes:
        a.margins(0.2)
    fig.tight_layout()
    plt.savefig('Top_N_Graphs.png')