import torch as T
import networkx as nx
import tqdm
from Reward import log_reward

def calculate_true_likelihoods(vars, probs):
    n = len(vars)
    nsq = len(vars)**2

    all_rewards = []
    for i in tqdm.trange(2**nsq):
        in_bin = bin(i)[2:].zfill(nsq)
        as_adj = T.tensor([int(bit) for bit in in_bin], dtype = T.long).reshape((n,n))
        reward = log_reward(as_adj.unsqueeze(0), probs)

        directed = nx.DiGraph(as_adj.numpy())
        is_dag = nx.is_directed_acyclic_graph(directed)

        reward *= is_dag
        all_rewards.append(reward)


    all_rewards = T.stack(all_rewards, dim= 0)

    rel_likelihoods = all_rewards / all_rewards.sum(dim= 0)

    return rel_likelihoods