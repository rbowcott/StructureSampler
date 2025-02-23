import torch as T
import numpy as np
from LMReward import LMReward

lmreward = LMReward()

def all_likelihoods(vars):
    n = len(vars)
    links = T.zeros((n, n))
    not_linked = T.zeros((n, n))
    for i in range(n):
        for j in range(n):
            relation = f'{vars[i]} causes {vars[j]}'
            unrelated = f'{vars[i]} and {vars[j]} are not causally linked'
            links[i, j] = lmreward.str_loglikelihood(relation)
            not_linked[i,j] = lmreward.str_loglikelihood(unrelated)
    return (links, not_linked)

def log_reward(adj, str_probs):
    #Given adjacency matrix and dictionary, finds log likelihood of that causal graph
    bs, nodes, _ = adj.shape
    links, not_linked = str_probs

    adj_t = T.transpose(adj, 1, 2)
    id = T.eye(nodes, device = adj.device, dtype= T.long).unsqueeze(0).expand(bs, -1, -1)
    not_adj = ~((adj | adj_t) | id)

    reward = (T.mul(adj,links) + 0.5 * T.mul(not_adj, not_linked)).sum((1,2))

    return reward
