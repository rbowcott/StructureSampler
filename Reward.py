import torch as T
from LMReward import LMReward

lmreward = LMReward()
pad = lmreward.tokenizer.eos_token

def all_likelihoods(vars):

    n = len(vars)
    starter = 'Which is true:'

    ab = T.zeros((n, n))
    ba = T.zeros((n, n))
    no = T.zeros((n, n))

    for i in range(n):
        for j in range(n):
            starter += f'1 {vars[i]} causes {vars[j]}, 2 {vars[j]} causes {vars[i]}, 3 there is no causal link.'
            ab[i,j], ba[i,j], no[i,j] = lmreward.str_loglikelihood(starter, [' 1.', ' 2.', ' 3.'])

    yes = (ab + T.t(ba)) / 2
    no = (no + T.t(no)) / 2

    yes -= T.min(yes)
    no -= T.min(no)

    return yes, no

def log_reward(adj, str_logprobs):
    #Given adjacency matrix and dictionary, finds log likelihood of the causal graph
    bs, nodes, _ = adj.shape
    links, not_linked = str_logprobs

    adj_t = T.transpose(adj, 1, 2)
    id = T.eye(nodes, device = adj.device, dtype= T.long).unsqueeze(0).expand(bs, -1, -1)
    not_adj = ~((adj | adj_t) | id)

    reward = (T.mul(adj,links) + 0.5 * T.mul(not_adj, not_linked)).sum((1,2))

    return reward