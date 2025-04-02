import torch as T
from LMReward import LMReward

lmreward = LMReward()
pad = lmreward.tokenizer.eos_token

# def all_likelihoods(vars):
#     n = len(vars)
#     links_unnorm = T.zeros((n, n))
#     not_linked_unnorm = T.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             relation = f'{pad} {vars[i]} causes {vars[j]}'
#             unrelated = f'{pad} {vars[i]} and {vars[j]} are not causally linked'
#             links_unnorm[i, j] = lmreward.str_loglikelihood(relation)
#             not_linked_unnorm[i,j] = lmreward.str_loglikelihood(unrelated)
    
#     links = links_unnorm / (links_unnorm + T.t(links_unnorm) + 0.5 * (not_linked_unnorm + T.t(not_linked_unnorm)))
#     not_linked = 0.5 * (not_linked_unnorm + T.t(not_linked_unnorm)) / (links_unnorm + T.t(links_unnorm) + 0.5 * (not_linked_unnorm + T.t(not_linked_unnorm)))
#     return (-links, -not_linked)

def all_likelihoods(vars):
# Have not permuted the initial list here - still to do.
# Normalise just between yes and no for (i, j). Now normalising between (i,j) and (j,i), too.

    n = len(vars)
    starter = f'{pad} Consider {n} objects: '
    for v in vars[:-1]:
        starter += f'{v}, '
    starter += f'{vars[-1]}.'
    
    yes = T.zeros((n,n))
    no = T.zeros((n, n))

    for i in range(n):
        for j in range(n):
            question = starter + f' Does {vars[i]} cause {vars[j]}? '
            yes[i,j] = lmreward.str_loglikelihood(question + 'Yes.')
            no[i,j] = lmreward.str_loglikelihood(question + 'No.')

    yesnorm = yes / (yes + T.t(yes) + no + T.t(no))
    nonorm = no / (yes + T.t(yes) + no + T.t(no))

    return (-yesnorm, -nonorm)

def log_reward(adj, str_logprobs):
    #Given adjacency matrix and dictionary, finds log likelihood of the causal graph
    bs, nodes, _ = adj.shape
    links, not_linked = str_logprobs

    adj_t = T.transpose(adj, 1, 2)
    id = T.eye(nodes, device = adj.device, dtype= T.long).unsqueeze(0).expand(bs, -1, -1)
    not_adj = ~((adj | adj_t) | id)

    reward = (T.mul(adj,links) + 0.5 * T.mul(not_adj, not_linked)).sum((1,2))

    return reward