import torch as T
import torch.nn.functional as F
from LMReward import LMReward

lmreward = LMReward()
pad = lmreward.tokenizer.eos_token

def all_likelihoods(vars):
    n = len(vars)
    starter = f'{pad} Rains cause flooding. Smoking causes cancer. Consider '
    for v in vars[:-1]:
        starter += f'{v}, '
    starter += f' and {vars[-1]}.'
    
    likelihoods = T.zeros((n, n, 2))

    for i in range(n):
        for j in range(n):
            question = f' Does {vars[i]} cause {vars[j]}? '
            likelihoods[i, j] = lmreward.str_loglikelihood(starter + question, [f' Yes', f' No'])

    normalised = F.log_softmax(likelihoods, dim=-1)
    yes = normalised[:, :, 0]
    no = normalised[:, :, 1]

    return (yes, no)

def log_reward(adj, str_logprobs):
    #Given adjacency matrix and dictionary, finds log likelihood of the causal graph
    bs, nodes, _ = adj.shape
    links, not_linked = str_logprobs

    adj_t = T.transpose(adj, 1, 2)
    id = T.eye(nodes, device = adj.device, dtype= T.long).unsqueeze(0).expand(bs, -1, -1)
    not_adj = ~((adj | adj_t) | id)

    reward = (T.mul(adj,links) + 0.5 * T.mul(not_adj, not_linked)).sum((1,2))

    return reward