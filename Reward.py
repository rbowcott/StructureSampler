import torch as T
import torch.nn.functional as F
from LMReward import LMReward

lmreward = LMReward()
pad = lmreward.tokenizer.eos_token

def all_likelihoods(vars):
    n = len(vars)
    starter = f'{pad} Rains cause Flooding. Smoking causes Cancer.'
    
    # for v in vars[:-1]:
    #     starter += f'{v}, '
    # starter += f' and {vars[-1]}.'
    
    likelihoods = T.zeros((n, n, 2))

    # Thought this might better capture likelihood than 'Does {i} cause {j}? Y/N'
    for i in range(n):
        for j in range(n):
            question = f' Does {vars[i]} cause {vars[j]}?'
            s1 = ' Yes.'
            s2 = ' No.'
            likelihoods[i, j] = lmreward.str_loglikelihood(starter + question, [s1, s2])

    # normalised = F.log_softmax(likelihoods, dim=-1)
    # yes = normalised[:, :, 0]
    # no = normalised[:, :, 1]

    yes = likelihoods[:, :, 0] / (likelihoods[:, :, 0] + likelihoods[:, :, 1])
    no = likelihoods[:, :, 1] / (likelihoods[:, :, 0] + likelihoods[:, :, 1])

    return (-yes, -no)

def log_reward(adj, str_logprobs):
    #Given adjacency matrix and dictionary, finds log likelihood of the causal graph
    bs, nodes, _ = adj.shape
    links, not_linked = str_logprobs

    adj_t = T.transpose(adj, 1, 2)
    id = T.eye(nodes, device = adj.device, dtype= T.long).unsqueeze(0).expand(bs, -1, -1)
    not_adj = ~((adj | adj_t) | id)

    reward = (T.mul(adj,links) + 0.5 * T.mul(not_adj, not_linked)).sum((1,2))

    return reward