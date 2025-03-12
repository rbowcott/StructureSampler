import torch as T
from Reward import log_reward

def calculate_true_likelihoods(vars, probs):
    n = len(vars)
    nsq = len(vars)**2

    all_rewards = []
    for i in range(2**nsq):
        in_bin = bin(i)[2:].zfill(nsq)
        to_list = log_reward(T.tensor([int(bit) for bit in in_bin], dtype = T.long).reshape((n,n)).unsqueeze(0), probs)
        all_rewards.append(to_list)
    print('Calculated all Rewards')

    rel_likelihoods = all_rewards / sum(all_rewards)

    return rel_likelihoods