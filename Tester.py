import torch as T
from Reward import all_likelihoods, log_reward
from GraphVisualiser import visualise_top_n


'''
TODO:

-Evaluate every graph likelihood for Cancer graph.
-Visualise top n.
-Important to GFlownet and compare.
'''

vars = ['Smoking', 'Cancer', 'Drinking', 'Exercise']
n = len(vars)
nsq = len(vars)**2

probs = all_likelihoods(vars)

all_rewards = []
for i in range(2**nsq):
    in_bin = bin(i)[2:].zfill(nsq)
    to_list = log_reward(T.tensor([int(bit) for bit in in_bin], dtype = T.long).reshape((n,n)).unsqueeze(0), probs)
    all_rewards.append(to_list)
print('Calculated all Rewards')

rel_likelihoods = all_rewards / all_rewards.sum(0)

### Model goes here



### Calculate the log reward
### For a given sampled graph, flatten, convert to integer and index by this to get true reward