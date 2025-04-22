from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as T
import torch.nn.functional as F
from LMReward import LMReward
from Reward import all_likelihoods

lmreward = LMReward()
pad = lmreward.tokenizer.eos_token

starter = f'{pad} Which is true: 1. thunder causes lightning, 2. lightning causes thunder, 3. there is no causal relationship.'

str1 = f' 1'
str2 = f' 2'
str3 = f' 3'

ll1 = lmreward.str_loglikelihood(starter, str1)
ll2 = lmreward.str_loglikelihood(starter, str2)
ll3 = lmreward.str_loglikelihood(starter, str3)

min = T.min(T.tensor((ll1, ll2, ll3)))

print(ll1, ll2, ll3, min)

vars = ['thunder', 'lightning']

print(all_likelihoods(vars))


# 1. Rewrite reward to process each set of objects individually, but each set of prompts in parallel - ie way to process final sections without
# recalculating log likelihood each time.
# 2. Subtract min so reward is always positive.

# At some point, will need to process all strings together. Should be easy enough: reshape before and after, keep track of what goes where.
# Then, lets first look to just subtract the max and see how that affects the reward. At the moment, the more negative, the greater the reward. This not correct.
# Need to rework calculation of TrueLikelihoods at the end of this, too.