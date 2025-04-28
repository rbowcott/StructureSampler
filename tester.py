import torch as T
import torch.nn.functional as F
from LMReward import LMReward

lmreward = LMReward()
pad = lmreward.tokenizer.eos_token

str1 = f'{pad} Rains cause Flooding. Smoking causes Cancer. Does lightning cause thunder?'
str2 = [' Yes.', ' No.']

print(lmreward.str_loglikelihood(str1, str2))

## There is some signal with A causes B, B causes A. 
## Not currently set up for that though. In particular, need a _ does not cause _.
## However now, no signal. 