import torch as T
import numpy as np
import tqdm
from Reward import all_likelihoods, log_reward
from CycleMask import initialise_state, update_state

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

vars = ['bananas', 'lightning', 'thunder']
n = len(vars)
nsq = n**2
probs = all_likelihoods(vars)

#Creating network & training parameters
n_hid = 256
n_layers = 2

bs = 16
uniform_pb = True
var = 1

def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
    return T.nn.Sequential(*(sum(
        [[T.nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

# Initialising training vals
Z = T.zeros((1,)).to(device)  
model = make_mlp([nsq] + [n_hid] * n_layers + [2*nsq+1]).to(device)
opt = T.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[Z], 'lr':0.1} ])
Z.requires_grad_()

losses = []   
zs = []   
its = 1000

for it in tqdm.trange(its):
    opt.zero_grad()
   
    z = T.zeros((bs, n, n), dtype=T.long).to(device)   
    done = T.full((bs,), False, dtype=T.bool).to(device)   
       
    action = None
   
    ll_diff = T.zeros((bs,)).to(device)   
    ll_diff += Z

    state = initialise_state(bs, n)
    nd = bs 

    while T.any(~done):

        pred = model(T.reshape(z[~done], (nd, nsq)).float())
       
        mask = T.cat([ T.reshape(state['mask'][~done], (nd, nsq)), T.zeros((nd, 1), device = device)], 1)
        logits = (pred[...,:nsq+1] - 100000000*mask).log_softmax(1)  

        init_edge_mask = T.reshape((z[~done]== 0).float(), (nd, nsq) ) 
        back_logits = ( (0 if uniform_pb else 1)*pred[...,nsq+1:2*nsq+1] - 100000000*init_edge_mask).log_softmax(1)

        if action is not None:   #All but first pass
            ll_diff[~done] -= back_logits.gather(1, action[action!=nsq].unsqueeze(1)).squeeze(1)

        #Sampling actions
        exp_weight= 0.3
        temp = 1.
        sample_probs = (1-exp_weight)*(logits/temp).softmax(1) + exp_weight*(1-mask) / (1-mask+0.0000001).sum(1).unsqueeze(1)
       
        action = sample_probs.multinomial(1)
        ll_diff[~done] += logits.gather(1, action).squeeze(1)

        terminate = (action==nsq).squeeze(1)
       
        done[~done] |= terminate
        nd = (~done).sum()

        with T.no_grad():
            #New edges added for each graph
            ne = T.zeros((nd, nsq), dtype = T.long).scatter_add(1, action[~terminate], T.ones(action[~terminate].shape, dtype=T.long, device=device))
            z[~done] += ne.reshape((nd, n, n))

        state = update_state(state, z, done, action[~terminate], n)
        lrw = log_reward(z, probs)
    
    ll_diff -= lrw

    loss = (ll_diff**2).sum()/bs
       
    loss.backward()

    opt.step()

    losses.append(loss.item())
 
    zs.append(Z.item())

    if it%100==0:
        print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())

T.save({
    'model_state_dict': model.state_dict(),
    'Z': Z.item(),
    'n_hid': n_hid,
    'n_layers': n_layers,
    'vars': vars,
    'probs' : probs
}, 'basicsampler.pt')