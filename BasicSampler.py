import torch as T
import numpy as np
import tqdm
import pickle
from Reward import log_reward
from CycleMask import initialise_state, update_state

'''
-Best way to output results
'''
device = T.device('cpu')

vars = ['storm', 'wind', 'rain', 'wet', 'flooding']
n = len(vars)
nsq = n**2

#Creating neural net & training parameters
n_hid = 256
n_layers = 2

bs = 16
uniform_pb = True
var = 1

def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
    return T.nn.Sequential(*(sum(
        [[T.nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

# Initialising values for training
Z = T.zeros((1,)).to(device)  
model = make_mlp([nsq] + [n_hid] * n_layers + [2*nsq+1]).to(device)
opt = T.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[Z], 'lr':0.1} ])
Z.requires_grad_()

losses = []   #Loss incurred at end of each training run
zs = []   #Learnt Z values
all_visited = []   #Records all sampled termination states

for it in tqdm.trange(5000):
    opt.zero_grad()
   
    z = T.zeros((bs, n, n), dtype=T.long).to(device)   #Adjacency matrices
    done = T.full((bs,), False, dtype=T.bool).to(device)   #Stores whether each terminated
       
    action = None
   
    ll_diff = T.zeros((bs,)).to(device)   #Stores log loss. (Objective is trajectory balance).
    ll_diff += Z

    state = initialise_state(bs, n)
    nd = bs #Number of incomplete graphs

    while T.any(~done):
       
        pred = model(T.reshape(z[~done], (nd, nsq)).float())
       
        mask = T.cat([ T.reshape(state['mask'][~done], (nd, nsq)), T.zeros((nd, 1), device = device)], 1)   #Reshape: [1,2,3,4] <-> [[1,2],[3,4]]
        logits = (pred[...,:nsq+1] - 1000000000*mask).log_softmax(1)  

        init_edge_mask = T.reshape((z[~done]== 0).float(), (nd, nsq) ) #Same for backwards direction
        back_logits = ( (0 if uniform_pb else 1)*pred[...,nsq+1:2*nsq+1] - 1000000000*init_edge_mask).log_softmax(1)

        if action is not None:   #All but first pass
            ll_diff[~done] -= back_logits.gather(1, action[action!=nsq].unsqueeze(1)).squeeze(1)

        #Sampling action for each element in the batch    
        exp_weight= 0.
        temp = 1
        sample_ins_probs = (1-exp_weight)*(logits/temp).softmax(1) + exp_weight*(1-mask) / (1-mask+0.0000001).sum(1).unsqueeze(1)
       
        action = sample_ins_probs.multinomial(1)    #(nd, 1)
        ll_diff[~done] += logits.gather(1, action).squeeze(1)

        terminate = (action==nsq).squeeze(1)    #(nd)

        for x in z[~done][terminate]:
            graph = (T.reshape(x, (nsq,))*(2**T.arange(nsq))).sum().item()   #Converts to index of all_visited list
            all_visited.append(graph)
       
        done[~done] |= terminate
        nd = (~done).sum()

        with T.no_grad():
            #New edges added for each graph
            ne = T.zeros((nd, nsq), dtype = T.long).scatter_add(1, action[~terminate], T.ones(action[~terminate].shape, dtype=T.long, device=device))
            z[~done] += ne.reshape((nd, n, n))

        state = update_state(state, z, done, action[~terminate], n)
        lrw = log_reward(z)
    
    ll_diff -= lrw

    loss = (ll_diff**2).sum()/bs
       
    loss.backward()

    opt.step()

    losses.append(loss.item())
 
    zs.append(Z.item())

    if it%100==0:
        print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())
        emp_dist = np.bincount(all_visited[-50000:], minlength=2**nsq).astype(float)
        emp_dist /= emp_dist.sum()

pickle.dump([losses,zs,all_visited], open(f'out.pkl','wb'))