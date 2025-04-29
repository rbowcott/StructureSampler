import torch as T
import tqdm
from Reward import all_likelihoods
from CycleMask import initialise_state, update_state
from GraphVisualiser import visualise_top_n
from TrueLikelihoods import calculate_true_likelihoods

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

sampler = T.load('basicsampler.pt', map_location=device, weights_only=True)

def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
    return T.nn.Sequential(*(sum(
        [[T.nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

def sample_graphs(model, its, vars, device, bs=16):
    n = len(vars)
    nsq = n**2
    
    all_samples = []
    
    for it in tqdm.trange(its):
        # Initialize graph adjacency matrix
        z = T.zeros((bs, n, n), dtype=T.long).to(device)
        done = T.full((bs,), False, dtype=T.bool).to(device)
        
        # Initialize state
        state = initialise_state(bs, n)
        nd = bs
        
        while T.any(~done):
            pred = model(T.reshape(z[~done], (nd, nsq)).float())
            
            mask = T.cat([ T.reshape(state['mask'][~done], (nd, nsq)), T.zeros((nd, 1), device = device)], 1)
            logits = (pred[...,:nsq+1] - 100000000*mask).log_softmax(1)  
            
            # Sample action based on logits
            probs = logits.softmax(1)
            action = probs.multinomial(1)

            terminate = (action==nsq).squeeze(1)               
            for x in z[~done][terminate]:
                all_samples.append(x)

            done[~done] |= terminate
            nd = (~done).sum()
            
            with T.no_grad():
                #New edges added for each graph
                ne = T.zeros((nd, nsq), dtype = T.long).scatter_add(1, action[~terminate], T.ones(action[~terminate].shape, dtype=T.long, device=device))
                z[~done] += ne.reshape((nd, n, n))

            state = update_state(state, z, done, action[~terminate], n)
        
    return all_samples

# Extract model parameters
n_hid = sampler.get('n_hid')  
n_layers = sampler.get('n_layers')  
vars = sampler.get('vars')
probs = sampler.get('probs')

# Recreate model architecture
n = len(vars)
nsq = n**2
model = make_mlp([nsq] + [n_hid] * n_layers + [2*nsq+1]).to(device)

# Load weights
model.load_state_dict(sampler['model_state_dict'])

# Set model to evaluation mode
model.eval()

#Calculate true likelihoods of graphs under language model
tls = calculate_true_likelihoods(vars, probs)

# Sample graphs
its = 3000
bs = 16
samples = sample_graphs(model, its, vars, device)

# Visualize top samples
visualise_top_n(samples, n_graphs = 3, labels = vars, n_samples= its * bs, true_likelihood=tls)