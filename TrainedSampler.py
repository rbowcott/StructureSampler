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

def sample_graphs(model, n_samples, vars, device):
    n = len(vars)
    nsq = n**2
    
    all_samples = []
    
    for it in tqdm.trange(n_samples):
        # Initialize graph adjacency matrix
        z = T.zeros((1, n, n), dtype=T.long).to(device)
        done = T.full((1,), False, dtype=T.bool).to(device)
        
        # Initialize state
        state = initialise_state(1, n)
        
        while not done.item():
            # Forward pass through the model
            pred = model(T.reshape(z, (1, nsq)).float())
            
            # Create mask for valid actions
            mask = T.cat([T.reshape(state['mask'], (1, nsq)), T.zeros((1, 1), device=device)], 1)
            logits = (pred[...,:nsq+1] - 100000000*mask).log_softmax(1)
            
            # Sample action based on logits
            probs = logits.softmax(1)
            action = probs.multinomial(1)
            
            # Check if terminated
            terminate = (action == nsq).item()
            
            if terminate:
                done[0] = True
            else:
                # Update graph with new edge
                action_idx = action.item()
                i, j = action_idx // n, action_idx % n
                z[0, i, j] = 1
                
                # Update state
                state = update_state(state, z, done, action, n)
        
        # Add completed graph to samples
        all_samples.append(z[0].cpu().numpy())
        
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
n_samples = 25 
samples = sample_graphs(model, n_samples, vars, device)

# Visualize top samples
visualise_top_n(samples, n_graphs = 12, labels = vars, its = n_samples, true_likelihood=tls)