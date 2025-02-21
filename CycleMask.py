import torch as T

def initialise_state(batch_size, n):
    '''
    Takes:
        (Int) Batch Size
        (Int) Number of Variables

    Returns:
        Dictionary with keys 'adjacency', 'transitive closure of transpose' & 'mask'
    '''
    state = {}
   
    # Initialize adjacency matrices
    state['adjacency'] = T.zeros((batch_size, n, n), dtype=T.long)
   
    # Initialize transitive closure of transpose
    state['closure_T'] = T.eye(n, dtype=T.long).unsqueeze(0).expand_as(state['adjacency'])
   
    # Initialize mask
    state['mask'] = T.eye(n, dtype = T.long).unsqueeze(0).expand_as(state['adjacency'])

    return state

def update_state(state, z, done, actions, n):
    '''
    Takes:
        (Dictionary) State
        (Tensor, Long) Adjacency matrices
        (Tensor, Boolean) Completed trajectories
        (Tensor, Long) Non-terminating edges just added
        (Int) Number of Variables

    Returns:
        Dictionary with same keys as before
    '''
    state['adjacency'][~done] = z[~done]
    state['adjacency'][done] = 0

    srcs = actions // n 
    targets = actions % n

    source_rows = T.gather(state['closure_T'][~done], 1, srcs.unsqueeze(2).expand(-1, -1, state['closure_T'][~done].size(2)))
    target_cols = T.gather(state['closure_T'][~done], 2, targets.unsqueeze(2).expand(-1, state['closure_T'][~done].size(1), -1))

    s = T.logical_and(source_rows, target_cols).long()
    print("s:", s)
    t = state['closure_T'][~done].clone().detach()
    i = T.where(~done)
    # state['closure_T'][~done].copy_(T.logical_or(t, s))
    state['closure_T'][i] = T.logical_or(state['closure_T'][i], s)
    print("state['closure_T'][~done]:", state['closure_T'][~done])
    state['closure_T'][done] = T.eye(n, dtype=T.long)

    state['mask'] = state['adjacency'] + state['closure_T']

    return state