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

    #Initialize reverse: just while still getting bidirectional edges
    state['reverse_adjacency'] = state['adjacency'].transpose(1,2).clone().detach()
   
    # Initialize transitive closure of transpose
    state['closure_T'] = T.eye(n, dtype=T.long).unsqueeze(0).expand_as(state['adjacency'])
   
    # Initialize mask
    state['mask'] = state['closure_T'].clone().detach()

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

    state['reverse_adjacency'] = state['adjacency'].transpose(1, 2).clone().detach()

    srcs = actions // n 
    targets = actions % n
    source_rows = T.gather(state['closure_T'][~done], 1, srcs.unsqueeze(2).expand(-1, -1, state['closure_T'].size(2)))
    target_cols = T.gather(state['closure_T'][~done], 2, targets.unsqueeze(2).expand(-1, state['closure_T'].size(1), -1))
    state['closure_T'][~done] |= T.logical_and(source_rows, target_cols)
    state['closure_T'][done] = T.eye(n, dtype= T.long)

    #Update mask
    state['mask'] = (state['adjacency'].bool() |
                     state['closure_T'].bool() |
                     state['reverse_adjacency'].bool()).long()

    return state
