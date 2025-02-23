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
    state['closure_T'] = T.eye(n, dtype=T.long).unsqueeze(0).repeat(batch_size, 1, 1)
   
    # Initialize mask
    state['mask'] = T.eye(n, dtype = T.long).unsqueeze(0).repeat(batch_size, 1, 1)

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

    state['closure_T'][~done] |= T.logical_and(source_rows, target_cols).long()
    state['closure_T'][done] = T.eye(n, dtype=T.long)

    state['mask'] = state['adjacency'] + state['closure_T']

    return state