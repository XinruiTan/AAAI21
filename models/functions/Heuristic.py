import numpy as np

__all__ = ['heuristic']

def heuristic(entropy, time, latency_c):

    entropy = np.sort(entropy) 
    b_size = entropy.shape[0]
    t_size = b_size - 1

    for p in [0.01 * p for p in range(1, 101, 1)]:
    # for p in [0.0001 * p for p in range(1, 10001, 1)]:

        sum = 0
        pk = np.zeros((b_size,))
        
        for b in range(b_size):
            pk[b] = np.power((1-p), b) * p
            sum += pk[b]

        a = 1/sum
        pk = a * pk
        
        if (pk * time).sum() <= latency_c:
            return get_t(entropy, pk)

def get_t(entropy, pk):

    data_size = entropy.shape[1]
    t_size = entropy.shape[0] - 1
    t = np.zeros((t_size,))

    p = 0

    for b in range(t_size):
        p += pk[b]
        t[b] = entropy[b, np.min([np.int(np.floor(data_size * p)), data_size - 1])]

    return t



        