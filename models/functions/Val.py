import numpy as np

__all__ = ['val']

def val(threshold, correct, entropy, avg_b_time):

    b_size = len(avg_b_time)
    t_size = b_size - 1
    data_size = entropy.shape[1]
    entropy = entropy[:t_size]

    t = np.matmul(np.array([threshold]).transpose(), np.ones((1, data_size)))

    t = entropy - t

    previous = np.heaviside(t, 1.0)
    current = np.heaviside(-t, 0.0)

    o, c = 0.0, 0.0

    for b in range(t_size):
        g = current[b]
        for i in range(b):
            g = g * previous[i]
        o += np.mean(g * correct[b])
        c += np.mean(g * avg_b_time[b])

    g = np.ones((data_size,))
    for b in range(t_size):
        g = g * previous[b]
    o += np.mean(g * correct[t_size])
    c += np.mean(g * avg_b_time[t_size])

    return o, c

