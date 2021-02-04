import itertools
import numpy as np

__all__ = ['dp_solver']


def  dp_solver(model, correct, entropy, candidates, latency_c):
    b_size = len(model.avg_b_time)
    t_size = b_size - 1
    data_size = entropy.shape[1]

    best_acc = 0
    opt_t = None

    for t in itertools.product(candidates, repeat=t_size):

        exited = np.array([], dtype=np.int32)
        acc = 0.0
        time = 0.0

        for b in range(t_size):
            existing = np.setdiff1d(np.where(entropy[b] < t[b])[0], exited)
            exited = np.union1d(exited, existing)
            acc += np.take(correct[b], existing).sum()
            time += existing.shape[0] * model.avg_b_time[b]

        existing = np.setdiff1d(np.array(range(data_size), dtype=np.int32), exited)
        acc += np.take(correct[t_size], existing).sum()
        time += existing.shape[0] * model.avg_b_time[t_size]

        acc = acc/data_size
        time = time/data_size

        if time <= latency_c and acc > best_acc:
            best_acc = acc
            opt_t = t
    
    return opt_t

