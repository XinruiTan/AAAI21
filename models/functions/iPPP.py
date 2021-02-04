# import autograd.numpy as np
import numpy as np
# from autograd import grad
from scipy.optimize import fsolve, root_scalar

__all__ = ['ippp', 'ippp_start']
factor = 1.0

def ippp(model, correct, entropy, initial, latency_c, box_bound, args):

    b_size = len(model.avg_b_time)
    t_size = b_size - 1
    data_size = entropy.shape[1]
    entropy = entropy[:t_size]

    grad_g = g_class(model, correct, entropy, latency_c, args)
    # grad_phi = grad(phi, 0)
    box = Box(box_bound, t_size)

    t = np.array(initial)
    lr = (args.eta - args.mu)/(args.eta + args.mu)
    opt_t = None
    opt = np.inf

    for i in range(args.T):

        v = t
        w = t

        for j in range(args.J):
            # print(grad_phi(w, t, correct, entropy, model.avg_b_time, latency_c, args.gamma, args.beta, args.k))
            v_old = v
            # v = box.projection(w - (1 / args.eta) * grad_phi(w, t, correct, entropy, model.avg_b_time, latency_c, args.gamma, args.beta, args.k))
            v = box.projection(w - (1 / args.eta) * grad_g.grad(w, t))
            w = v + lr * (v - v_old)

        t_old = t
        t = v

        if np.square(t - t_old).sum() < opt:
            opt = np.square(t - t_old).sum()
            opt_t = t_old
        
        # if i % (10) == 0:
        #     print(' * iPPP @ {current}/{total}'.format(current=i+1, total=args.T))
        #     print(t)
        #     print(np.array(initial) - t)
    
    # print(t)

    return opt_t


class Box:

    def __init__(self, bound, size):
        self.upper = np.zeros((size,))
        self.lower = np.zeros((size,))
        if bound > 0:
            self.upper = bound * np.ones((size,))
        else:
            self.lower = bound * np.ones((size,))

    def projection(self, t):
        t = np.minimum(t, self.upper)
        t = np.maximum(t, self.lower)
        return t


def phi(threshold, t_t, correct, entropy, avg_b_time, latency_c, gamma, beta, k):

    b_size = len(avg_b_time)
    t_size = b_size - 1
    data_size = entropy.shape[1]

    t = np.matmul(np.array([threshold]).transpose(), np.ones((1, data_size)))

    t = entropy - t

    previous = 1 / (1 + np.exp(-k * t))
    current = 1 / (1 + np.exp(k * t))

    o = 0
    c = -latency_c

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

    return -o + 0.5 * gamma * np.square(threshold - t_t).sum() + 0.5 * beta * np.square(np.maximum(c, 0))


def ippp_start(model, correct, entropy, latency_c, box_bound, args):
    f_c = f_class(model, correct, entropy, latency_c, box_bound, args)
    # root = fsolve(f_c.f, 0.8, maxfev=100000)
    res = root_scalar(f_c.f, bracket=[0, 1], method='brentq')
    root = res.root
    # print(f_c.box.upper - 1.0 * root * f_c.gradient)
    # print(root)
    return f_c.box.projection(f_c.box.upper - 1.0 * root * f_c.gradient)
    
class f_class:

    def __init__(self, model, correct, entropy, latency_c, box_bound, args):
        self.b_size = len(model.avg_b_time)
        self.t_size = self.b_size - 1
        self.data_size = entropy.shape[1]
        self.correct = correct
        self.entropy = entropy[:self.t_size]
        self.avg_b_time = np.array(model.avg_b_time) / latency_c

        self.box = Box(box_bound, self.t_size)
        self.k = args.k 

        # grad_o = grad(self.objective, 0)
        self.gradient = np.ones((self.t_size,))
        # self.gradient = grad_o(self.box.upper)
        # self.gradient = grad_o(self.box.lower)
        # print(self.gradient)

    def objective(self, threshold):
        t = np.matmul(np.array([threshold]).transpose(), np.ones((1, self.data_size)))

        t = self.entropy - t

        previous = 1 / (1 + np.exp(-self.k * t))
        current = 1 / (1 + np.exp(self.k * t))

        o = 0

        for b in range(self.t_size):
            g = current[b]
            for i in range(b):
                g = g * previous[i]
            o += np.mean(g * self.correct[b])

        g = np.ones((self.data_size,))
        for b in range(self.t_size):
            g = g * previous[b]
        o += np.mean(g * self.correct[self.t_size])

        return -o

    def constraint(self, threshold):
        t = np.matmul(np.array([threshold]).transpose(), np.ones((1, self.data_size)))

        t = self.entropy - t

        previous = 1 / (1 + np.exp(-self.k * t))
        current = 1 / (1 + np.exp(self.k * t))

        c = - factor

        for b in range(self.t_size):
            g = current[b]
            for i in range(b):
                g = g * previous[i]
            c += np.mean(g * self.avg_b_time[b])

        g = np.ones((self.data_size,))
        for b in range(self.t_size):
            g = g * previous[b]
        c += np.mean(g * self.avg_b_time[self.t_size])

        return c
    
    def f(self, s):
        return self.constraint(self.box.upper - s * self.gradient)


class g_class:

    def __init__(self, model, correct, entropy, latency_c, args):
        self.b_size = len(model.avg_b_time)
        self.t_size = self.b_size - 1
        self.data_size = entropy.shape[1]
        self.correct = correct
        self.entropy = entropy[:self.t_size]
        self.avg_b_time = model.avg_b_time
        self.latency_c = latency_c

        self.k = args.k 
        self.gamma = args.gamma
        self.beta = args.beta

    def grad(self, threshold, t_t):
        t = np.matmul(np.array([threshold]).transpose(), np.ones((1, self.data_size)))

        t = self.entropy - t

        previous = 1 / (1 + np.exp(-self.k * t))
        current = 1 / (1 + np.exp(self.k * t))

        g_matrix = np.zeros((self.b_size, self.data_size))

        c = - self.latency_c

        for b in range(self.t_size):
            g = current[b]
            for i in range(b):
                g = g * previous[i]
            g_matrix[b, :] = g
            c += np.mean(g * self.avg_b_time[b])
            # o += np.mean(g * self.correct[b])

        g = np.ones((self.data_size,))
        for b in range(self.t_size):
            g = g * previous[b]
        c += np.mean(g * self.avg_b_time[self.t_size])
        g_matrix[-1, :] = g

        g_matrix = self.k * g_matrix

        o_matrix = np.zeros((self.b_size, self.data_size))
        c_matrix = np.zeros((self.b_size, self.data_size))

        for b in range(self.b_size):
            o_matrix[b, :] = g_matrix[b, :] * self.correct[b]
            if c > 0.0:
                c_matrix[b, :] = c * self.beta * g_matrix[b, :] * self.avg_b_time[b]

        grad = np.zeros((self.t_size))

        for b in range(self.t_size):
            for i in range(b+1):
                if i == b:
                    grad[i] -= np.mean(o_matrix[b] * previous[i])
                    if c > 0.0:
                        grad[i] += np.mean(c_matrix[b] * previous[i])
                else:
                    grad[i] += np.mean(o_matrix[b] * current[i])
                    if c > 0.0:
                        grad[i] -= np.mean(c_matrix[b] * current[i])

        for b in range(self.t_size):
            grad[b] += np.mean(o_matrix[self.t_size] * current[b])
            if c > 0.0:
                grad[b] -= np.mean(c_matrix[self.t_size] * current[b])

        grad += self.gamma * (np.array(threshold) - t_t)

        return grad