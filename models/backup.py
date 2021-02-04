

def start(model, correct, entropy, latency_c, box_bound, args):
    f_c = f_class(model, correct, entropy, latency_c, box_bound, args)
    root = fsolve(f_c.f, 0.0)
    # print(root)
    return f_c.box.projection(f_c.box.upper - 1.0 * root * f_c.gradient)
    
class f_class:

    def __init__(self, model, correct, entropy, latency_c, box_bound, args):
        self.b_size = len(model.avg_b_time)
        self.t_size = self.b_size - 1
        self.data_size = entropy.shape[1]
        self.correct = correct
        self.entropy = entropy[:self.t_size]
        self.latency_c = latency_c
        self.avg_b_time = model.avg_b_time

        self.box = Box(box_bound, self.t_size)
        self.k = args.k 

        grad_o = grad(self.objective, 0)
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

        c = -self.latency_c

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