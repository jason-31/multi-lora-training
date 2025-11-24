import torch


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamWOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0: # initialize the data if empty
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                grad = p.grad.data

                # update the first moment estimate
                t = state['t'] + 1
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)

                lr_adjusted = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                
                # update parameters in place
                p.data -= lr_adjusted * (m / (v.sqrt() + group['eps']))
                p.data -= lr * weight_decay * p.data

                state['m'] = m
                state['v'] = v
                state['t'] = t
        return loss

    def set_lr(self, lr: float):
        for group in self.param_groups:
            group['lr'] = lr