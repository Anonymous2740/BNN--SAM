from mmcv.runner.optimizer.builder import OPTIMIZERS
from pandas import array
from torch.optim import Optimizer
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import torch.optim.sgd as sgd
from mmdet.core.optimizer.PCGrad import PCGrad
if torch.__version__ <= '1.7.0':
    import torch.optim.functional as Fuc
else:
    import torch.optim._functional as Fuc
from tools.analysis_tools.toy import pcgrad
from scipy.optimize import minimize, Bounds, minimize_scalar

from torch.distributions.bernoulli import Bernoulli
from typing import List
from torch import Tensor

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


@OPTIMIZERS.register_module()
class SAM_BNN(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4, c = 0.5, amsgrad=False, reduction='mean'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)

        super(SAM_BNN, self).__init__(params, defaults)
        self._reduction = reduction
        self.c = c

    def _project_conflicting(self, grad_vec, has_grads, shapes=None):
        num_tasks = len(grad_vec)
        # grad_vec: [num_tasks, dim]
        # list to tensor
        grads = torch.stack(grad_vec)/100
        # grad_vec1 = torch.tensor([item.cpu().detach().numpy() for item in grad_vec]).cuda(0)
         # the form of grad_vec = torch.Tensor([[g11,g21],[g12,g22]])
        if self._reduction:
            g0 = grads.mean(0)#calculate mean values for cows
        else:
            g0 = grads.sum(0)#calculate sum values for cows            
        GG = grads.mm(grads.t())
        x_start = np.ones(num_tasks)/num_tasks
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.cpu().numpy()
        b = x_start.copy()
        c = (self.c*g0.norm()).cpu().item()
        def objfn(x):
            return (-x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + \
                    c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww= torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-4)
        g = (g0 - lmbda * gw) / (1 + lmbda)
        return g * 100

    def _set_grad(self, grads):
        '''
            set the modified gradients to the network
        '''
        idx = 0
        # for group in self._optim.param_groups:
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for i,obj in enumerate(objectives):
            if i == 0:
                obj.backward(retain_graph=True)
            else:
            # replacing the last backward(retain_graph=True) to backward() saves much GPU memory
                obj.backward()
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        # for group in self._optim.param_groups:
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                # p = p[1]
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
        
    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    grad = p.grad
                    grads.append(grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            Fuc.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )

        return loss


