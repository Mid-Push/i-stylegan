import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable

class BaseFlow(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()

def varify(x):
    return torch.autograd.Variable(torch.from_numpy(x))

def oper(array,oper,axis=-1,keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper

def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    maximum = lambda x: x.max(axis)[0]
    A_max = oper(A,maximum,axis,True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max
    return B

delta = 1e-6
logsigmoid = lambda x: -F.softplus(-x)
log = lambda x: torch.log(x*1e2)-np.log(1e2)
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out

class DenseSigmoidFlow(nn.Module):
    def __init__(self, hidden_dim, in_dim=1, out_dim=1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.act_a = lambda x: F.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: torch.softmax(x, dim=3)
        self.act_u = lambda x: torch.softmax(x, dim=3)

        self.u_ = torch.nn.Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = torch.nn.Parameter(torch.Tensor(out_dim, hidden_dim))
        self.num_params = 3* hidden_dim + in_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x, dsparams):
        delta = 1e-7
        inv = np.log(np.exp(1 - delta) - 1)
        ndim = self.hidden_dim
        pre_u = self.u_[None, None, :, :] + dsparams[:, :, -self.in_dim:][:, :, None, :]
        pre_w = self.w_[None, None, :, :] + dsparams[:, :, 2 * ndim:3 * ndim][:, :, None, :]
        a = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, :, None, :], 3) + b
        sigm = torch.selu(pre_sigm)
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=3)
        #x_ = torch.special.logit(x_pre, eps=1e-5)
        #xnew = x_
        xnew = x_pre
        return xnew


class DDSF(nn.Module):
    def __init__(self, n_blocks=1, hidden_dim=16):
        super().__init__()
        self.num_params = 0
        if n_blocks == 1:
            model = [DenseSigmoidFlow(hidden_dim, in_dim=1, out_dim=1)]
        else:
            model = [DenseSigmoidFlow(hidden_dim=hidden_dim, in_dim=1, out_dim=hidden_dim)]
            for _ in range(n_blocks-2):
                model += [DenseSigmoidFlow(hidden_dim=hidden_dim, in_dim=hidden_dim, out_dim=hidden_dim)]
            model += [DenseSigmoidFlow(hidden_dim=hidden_dim, in_dim=hidden_dim, out_dim=1)]
        self.model = nn.Sequential(*model)
        for block in self.model:
            self.num_params += block.num_params

    def forward(self, x, dsparams):
        x = x.unsqueeze(2)
        start = 0
        for block in self.model:
            block_dsparams = dsparams[:,:,start:start+block.num_params]
            x = block(x, block_dsparams)
            start += block.num_params
        return x.squeeze(2)

def compute_jacobian(inputs, outputs):
    batch_size = outputs.size(0)
    outVector = torch.sum(outputs,0).view(-1)
    outdim = outVector.size()[0]
    jac = torch.stack([torch.autograd.grad(outVector[i], inputs,
                                     retain_graph=True, create_graph=True)[0].view(batch_size, outdim) for i in range(outdim)], dim=1)
    jacs = [jac[i,:,:] for i in range(batch_size)]
    print(jacs[1])

if __name__ == '__main__':

    flow = DDSF(n_blocks=10, hidden_dim=50)
    x = torch.arange(20).view(10, 2)/10.-1.
    x = Variable(x, requires_grad=True)

    dsparams = torch.randn(1, 2, 2*flow.num_params).repeat(10,1,1)
    y = flow(x, dsparams)
    print(x, y)
    compute_jacobian(x, y)

    """
    flow = ConvDenseSigmoidFlow(1,256,1)
    dsparams = torch.randn(1, 2, 1000).repeat(10,1,1)
    x = torch.arange(20).view(10,2,1).repeat(1,1,4).view(10,2,2,2)/10.
    print(x.size(), dsparams.size())
    out = flow(x, dsparams)
    print(x, out.flatten(2), out.size())
    flow = ConvDDSF(n_blocks=3)
    dsparams = torch.randn(1, 2, flow.num_params).repeat(10,1,1)
    x = torch.arange(80).view(10,2,4).view(10,2,2,2)/10
    print(x.size(), dsparams.size())
    out = flow(x, dsparams)
    print(x, out.flatten(2), out.size())
    """

