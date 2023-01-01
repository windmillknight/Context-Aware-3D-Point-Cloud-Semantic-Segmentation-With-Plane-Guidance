import torch.nn as nn
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm
                )
            )


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)



class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))



#
# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#         The losses are averaged across observations for each minibatch.
#
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
#                                    putting more focus on hard, misclassi?ed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#
#
#     """
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         # alpha 0.75
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         #print(class_mask)
#
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P*class_mask).sum(1).view(-1,1)
#
#         log_p = probs.log()
#         #print('probs size= {}'.format(probs.size()))
#         #print(probs)
#
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
#         #print('-----bacth_loss------')
#         #print(batch_loss)
#
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        F_loss = -targets * self.alpha * ((1. - inputs) ** self.gamma) * torch.log(inputs + 1e-8) \
        - (1. - targets) * (1. - self.alpha) * (inputs ** self.gamma) * torch.log(1. - inputs + 1e-8)

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class ResnetBlockConv2d(nn.Module):
    ''' 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm2d(size_in)
        self.bn_1 = nn.BatchNorm2d(size_h)

        self.fc_0 = nn.Conv2d(size_in, size_h, 1)
        self.fc_1 = nn.Conv2d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, glob=False):
        super().__init__()
        self.c_dim = c_dim
        self.glob = glob
        self.fc_pos = nn.Conv2d(dim, 2*hidden_dim, 1)
        self.block_0 = ResnetBlockConv2d(2*hidden_dim, size_out=hidden_dim)
        self.block_1 = ResnetBlockConv2d(2*hidden_dim, size_out=hidden_dim)
        self.block_2 = ResnetBlockConv2d(2*hidden_dim, size_out=hidden_dim)
        self.block_3 = ResnetBlockConv2d(2*hidden_dim, size_out=hidden_dim)
        self.block_4 = ResnetBlockConv2d(2*hidden_dim, size_out=hidden_dim)
        self.fc_d = nn.Conv2d(2*hidden_dim, hidden_dim, 1)
        self.fc_c = nn.Conv2d(hidden_dim, c_dim, 1)
        if self.glob:
            self.fc_c_1 = nn.Conv2d(hidden_dim, hidden_dim*2, 1)
            self.fc_c_2 = nn.Conv2d(hidden_dim*2, c_dim, 1)
        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        if self.glob:
            pooled = self.pool(net, dim=2, keepdim=True)
            pooled = self.fc_c_1(self.actvn(pooled))
            c = self.fc_c_2(self.actvn(pooled))
        else:
            pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=1)
            d = self.fc_d(self.actvn(net))
            c = self.fc_c(self.actvn(d))

        return net, c

class ResnetPointnet2(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, glob=False):
        super().__init__()
        self.c_dim = c_dim
        self.glob = glob
        self.fc_pos = nn.Conv2d(dim, 2*hidden_dim, 1)
        self.block_0 = ResnetBlockConv2d(2*hidden_dim, size_out=2*hidden_dim)
        self.block_1 = ResnetBlockConv2d(2*hidden_dim, size_out=2*hidden_dim)
        self.block_2 = ResnetBlockConv2d(2*hidden_dim, size_out=2*hidden_dim)
        self.block_3 = ResnetBlockConv2d(2*hidden_dim, size_out=2*hidden_dim)
        self.block_4 = ResnetBlockConv2d(2*hidden_dim, size_out=hidden_dim)
        self.fc_d = nn.Conv2d(2*hidden_dim, hidden_dim, 1)
        self.fc_c = nn.Conv2d(hidden_dim, c_dim, 1)
        if self.glob:
            self.fc_c_1 = nn.Conv2d(hidden_dim, hidden_dim*2, 1)
            self.fc_c_2 = nn.Conv2d(hidden_dim*2, c_dim, 1)
        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        # pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        # pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        # pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        # pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        if self.glob:
            pooled = self.pool(net, dim=2, keepdim=True)
            pooled = self.fc_c_1(self.actvn(pooled))
            c = self.fc_c_2(self.actvn(pooled))
        else:
            pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=1)
            d = self.fc_d(self.actvn(net))
            c = self.fc_c(self.actvn(d))

        return net, c