from typing import List, Optional, Union

import torch
import torch.distributed
from torchpack import distributed
from rrelu.utils.metric import AverageMeter
from rrelu.utils.misc import list_mean, list_sum
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
__all__ = ["ddp_reduce_tensor", "DistributedMetric"]
def ddp_reduce_tensor(
    tensor: torch.Tensor, reduce="mean"
) -> Union[torch.Tensor, List[torch.Tensor]]:
    tensor_list = [torch.empty_like(tensor) for _ in range(distributed.size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list

class DistributedMetric(object):
    """
    Horovod: average metrics from distributed training.
    """

    def __init__(self,name: Optional[str] = None):
        self.name = name
        self.sum = torch.zeros(1)[0]
        self.count = torch.zeros(1)[0]

    def update(self, val, delta_n=1):
        import torch.distributed as dist

        val *= delta_n
        dist.all_reduce(val.clone())
        self.sum += val.detach().cpu()
        self.count += delta_n

    @property
    def avg(self):
        return self.sum / self.count


class DistributedTensor(object):
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.sum = None
        self.count = torch.zeros(1)[0]
        self.synced = False

    def update(self, val, delta_n=1):
        val *= delta_n
        if self.sum is None:
            self.sum = val.detach()
        else:
            self.sum += val.detach()
        self.count += delta_n

    @property
    def avg(self):
        import torch.distributed as dist

        if not self.synced:
            dist.all_reduce(self.sum)
            self.synced = True
        return self.sum / self.count




# class DistributedMetric(object):
#     """Average metrics for distributed training."""

#     def __init__(self, name: Optional[str] = None, backend="ddp"):
#         self.name = name
#         self.sum = 0
#         self.count = 0
#         self.backend = backend

#     def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
#         val *= delta_n
#         if type(val) in [int, float]:
#             val = torch.Tensor(1).fill_(val).cuda()
#         if self.backend == "ddp":
#             self.count += ddp_reduce_tensor(
#                 torch.Tensor(1).fill_(delta_n).cuda(), reduce="sum"
#             )
#             self.sum += ddp_reduce_tensor(val.detach(), reduce="sum")
#         else:
#             raise NotImplementedError

#     @property
#     def avg(self):
#         if self.count == 0:
#             return torch.Tensor(1).fill_(-1)
#         else:
#             return self.sum / self.count



def get_net_device(net):
    return net.parameters().__next__().device
def set_running_statistics(model, data_loader, distributed=False):
    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if distributed:
                bn_mean[name] = DistributedTensor(name + "#mean")
                bn_var[name] = DistributedTensor(name + "#var")
            else:
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = (
                        x.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = (
                        batch_var.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    if len(bn_mean) == 0:
        # skip if there is no batch normalization layers in the network
        return

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(get_net_device(forward_model))
            forward_model(images)

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
