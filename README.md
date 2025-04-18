<h1 align="center">
  <br/>
    Reliable ReLU Toolbox (RReLU) To Enhance Resilience of DNNs 
  </br>
</h1>
<p align="center">
<a href="#background">Background</a> •
<a href="#usage">Usage</a> •
<a href="#code">Code</a> •
<a href="#citation">Citation</a> •
</p>

## Background
The Reliable ReLU Toolbox (RReLU) is a powerful reliability tool designed to enhance the resiliency of deep neural networks (DNNs) by generating reliable ReLU activation functions.
It is Implemented for the popular PyTorch deep learning platform.
RReLU allows users to find a clipped ReLU activation function using various methods.
This tool is highly versatile for dependability and reliability research, with applications ranging from resiliency analysis of classification networks to training resilient models and improving DNN interpretability.

RReLU includes all state-of-the-art activation restriction methods. These methods offer several advantages: they do not require retraining the entire model, avoid the complexity of fault-aware training, and are non-intrusive, meaning they do not necessitate any changes to an accelerator.
RReLU serves as the research code accompanying the paper (ProAct: Progressive Training for Hybrid Clipped Activation Function to Enhance Resilience of DNNs), and it includes implementations of the following algorithms:

* **ProAct** (the proposed algorithm) ([paper](https://arxiv.org/abs/2406.06313) and ([code](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master/rrelu/search_bound/proact.py)).
* **FitAct** ([paper](https://arxiv.org/pdf/2112.13544) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master/rrelu/search_bound/fitact.py)).
* **FtClipAct** ([paper](https://arxiv.org/pdf/1912.00941) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master/rrelu/search_bound/ftclip.py)).
* **Ranger** ([paper](https://arxiv.org/pdf/2003.13874) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master/rrelu/search_bound/ranger.py)).

### Installing

**From Source**
Download this repository into your project folder.
```
git clone https://github.com/hamidmousavi0/reliable-relu-toolbox.git
```

### Importing

Import the entire package:

```python
import rrelu
```

In order to use multiple gpu : 

```python
pip install horovod
import horovod.torch as hvd
# Initialize Horovod
hvd.init()
# Pin GPU to be used to process local rank (one GPU per process)
if torch.cuda.is_available():
torch.cuda.set_device(hvd.local_rank())
args.num_gpus = hvd.size()
args.rank = hvd.rank()
```

create a model : 

```python
from rrelu.setup import build_model
model = build_model("model_name":string, n_classes:inetger,dropout_rate:float).cuda()
```
support models on CIFAR-10 and CIFAR-100 : **resnet20,resnet32,resnet44,resnet56,vgg11_bn,vgg13_bn,vgg16_bn, vgg19_bn, mobilenetv2_x0_5,mobilenetv2_x0_75,shufflenetv2_x1_5**

All models are in https://github.com/chenyaofo/pytorch-cifar-models/tree/master
 
support_models on ImageNet = **"resnet20","resnet32","resnet44","resnet56","vgg11_bn,vgg13_bn","vgg16_bn",'vgg19_bn', "mobilenetv2_x0_5","mobilenetv2_x0_75","shufflenetv2_x1_5"**

create a data loader : 

```python
from rrelu.setup import build_data_loader
data_loader_dict, n_classes = build_data_loader(
        args.dataset,
        args.image_size,
        args.batch_size,
        args.n_worker,
        args.data_path,
        num_replica=args.num_gpus,
        rank= args.rank
    )
```


replace activatoin function with clipped version and evaluate : 

```python
if args.bitflip == 'fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param is not None:
                    param.copy_(torch.tensor(Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word, n_frac=args.n_frac, n_int=args.n_int).get_val(),dtype=torch.float32,device='cuda').cuda())        
    if args.name_relu_bound!='none':
        model = replace_act(model, args.name_relu_bound, args.name_serach_bound, data_loader_dict, args.bounds_type, args.bitflip,args.pretrained_model,args.dataset,is_root=(hvd.rank() == 0))
        if args.pretrained_model:
            model.load_state_dict(torch.load('pretrained_models/{}/{}/{}_{}_{}_{}.pth'.format(args.dataset,args.model,args.name_relu_bound,args.name_serach_bound,args.bounds_type,args.bitflip),map_location='cuda:0'))
        else:
            torch.save(model.state_dict(), 'pretrained_models/{}/{}/{}_{}_{}_{}.pth'.format(args.dataset,args.model,args.name_relu_bound,args.name_serach_bound,args.bounds_type,args.bitflip))    

    print(f"{args.dataset} {args.model} {args.name_relu_bound} {args.name_serach_bound} {args.bounds_type} {args.bitflip} {args.iterations}{args.pretrained_model}")
    
    if args.pretrained_model or args.name_relu_bound=='none':
        print(f"Model accuracy in {args.bitflip} format after replacing ReLU activation functions: {eval(model, data_loader_dict)}")
        for fault_rate in args.fault_rates:
            val_results_fault = eval_fault(model, data_loader_dict, fault_rate, args.iterations, args.bitflip, args.n_word, args.n_frac, args.n_int)
            print(f"top1 = {val_results_fault['val_top1']}, top5 = {val_results_fault['val_top1']}, Val_loss = {val_results_fault['val_loss']}, fault_rate = {val_results_fault['fault_rate']}")

```


### run search in command line 
When Download this repository into your project folder.
```python
horovodrun -np 1  python search.py --dataset cifar100 --data_path "./dataset/cifar100"  --batch_size 128 --model "shufflenetv2_x1_5" --n_worker 8 \
                     --name_relu_bound "proact"  --name_serach_bound "proact" --bounds_type "layer" --bitflip "fixed" --image_size 32  --pretrained_model
```


## Code

### Structure

The main source code of framework is held in `rrelu`, which carries `search_bounds`, `relu_bounds` , `extended pytorchfi` and other  implementations.


## Citation

View the [published paper](https://arxiv.org/abs/2406.06313). If you use or reference rrelu, please cite:

```
@article{mousavi2024proact,
  title={ProAct: Progressive Training for Hybrid Clipped Activation Function to Enhance Resilience of DNNs},
  author={Mousavi, Seyedhamidreza and Ahmadilivani, Mohammad Hasan and Raik, Jaan and Jenihhin, Maksim and Daneshtalab, Masoud},
  journal={arXiv preprint arXiv:2406.06313},
  year={2024}
}
```
## Acknowledgment

We acknowledge the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no
