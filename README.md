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

* **ProAct** (the proposed algorithm) ([paper](https://arxiv.org/abs/2406.06313) and ([code](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master/src/rrelu/search_bound/proact.py)).
* **FitAct** ([paper](https://arxiv.org/pdf/2112.13544) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master/src/rrelu/search_bound/fitact.py)).
* **FtClipAct** ([paper](https://arxiv.org/pdf/1912.00941) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master/src/rrelu/search_bound/ftclip.py)).
* **Ranger** ([paper](https://arxiv.org/pdf/2003.13874) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master/src/rrelu/search_bound/ranger.py)).

## Usage
you can download the rrelu on PyPI [here](https://pypi.org/project/rrelu/).

### Installing

**From Pip**

Install using `pip install rrelu`

**From Source**
Download this repository into your project folder.

### Importing

Import the entire package:

```python
import rrelu
```

In order to use multiple gpu : 

```python
pip install torchpack
from torchpack import distributed as dist
dist.init()
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(dist.local_rank())
```

create a model : 

```python
from rrelu.setup import build_model
model = build_model("model_name":string, n_classes:inetger,dropout_rate:float).cuda()
```

create a data loader : 

```python
from rrelu.setup import build_data_loader
data_loader_dict, n_classes = build_data_loader(
        dataset:string,
        image_size,
        batch_size,
        n_worker,
        data_path,
        dist.size(), # for multiple gpu 
        dist.rank(), # for multiple gpu
    )
```

load pretrained model : 

```python
checkpoint = load_state_dict_from_file(args.init_from)
model.load_state_dict(checkpoint) 
```

change the representatoin to fixed-point : 

```python
for name, param in model.named_parameters():
    if param!=None:
      param.copy_(torch.tensor(Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word,n_frac=args.n_frac,n_int=args.n_int).get_val()))
```
replace activatoin function with clipped version : 

```python
from rrelu.setup import replace_act
model = replace_act(model,args.name_relu_bound,args.name_serach_bound,data_loader_dict,args.bounds_type,args.bitflip)
```

evaluate the clipped model on various fault rates : 

```python
from rrelu.setup import eval_fault
for fault_rate in args.fault_rates: # fault_rates = [10^-7,3 * 10^-7 ,10^-6 , 3 * 10^-6 , 10^-5 , 3 * 10^-5]
            val_results_fault = eval_fault(model,data_loader_dict,fault_rate,args.iterations,args.bitflip,args.n_word , args.n_frac, args.n_int)
```

Import a specific module:

```python
from rrelu.search_bound import proact_bounds 
```

### run search in command line 
When Download this repository into your project folder.
```python
torchpack dist-run -np 1 python search.py --dataset "dataset name (CIFAR10, CIFAR100)" --data_path "path to the dataset" --model "name of the model" --init_from "pretrained file path" \
                      --name_relu_bound "name of bounded relu" --name_serach_bound "name of the search algorithm" --bounds_type "type of thresholds" --bitflip "value representaiton"
```


## Code

### Structure

The main source code of framework is held in `src/rrelu`, which carries `search_bounds`, `relu_bounds` , `extended pytorchfi` and other  implementations.


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
