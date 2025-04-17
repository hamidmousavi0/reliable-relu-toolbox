import torch 
import matplotlib.pyplot as plt
import seaborn as sns
list_of_layers=["features.1", "features.4", "features.7", "features.9","features.11","classifier.2","classifier.5"]
def plot_f (keys): 
    fig, axes = plt.subplots(nrows=len(keys),ncols=2, sharex=False, figsize=(4,8))
    fig.tight_layout()
    for i,key in enumerate(keys):
        original_ranger = torch.load("relus/original_"+key+".pt",map_location=torch.device('cpu'))
        ranger_fault = torch.load("relus/faulty"+key+".pt",map_location=torch.device('cpu'))
        d = {'col1':original_ranger.flatten().detach().cpu().numpy() , 'col2':ranger_fault.flatten().detach().cpu().numpy() }
        print(d['col1'].shape)
        sns.kdeplot(d['col1'],fill=True,common_grid=False,multiple="stack",ax=axes[i,0],clip=(0,1),color="red")
        sns.kdeplot(d['col1'],fill=True,common_grid=False,multiple="stack",ax=axes[i,1],clip=(1,None),color="red")
        sns.kdeplot(d['col2'],fill=False,common_grid=False,multiple="stack",ax=axes[i,0],clip=(0,1),color="blue")
        sns.kdeplot(d['col2'],fill=False,common_grid=False,multiple="stack",ax=axes[i,1],clip=(0,None),color="blue")
        plt.savefig("test.png") 

plot_f(list_of_layers)