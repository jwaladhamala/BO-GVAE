
## Bayesian Optimization on Large Graphs via a Graph Convolutional Generative Model
This repository provides the code and data described in the paper:

**[Bayesian Optimization on Large Graphs via a Graph Convolutional Generative Model: Application in Cardiac Model Personalization]([https://arxiv.org/abs/1907.01406](https://arxiv.org/abs/1907.01406))**
  <a href="http://jwaladhamala.com/" target="_blank">Jwala Dhamala</a>,  Sandesh Ghimire,  John L. Sapp,  B. Milan Horachek,  <a href="[https://pht180.rit.edu/cblwang/](https://pht180.rit.edu/cblwang/)" target="_blank">Linwei Wang</a>,

To be presented at <a href="[https://www.miccai2019.org/](https://www.miccai2019.org/)" target="_blank">MICCAI 2019</a>

Please cite the following if you use the data or the model in your work:
```
@inproceedings{dhamala2019graph,
  title={Bayesian Optimization on Large Graphs via a Graph Convolutional Generative Model: Application in Cardiac Model Personalization},
    author={Jwala Dhamala and Sandesh Ghimire and John L. Sapp and B. Milan Horacek and Linwei Wang},
  booktitle={MICCAI},
  year={2019}
}
```

# Code

### Requirements

The key requirements are listed under `requirements.txt`
In addition, the code  uses  modified versions of <a href="https://github.com/fmfn/BayesianOptimization" target="_blank">bayesopt</a> and   <a href="https://github.com/rusty1s/pytorch_geometric" target="_blank">torch_geometric  </a>. The modified versions of bayesopt and torch_geometric are included with this repo. Go to each folder and install them in the `develop` mode.  

    python setup.py develop

### Running

bogvae is composed of two stages:

- Stage 1: Training a generative VAE model (either fvae or gvae) 
- Stage 2: Utilizing VAE from stage 1, optimize model parameters.

`main.py` provides functionality to either of these stages in isolation or in succession. 

**Configurations:** The configurations for Stage 1 and Stage 2 are provided in the form of a `.json` file. Two examples are included in the folder `config`. 
**Results**: For each run, a folder named `model_name (from .json file)` is created in the experiment directory inside which a copy of the `.json` file, trained model, training logs, diagrams and other results are saved.

#### Example scripts:
To run stage 1 with settings listed in `params_fvae.json`:

    python main.py --config params_fvae --stage 1

To run stage 2 (assuming stage 1 is complete) with settings listed in `params_fvae.json`:

    python main.py --config params_fvae --stage 2

To run stage 1 and stage 2 with settings listed in `params_fvae.json`:

    python main.py --config params_fvae --stage 12

To evaluate pre-trained VAE from stage 1 with settings listed in `params_fvae.json`:

    python main.py --config params_fvae --stage 3


# Data

bogvae requires two types of data: 1) training data of tissue properties to train VAE and 2) patient cases with EKG signals to estimate unknown parameters.  Please download data from the link below and extract them in the `data/` folder. Experiments are saved under `experiments/` folder.

<a href="https://drive.google.com/file/d/1F0p7RzNnTs_3zHPUc5fb0hh0LpxfbnEq/view?usp=sharing" target="_blank">sample data and experiment</a>



# Contact
Please don't hesitate to contact me for any questions or comments. My contact details are found in <a href="http://jwaladhamala.com/" target="_blank">my website</a>. 

