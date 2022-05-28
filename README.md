# ARMOR: Differential Model Distribution for Adversarially Robust Federated Learning

We formalize the notion of differential model robustness (DMR) under the federated learning (FL) context,
and explore how can DMR be realized in concrete FL protocols based on deep neural networks (NNs).
We develop the differential model distribution (DMD) technique,
which distribute different NN models by noise-aided adversarial training.
This is a proof-of-concept implementation of our differential model distribution (DMD) technique.


## Experimental Tracking Platform

To report real-time result to wandb.com, please change wandb ID to your own. \
wandb login {YOUR_WANDB_API_KEY}

## Requirements
* Python 3.6
* Torch 1.10.1
* Numpy 1.19.5

## Experiment Scriptss

To generate a set of FL models of K clients (for example, K=15),
which is the basis of our following experiments.

*  for MNIST, run
``` 
sh run_fed_train.sh 0 mnist 15
``` 

*  for CIFAR-10, run
``` 
sh run_fed_train.sh 0 cifar 50
``` 

To carry out differential adversarial training.

*  for MNIST, run
``` 
sh run_subfed_retrain.sh 0 mnist 15 0.25 600 0.1
``` 

*  for CIFAR-10, run
``` 
sh run_subfed_retrain.sh 0 cifar 50 0.25 300 0.1
``` 

To test differential model robustness.

*  for MNIST, run
``` 
sh run_attack_dmd.sh 0 mnist 15 0.25 600 0.1
``` 

*  for CIFAR-10, run
``` 
sh run_attack_dmd.sh 0 cifar 50 0.25 300 0.1
``` 

To analyze ATR and calculate ASR.

*  for MNIST, run
``` 
sh run_process_exp.sh mnist 15 0.25 600 0.1
``` 

*  for CIFAR-10, run
``` 
sh run_process_exp.sh cifar 50 0.25 300 0.1
``` 

## Reference
[1] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[2] Papernot, Nicolas, et al. "Technical report on the cleverhans v2. 1.0 adversarial examples library." arXiv preprint arXiv:1610.00768 (2016).

[3] Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561

[4] Wei, Kang, et al. "Federated learning with differential privacy: Algorithms and performance analysis." IEEE Transactions on Information Forensics and Security 15 (2020): 3454-3469.
