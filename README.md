# adversarial-attacks
Uses [PyTorch](http://pytorch.org/) to generate adversarial images against Google's Inception v3 model. Code based off [Roman Trusov's repo](https://github.com/Lextal/adv-attacks-pytorch-101). Experiments were run in support of the paper [Adversarial Attacks in Machine Learning](https://github.com/cgyulay/adversarial-attacks/blob/master/paper/main.pdf).

## Usage

An installation of PyTorch is assumed. Examples can then be generated by running `adversarial.py`. Input images as well as attack types (targeted vs. non-targeted) can be specified within this file. 

```bash
$ python adversarial.py
```

## Examples

Fast gradient sign attack with epsilon=0.05.
![fgsm](https://github.com/cgyulay/adversarial-attacks/blob/master/paper/adv_05_fgsm.png "fgsm")

Iterative non-targeted attack with epsilon=0.05.
![non-targ](https://github.com/cgyulay/adversarial-attacks/blob/master/paper/adv_05_non.png "non-targeted")

Iterative targeted attack with epsilon=0.02. The attack was successful given the target class "cello."
![targ](https://github.com/cgyulay/adversarial-attacks/blob/master/paper/adv_02_targ.png "targeted")
