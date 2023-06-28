# Understanding Catastrophic Forgetting

Cowork repo.

## Overview
### Current Work
I) Can we find a local minimum $\theta$ such that for a given $\gamma > 0$ and $\theta_0$, $\exists \ \theta$ s.t. $\|\theta - \theta_0\| \leq \gamma$ and $L(\theta) \approx 0$?

II) Can we find enforce flat minima constraint on the above $\theta$?

### File Description
- point_exp.py: Code for I and II. The default dataset used is MNIST.

- scmpoint.py: Shell-like wrapper for point_exp.py.

- utils.py: Declaration of complex functions.

Usage Example
Train a model with L1 and L2 coefficients of 1e-5 and 1e-4 centered at $\overrightarrow{\text{10}}$.
```
$ python scmpoint.py --gpu 1 --cpu 8 --L1 0.00001 --L2 0.0001 --center 10
```
