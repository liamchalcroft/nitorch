""" HyperNetworks (network to generate weights for second network).
Currently bare-bones but can add functionality for e.g. contextual hypernets,
Bayes-by-Hypernets, continual learning.
"""

import torch
from torch import nn as tnn



@nitorchmodule
class HyperNetwork(tnn.Module):

    def __init__(self,
                ):