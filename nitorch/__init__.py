import torch as _torch  # Necessary for linking extensions

# TODO:
# . check compatible cuda versions between torch and nitorch
#   (see torchvision.extension)

from . import core
from . import nn
from . import spatial
