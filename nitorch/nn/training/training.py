"""Tools to ease model training (like torch.ignite)"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nitorch.core.utils import benchmark, fold, unfold, rubiks_shuffle
from nitorch.core.py import make_tuple
from nitorch.nn.modules import Module
from nitorch.nn import DiceLoss
import string
import math
import os
import random


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    def SummaryWriter():
        raise ImportError('Optional dependency TensorBoard not found')


def split_train_val_test(data, split=[0.6, 0.1, 0.3], shuffle=False, seed=0):
    """Split sequence of data into train, validation and test.

    Parameters
    ----------
    data : [N,] list
        Input data.
    split : [3,] list, default=[0.6, 0.2, 0.2]
        Train, validation, test fractions.
    suffle : bool, default=False
        Randomly shuffle input data (with seed for reproducibility)
    seed : int, default=0
        Seed for random shuffling.

    Returns
    ----------
    train : [split[0]*N,] list
        Train split.
    val : [split[1]*N,] list
        Validation split.
    test : [split[2]*N,] list
        Test split.

    """
    N = len(data)
    # Ensure split is normalised
    split = [s / sum(split) for s in split]
    # Randomly shuffle input data (with seed for reproducibility)
    if shuffle:
        random.seed(seed)
        data = random.sample(data, N)
    # Do train/val/test split
    train, val, test = [], [], []
    for i, d in enumerate(data):
        if i < math.floor(split[0] * N):
            train.append(d)
        elif i < math.floor(sum(split[:2]) * N):
            val.append(d)
        elif i < math.floor(sum(split) * N):
            test.append(d)

    return train, val, test


def update_loss_dict(old, new, weight=1, inplace=True):
    """Update a dictionary of losses/metrics with a new batch

    Parameters
    ----------
    old : dict
        Previous (accumulated) dictionary of losses/metrics
    new : dict
        Dictionary of losses/metrics for the current batch
    weight : float, default=1
        Weight for the batch
    inplace : bool, default=True
        Modify the dictionary in-place

    Returns
    -------
    new : dict
        Updated (accumulated) dictionary of losses/metrics

    """
    if not inplace:
        old = dict(old)
    for key, val in new.items():
        if key in old.keys():
            old[key] += val * weight
        else:
            old[key] = val * weight
    return old


def normalize_loss_dict(losses, weight=1, inplace=True):
    """Normalize all losses in a dict.

    Parameters
    ----------
    losses : dict
        Accumulated dictionary of losses/metrics
    weight : float, default=1
        Sum of weights across all batches
    inplace : bool, default=True
        Modify the dictionary in-place

    Returns
    -------
    losses : dict
        Normalized dictionary of losses/metrics

    """
    if not inplace:
        losses = dict(losses)
    for key, val in losses.items():
        losses[key] /= weight
    return losses


class ModelTrainer:
    """A class that simplifies training a network."""

    _nb_steps = None
    _train_set = None
    _eval_set = None
    _tensorboard = None
    _tensorboard_callbacks = None
    random_state = []

    def __init__(self, model, train_set, eval_set=None,
                 optimizer=None,
                 nb_epoch=100,
                 nb_steps=None,
                 *, # the remaining parameters *must be* keywords
                 device=None,
                 dtype=None,
                 initial_epoch=0,
                 log_interval=10,
                 benchmark=False,
                 seed=None,
                 tensorboard=None,
                 save_model=None,
                 save_optimizer=None,
                 load_model=None,
                 load_optimizer=None,
                 show_losses=True,
                 show_metrics=False,
                 scheduler=ReduceLROnPlateau):
        """

        Parameters
        ----------
        model : Module
            Model to train.
            Its forward pass should accept a `loss` argument, and take as
            inputs the elements that pop out of the training set.
        train_set : sequence[tensor or tuple[tensor]]
            Training set.
            It should be a finite sequence of tensors or tuple of tensors.
        eval_set : sequence[tensor or tuple[tensor]], optional
            Evaluation set.
            It should be a finite sequence of tensors or tuple of tensors.
        optimizer : callable, default=Adam
            A function that takes trainable parameters as inputs and
            returns an Optimizer object.
        nb_epoch : int, default=100
            Number of epochs.
        nb_steps : int, default=`len(train_set) or 100`
            Number of steps per epoch.
            If the training set is a finite sequence (i.e., `len` is
            implemented), its length is used. Else, the training set
            is assumed to be infinite and the default number of steps
            is 100.
        scheduler : Scheduler, default=ReduceLROnPlateau

        Other Parameters
        ----------------
        device : torch.device, optional
            Device to use. By default, use the default cuda device if
            any, else use cpu.
        dtype : torch.dtype, optional
            Data type to use. By default use `torch.get_default_dtype`.
        initial_epoch : int, default=0
            First epoch
        log_interval : int, default=10
            Number of steps between screen updates.
        benchmark : bool, default=False
            Use the cudnn benchmarking utility that uses the first forward
            pass to compare different convolution algorithms and select the
            best performing one. You should only use this option if the
            spatial shape of your input data is constant across mini batches.
        seed : int, optional
            Manual seed to use for training. The seed is set when
            training starts. A context manager is used so that the global
            state is kept untouched. If `None`, use the global state.
        tensorboard : str, optional
            A path to the tensorboard log directory.
            If provided, losses and metrics are registered to the board
            by default.
        save_model : str, optional
            A path to save the model at each epoch. Can have a
            formatted component ('mymodel_{}.pth') for the epoch number.
        save_optimizer : str, optional
            A path to save the optimizer at each epoch. Can have a
            formatted component ('myoptim_{}.pth') for the epoch number.
        load_model : str, optional
            Path to saved weights to use to initialize the model.
        load_optimizer : str, optional
            Path to saved state to use to initialize the optimizer.
        show_losses : bool, default=True
            Print values of individual losses
        show_metrics : bool, default=False
            Print values of individual metrics
        """
        self.model = model
        self.train_set = train_set
        self.eval_set = eval_set
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.benchmark = benchmark
        self.seed = seed
        self.initial_seed = seed
        self.tensorboard = tensorboard
        self._tensorboard_callbacks = dict(train=dict(epoch=[], step=[]),
                                           eval=dict(epoch=[], step=[]))
        self.save_model = save_model
        self.save_optimizer = save_optimizer
        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.show_losses = show_losses
        self.show_metrics = show_metrics
        self.nb_epoch = nb_epoch
        self.nb_steps = nb_steps
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.dtype = dtype or torch.get_default_dtype()
        self.scheduler = scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)

        if self.load_model:
            self.model.load_state_dict(torch.load(self.load_model))
        if self.load_optimizer:
            self.optimizer.load_state_dict(torch.load(self.load_optimizer))

    def _update_nb_steps(self):
        def len_or(x, default):
            return len(x) if hasattr(x, '__len__') else default
        self._nb_train = self._nb_steps or len_or(self._train_set, 100)
        self._nb_eval = self._nb_steps or len_or(self._eval_set, 100)

    class _batch_iterator:
        def __init__(self, set, length):
            self.set = set
            self.length = length
        def __len__(self):
            return self.length
        def __iter__(self):
            d = 0
            while d < self.length:
                for batch in self.set:
                    if d >= self.length:
                        return
                    yield batch
                    d += 1

    @property
    def tensorboard(self):
        if self._tensorboard:
            return self._tensorboard
        else:
            return self._tensorboard

    @tensorboard.setter
    def tensorboard(self, val):
        if not val:
            self._tensorboard = val
        else:
            self._tensorboard = SummaryWriter(val)

    @property
    def nb_steps(self):
        return self._nb_steps

    @nb_steps.setter
    def nb_steps(self, val):
        self._nb_steps = val
        self._update_nb_steps()

    @property
    def train_set(self):
        if self._train_set:
            return self._batch_iterator(self._train_set, self._nb_train)
        else:
            return None

    @train_set.setter
    def train_set(self, val):
        self._train_set = val
        self._update_nb_steps()

    @property
    def eval_set(self):
        if self._eval_set:
            return self._batch_iterator(self._eval_set, self._nb_eval)
        else:
            return None

    @eval_set.setter
    def eval_set(self, val):
        self._eval_set = val
        self._update_nb_steps()

    def _train(self, epoch=0):
        """Train for one epoch"""

        self.model.train()
        epoch_loss = 0.
        epoch_losses = {}
        epoch_metrics = {}
        nb_batches = 0
        nb_steps = len(self.train_set)
        for n_batch, batch in enumerate(self.train_set):
            losses = {}
            metrics = {}
            # forward pass
            batch = make_tuple(batch)
            batch = tuple(torch.as_tensor(b, device=self.device) for b in batch)
            batch = tuple(b.to(dtype=self.dtype)
                          if b.dtype in (torch.half, torch.float, torch.double)
                          else b for b in batch)
            nb_batches += batch[0].shape[0]
            self.optimizer.zero_grad()
            output = self.model(*batch, _loss=losses, _metric=metrics)
            loss = sum(losses.values())
            # backward pass
            loss.backward()
            self.optimizer.step()
            # update average across batches
            with torch.no_grad():
                weight = float(batch[0].shape[0])
                epoch_loss += loss * weight
                update_loss_dict(epoch_losses, losses, weight)
                update_loss_dict(epoch_metrics, metrics, weight)
                # print
                if n_batch % self.log_interval == 0:
                    self._print('train', epoch, n_batch+1, nb_steps,
                                loss, losses, metrics)
                # tb callback
                if self.tensorboard:
                    tbopt = dict(inputs=batch, outputs=output,
                                 epoch=epoch, minibatch=n_batch, mode='train',
                                 loss=loss, losses=losses, metrics=metrics)
                    self.model.board(self.tensorboard, **tbopt)
                    for func in self._tensorboard_callbacks['train']['step']:
                        func(self.tensorboard, **tbopt)
                    del tbopt
        # print summary
        with torch.no_grad():
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('train', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)
            self._board('train', epoch, epoch_loss, epoch_metrics)
            # tb callback
            if self.tensorboard:
                tbopt = dict(epoch=epoch, loss=epoch_loss, mode='train',
                             losses=epoch_losses, metrics=epoch_metrics)
                self.model.board(self.tensorboard, **tbopt)
                for func in self._tensorboard_callbacks['train']['epoch']:
                    func(self.tensorboard, **tbopt)

        return epoch_loss

    def _eval(self, epoch=0):
        """Evaluate once"""
        if self.eval_set is None:
            return

        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_losses = {}
            epoch_metrics = {}
            nb_batches = 0
            nb_steps = len(self.eval_set)
            for n_batch, batch in enumerate(self.eval_set):
                losses = {}
                metrics = {}
                # forward pass
                batch = make_tuple(batch)
                batch = tuple(torch.as_tensor(b, device=self.device) for b in batch)
                batch = tuple(b.to(dtype=self.dtype)
                              if b.dtype in (torch.half, torch.float, torch.double)
                              else b for b in batch)
                nb_batches += batch[0].shape[0]
                self.optimizer.zero_grad()
                output = self.model(*batch, _loss=losses, _metric=metrics)
                loss = sum(losses.values())
                # update average across batches
                weight = float(batch[0].shape[0])
                epoch_loss += loss * weight
                update_loss_dict(epoch_losses, losses, weight)
                update_loss_dict(epoch_metrics, metrics, weight)
                # print
                if n_batch % self.log_interval == 0:
                    self._print('eval', epoch, n_batch + 1, nb_steps,
                                loss, losses, metrics)
                # tb callback
                if self.tensorboard:
                    tbopt = dict(inputs=batch, outputs=output,
                                 epoch=epoch, minibatch=n_batch, mode='eval',
                                 loss=loss, losses=losses, metrics=metrics)
                    self.model.board(self.tensorboard, **tbopt)
                    for func in self._tensorboard_callbacks['eval']['step']:
                        func(self.tensorboard, **tbopt)

            # print summary
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('eval', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)
            self._board('eval', epoch, epoch_loss, epoch_metrics)
            # tb callback
            if self.tensorboard:
                tbopt = dict(epoch=epoch, loss=epoch_loss, mode='eval',
                             losses=epoch_losses, metrics=epoch_metrics)
                self.model.board(self.tensorboard, **tbopt)
                for func in self._tensorboard_callbacks['eval']['epoch']:
                    func(self.tensorboard, **tbopt)

        return epoch_loss

    def _print(self, mode, n_epoch, n_batch, nb_steps, loss,
               losses=None, metrics=None, last=False):
        """Pretty printing

        Parameters
        ----------
        mode : {'train', 'eval'}
        n_epoch : int
            Index of current epoch (starts at one)
        n_batch : int
            Index of current batch (starts at one)
        nb_steps : int
            Total number of batches
        loss : () tensor
            Loss for this batch
        losses : dict[str: () tensor]
            Loss components for this batch
        metrics : dict[str: () tensor]
            Metrics for this batch
        last : bool, default=False
            Is this the end of the batch?
            If True, loss/losses/metrics should contain the average loss
            across all batches.

        """
        name = 'Train' if mode == 'train' else 'Eval '
        if last:
            pct = 1
            bar = '[' + '=' * 10 + ']'
        else:
            pct = n_batch/nb_steps
            len_arrow = min(math.floor(pct*10 + 0.5), 9)
            bar = '[' + '=' * len_arrow + '>' + ' ' * (9-len_arrow) + ']'

        lepoch = str(len(str(self.nb_epoch)))
        evolution = '{:s} | {:' + lepoch + 'd} | {:3.0f}% ' + bar + ' '
        evolution = evolution.format(name, n_epoch, pct*100)

        values = ''
        if mode == 'train':
            values += '| loss = {:12.6g} '.format(loss.item())
            if losses and self.show_losses:
                values += '|'
                for key, val in losses.items():
                    values += ' {}: {:12.6g} '.format(key, val.item())
        if metrics and (mode == 'eval' or self.show_metrics):
            values += '|'
            for key, val in metrics.items():
                values += ' {}: {:12.6g} '.format(key, val.item())

        print(evolution + values, end='\r', flush=True)
        if last:
            print('')

    def _board(self, mode, epoch, loss, epoch_metrics):
        """Add losses and metrics to tensorboard."""
        if not self.tensorboard:
            return
        tb = self.tensorboard
        tb.add_scalars('loss', {mode: loss.item()}, epoch)
        for tag, value in epoch_metrics.items():
            tb.add_scalars(tag, {mode: value.item()}, epoch)
        tb.flush()

    def add_tensorboard_callback(self, func, mode='train', trigger='epoch'):
        """Register tensorboard callbacks

        Parameters
        ----------
        func : callable
            If trigger 'step', with signature
                `(tb, input, output, epoch, step, loss, losses, metrics)`
            If trigger 'epoch', with signature:
                `(tb, epoch, loss, losses, metrics)`
        mode : {'train', 'eval'}
            Trigger either during a training or evaluation call.
        trigger : {'epoch', 'step'}
            Trigger either at the end of a step or at the end of an epoch.

        """
        if mode not in self._tensorboard_callbacks.keys():
            self._tensorboard_callbacks[mode] = dict()
        if trigger not in self._tensorboard_callbacks[mode].keys():
            self._tensorboard_callbacks[mode][trigger] = list()
        self._tensorboard_callbacks[mode][trigger].append(func)

    def _hello(self, mode):
        """Tell the use what we are going to do (mode, device, dtype, ...)

        Parameters
        ----------
        mode : {'train', 'eval'}

        """
        if self.device.type == 'cuda':
            device = torch.cuda.get_device_name(self.device)
        else:
            assert self.device.type == 'cpu'
            device = 'CPU'
        dtype = str(self.dtype).split('.')[-1]
        if mode == 'train':
            hello = 'Training model {} for {} epochs (steps per epoch: {}) ' \
                    'on {} (dtype = {})'
            hello = hello.format(type(self.model).__name__, self.nb_epoch,
                                 len(self.train_set), device, dtype)
        else:
            hello = 'Evaluating model {} (minibatches: {}) on {} (dtype = {})'
            hello = hello.format(type(self.model).__name__,
                                 len(self.eval_set), device, dtype)
        print(hello, flush=True)

    def _save(self, epoch):
        """Save once"""
        if self.save_model:
            save_model = self._formatfile(self.save_model, epoch)
            dir_model = os.path.dirname(save_model)
            if dir_model:
                os.makedirs(dir_model, exist_ok=True)
            torch.save(self.model.state_dict(), save_model)
        if self.save_optimizer:
            save_optimizer = self._formatfile(self.save_optimizer, epoch)
            dir_optimizer = os.path.dirname(save_optimizer)
            if dir_optimizer:
                os.makedirs(dir_optimizer, exist_ok=True)
            torch.save(self.optimizer.state_dict(), save_optimizer)

    @staticmethod
    def _formatfile(file, epoch):
        """Format filename for an epoch"""
        keys = [tup[1] for tup in string.Formatter().parse(file)
                if tup[1] is not None]
        if len(keys) == 1:
            file = file.format(epoch)
        elif len(keys) > 1:
            raise ValueError('Cannot have more than one format key')
        return file

    def train(self):
        """Launch training"""
        self._hello('train')
        with torch.random.fork_rng(enabled=self.seed is not None):
            if self.seed is not None:
                torch.random.manual_seed(self.seed)
            self.initial_seed = torch.random.initial_seed()
            with benchmark(self.benchmark):
                self.model.to(dtype=self.dtype, device=self.device)
                self.epoch = self.initial_epoch
                self._eval(self.epoch)
                self._save(self.epoch)
                for self.epoch in range(self.epoch+1, self.nb_epoch+1):
                    train_loss = self._train(self.epoch)
                    print('Train loss: {}'.format(train_loss))
                    val_loss = self._eval(self.epoch)
                    self._save(self.epoch)
                    # scheduler
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        sched_loss = val_loss or train_loss
                        self.scheduler.step(sched_loss)
                    elif self.scheduler:
                        self.scheduler.step()

    def eval(self):
        """Launch evaluation"""
        self._hello('eval')
        self.model.to(dtype=self.dtype, device=self.device)
        self._eval()

    def init(self):
        """Initialize the random state + run one evaluation."""
        with torch.random.fork_rng(enabled=self.seed is not None):
            if self.seed is not None:
                torch.random.manual_seed(self.seed)
            self.initial_seed = torch.random.initial_seed()
            self.save_random_state()
            self.epoch = self.initial_epoch
            self.model.to(dtype=self.dtype, device=self.device)
            self._eval(self.epoch)
            self._save(self.epoch)

    def set_random_state(self):
        """Populate the random state using a saved state."""
        if self.random_state:
            cpu_state, *gpu_states = self.random_state
            devices = list(range(torch.cuda.device_count()))
            torch.set_rng_state(self.random_state[0])
            for device, state in zip(devices, gpu_states):
                torch.cuda.set_rng_state(state, device)

    def save_random_state(self):
        """Save the current random state."""
        devices = list(range(torch.cuda.device_count()))
        self.random_state = [torch.get_rng_state()]
        self.random_state.extend(torch.cuda.get_rng_state(device)
                                 for device in devices)

    def train1(self):
        """Train for one epoch."""
        with torch.random.fork_rng():
            self.set_random_state()
            self.model.to(dtype=self.dtype, device=self.device)
            self.epoch += 1
            self._train(self.epoch)
            self._eval(self.epoch)
            self._save(self.epoch)
            self.save_random_state()


class PreTrainer:
    """A class that allows self-supervised training via a number of methods.
    Useful for pre-training before using ModelTrainer for supervised tasks."""

    _nb_steps = None
    _train_set = None
    _eval_set = None
    _tensorboard = None
    _tensorboard_callbacks = None
    random_state = []

    def __init__(self, model, train_set, eval_set=None,
                 optimizer=None,
                 nb_epoch=100,
                 nb_steps=None,
                 *, # the remaining parameters *must be* keywords
                 loss=torch.nn.L1Loss(),
                 adv_model=None,
                 lambda_adv=1,
                 lambda_gp=10,
                 device=None,
                 dtype=None,
                 initial_epoch=0,
                 log_interval=10,
                 benchmark=False,
                 seed=None,
                 tensorboard=None,
                 save_model=None,
                 save_optimizer=None,
                 load_model=None,
                 load_optimizer=None,
                 show_losses=True,
                 show_metrics=False,
                 scheduler=ReduceLROnPlateau):
        """

        Parameters
        ----------
        model : Module
            Model to train.
            Its forward pass should accept a `loss` argument, and take as
            inputs the elements that pop out of the training set.
        train_set : sequence[tensor or tuple[tensor]]
            Training set.
            It should be a finite sequence of tensors or tuple of tensors.
        eval_set : sequence[tensor or tuple[tensor]], optional
            Evaluation set.
            It should be a finite sequence of tensors or tuple of tensors.
        optimizer : callable, default=Adam
            A function that takes trainable parameters as inputs and
            returns an Optimizer object.
        nb_epoch : int, default=100
            Number of epochs.
        nb_steps : int, default=`len(train_set) or 100`
            Number of steps per epoch.
            If the training set is a finite sequence (i.e., `len` is
            implemented), its length is used. Else, the training set
            is assumed to be infinite and the default number of steps
            is 100.
        scheduler : Scheduler, default=ReduceLROnPlateau

        Other Parameters
        ----------------
        loss : callable
            Loss to use for pre-training task.
        adv_model : nitorch Module or torch Sequential, default=None
            If not None, will use adversarial loss weighted by lambda_adv
        lambda_adv : int or float, default=1
            If adversarial loss used then total loss will be:
                Loss_total = loss + lambda_adv * (adv_model(y_hat))
        lambda_gp : int or float, default=10
            Gradient penalty for discriminator training, as per Wasserstein GAN
        device : torch.device, optional
            Device to use. By default, use the default cuda device if
            any, else use cpu.
        dtype : torch.dtype, optional
            Data type to use. By default use `torch.get_default_dtype`.
        initial_epoch : int, default=0
            First epoch
        log_interval : int, default=10
            Number of steps between screen updates.
        benchmark : bool, default=False
            Use the cudnn benchmarking utility that uses the first forward
            pass to compare different convolution algorithms and select the
            best performing one. You should only use this option if the
            spatial shape of your input data is constant across mini batches.
        seed : int, optional
            Manual seed to use for training. The seed is set when
            training starts. A context manager is used so that the global
            state is kept untouched. If `None`, use the global state.
        tensorboard : str, optional
            A path to the tensorboard log directory.
            If provided, losses and metrics are registered to the board
            by default.
        save_model : str, optional
            A path to save the model at each epoch. Can have a
            formatted component ('mymodel_{}.pth') for the epoch number.
        save_optimizer : str, optional
            A path to save the optimizer at each epoch. Can have a
            formatted component ('myoptim_{}.pth') for the epoch number.
        load_model : str, optional
            Path to saved weights to use to initialize the model.
        load_optimizer : str, optional
            Path to saved state to use to initialize the optimizer.
        show_losses : bool, default=True
            Print values of individual losses
        show_metrics : bool, default=False
            Print values of individual metrics
        """
        self.model = model
        self.train_set = train_set
        self.eval_set = eval_set
        self.loss = loss
        self.adv_model = adv_model
        if adv_model is not None:
            self.adv_opt = torch.optim.Adam(adv_model.parameters())
        self.lambda_adv = lambda_adv
        self.lambda_gp = lambda_gp
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.benchmark = benchmark
        self.seed = seed
        self.initial_seed = seed
        self.tensorboard = tensorboard
        self._tensorboard_callbacks = dict(train=dict(epoch=[], step=[]),
                                           eval=dict(epoch=[], step=[]))
        self.save_model = save_model
        self.save_optimizer = save_optimizer
        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.show_losses = show_losses
        self.show_metrics = show_metrics
        self.nb_epoch = nb_epoch
        self.nb_steps = nb_steps
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.dtype = dtype or torch.get_default_dtype()
        self.scheduler = scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)

        if self.load_model:
            self.model.load_state_dict(torch.load(self.load_model))
        if self.load_optimizer:
            self.optimizer.load_state_dict(torch.load(self.load_optimizer))

    def _update_nb_steps(self):
        def len_or(x, default):
            return len(x) if hasattr(x, '__len__') else default
        self._nb_train = self._nb_steps or len_or(self._train_set, 100)
        self._nb_eval = self._nb_steps or len_or(self._eval_set, 100)

    def wass_gp(self, disc, real, fake):
        # Adapted from example provided by @eriklindernoren on GitHub

        # debugging - print device of tensors
        device = real.device
        fake = fake.to(device)

        # assume [B, C, **] -> dim = length of shape excluding B & C
        dim = len(real.shape) - 2
        # random number to scale between real & fake
        shape = [real.shape[0], 1]
        _ = [shape.append(1) for i in range(dim)]
        eps = torch.rand(shape)
        eps = eps.to(device)

        mix = (real * eps + fake * (1 - eps)).requires_grad_(True)

        disc_mix = disc(mix)
        if isinstance(disc_mix, (list, tuple)):
            disc_mix = disc_mix[0]

        fake_ = torch.ones(disc_mix.shape, requires_grad=False)
        fake_ = fake_.to(device)

        grad = torch.autograd.grad(
            outputs=disc_mix,
            inputs=mix,
            grad_outputs=fake_,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        grad = grad[0]
        grad = grad.view(grad.shape[0], -1)

        gp = ((grad.norm(2, dim=1) - 1)**2)
        gp = gp.mean()
        return gp

    def rubiks_gen(self, image, kernel=[10,10,10]):
        image = unfold(image, kernel)
        image = rubiks_shuffle(image)
        image = fold(image, len(kernel))
        return image

    class _batch_iterator:
        def __init__(self, set, length):
            self.set = set
            self.length = length
        def __len__(self):
            return self.length
        def __iter__(self):
            d = 0
            while d < self.length:
                for batch in self.set:
                    if d >= self.length:
                        return
                    yield batch
                    d += 1

    @property
    def tensorboard(self):
        if self._tensorboard:
            return self._tensorboard
        else:
            return self._tensorboard

    @tensorboard.setter
    def tensorboard(self, val):
        if not val:
            self._tensorboard = val
        else:
            self._tensorboard = SummaryWriter(val)

    @property
    def nb_steps(self):
        return self._nb_steps

    @nb_steps.setter
    def nb_steps(self, val):
        self._nb_steps = val
        self._update_nb_steps()

    @property
    def train_set(self):
        if self._train_set:
            return self._batch_iterator(self._train_set, self._nb_train)
        else:
            return None

    @train_set.setter
    def train_set(self, val):
        self._train_set = val
        self._update_nb_steps()

    @property
    def eval_set(self):
        if self._eval_set:
            return self._batch_iterator(self._eval_set, self._nb_eval)
        else:
            return None

    @eval_set.setter
    def eval_set(self, val):
        self._eval_set = val
        self._update_nb_steps()

    def _train(self, epoch=0, adv=False, kernel=[10,10,10]):
        """Train for one epoch"""

        self.model.train()
        epoch_loss = 0.
        nb_batches = 0
        nb_steps = len(self.train_set)
        for n_batch, batch in enumerate(self.train_set):
            # forward pass
            batch = make_tuple(batch)
            batch = tuple(torch.as_tensor(b, device=self.device) for b in batch)
            batch = tuple(b.to(dtype=self.dtype)
                          if b.dtype in (torch.half, torch.float, torch.double)
                          else b for b in batch)
            nb_batches += batch[0].shape[0]
            target = batch[0]
            if len(batch) > 1:
                meta = batch[-1]
            else:
                meta = None
            self.optimizer.zero_grad()

            image = self.rubiks_gen(target, kernel)
            output = self.model(image, meta=meta, seg=False, gan=True, gan_meta=meta)

            if adv == True:
                self.adv_opt.zero_grad()
                real_true = self.adv_model(target)
                real_false = self.adv_model(output)
                grad_pen = self.wass_gp(self.adv_model, target, output)
                loss_adv_d = -torch.mean(real_true) + torch.mean(real_false) + self.lambda_gp * grad_pen
                loss_adv_d.backward()
                self.adv_opt.step()
                loss = self.loss(output, target) - torch.mean(real_false)
            else:
                loss = self.loss(output, target)
            # backward pass
            loss.backward()
            self.optimizer.step()
            # update average across batches
            with torch.no_grad():
                weight = float(batch[0].shape[0])
                epoch_loss += loss * weight
                # # print
                # if n_batch % self.log_interval == 0:
                #     self._print('train', epoch, n_batch+1, nb_steps,
                #                 loss)
                # # tb callback
                # if self.tensorboard:
                #     tbopt = dict(inputs=batch, outputs=output,
                #                  epoch=epoch, minibatch=n_batch, mode='train',
                #                  loss=loss)
                #     self.model.board(self.tensorboard, **tbopt)
                #     for func in self._tensorboard_callbacks['train']['step']:
                #         func(self.tensorboard, **tbopt)
                #     del tbopt
        # print summary
        with torch.no_grad():
            epoch_loss /= nb_batches
            # self._print('train', epoch, nb_steps, nb_steps,
            #             epoch_loss, last=True)
            # self._board('train', epoch)
            # # tb callback
            # if self.tensorboard:
            #     tbopt = dict(epoch=epoch, loss=epoch_loss, mode='train')
            #     self.model.board(self.tensorboard, **tbopt)
            #     for func in self._tensorboard_callbacks['train']['epoch']:
            #         func(self.tensorboard, **tbopt)

        return epoch_loss

    def _eval(self, epoch=0):
        """Evaluate once"""
        if self.eval_set is None:
            return

        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_losses = {}
            epoch_metrics = {}
            nb_batches = 0
            nb_steps = len(self.eval_set)
            for n_batch, batch in enumerate(self.eval_set):
                losses = {}
                metrics = {}
                # forward pass
                batch = make_tuple(batch)
                batch = tuple(torch.as_tensor(b, device=self.device) for b in batch)
                batch = tuple(b.to(dtype=self.dtype)
                              if b.dtype in (torch.half, torch.float, torch.double)
                              else b for b in batch)
                nb_batches += batch[0].shape[0]
                self.optimizer.zero_grad()
                output = self.model(*batch, _loss=losses, _metric=metrics)
                loss = sum(losses.values())
                # update average across batches
                weight = float(batch[0].shape[0])
                epoch_loss += loss * weight
                update_loss_dict(epoch_losses, losses, weight)
                update_loss_dict(epoch_metrics, metrics, weight)
                # print
                if n_batch % self.log_interval == 0:
                    self._print('eval', epoch, n_batch + 1, nb_steps,
                                loss, losses, metrics)
                # tb callback
                if self.tensorboard:
                    tbopt = dict(inputs=batch, outputs=output,
                                 epoch=epoch, minibatch=n_batch, mode='eval',
                                 loss=loss, losses=losses, metrics=metrics)
                    self.model.board(self.tensorboard, **tbopt)
                    for func in self._tensorboard_callbacks['eval']['step']:
                        func(self.tensorboard, **tbopt)

            # print summary
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('eval', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)
            self._board('eval', epoch, epoch_loss, epoch_metrics)
            # tb callback
            if self.tensorboard:
                tbopt = dict(epoch=epoch, loss=epoch_loss, mode='eval',
                             losses=epoch_losses, metrics=epoch_metrics)
                self.model.board(self.tensorboard, **tbopt)
                for func in self._tensorboard_callbacks['eval']['epoch']:
                    func(self.tensorboard, **tbopt)

        return epoch_loss

    def _print(self, mode, n_epoch, n_batch, nb_steps, loss,
               losses=None, metrics=None, last=False):
        """Pretty printing

        Parameters
        ----------
        mode : {'train', 'eval'}
        n_epoch : int
            Index of current epoch (starts at one)
        n_batch : int
            Index of current batch (starts at one)
        nb_steps : int
            Total number of batches
        loss : () tensor
            Loss for this batch
        losses : dict[str: () tensor]
            Loss components for this batch
        metrics : dict[str: () tensor]
            Metrics for this batch
        last : bool, default=False
            Is this the end of the batch?
            If True, loss/losses/metrics should contain the average loss
            across all batches.

        """
        name = 'Train' if mode == 'train' else 'Eval '
        if last:
            pct = 1
            bar = '[' + '=' * 10 + ']'
        else:
            pct = n_batch/nb_steps
            len_arrow = min(math.floor(pct*10 + 0.5), 9)
            bar = '[' + '=' * len_arrow + '>' + ' ' * (9-len_arrow) + ']'

        lepoch = str(len(str(self.nb_epoch)))
        evolution = '{:s} | {:' + lepoch + 'd} | {:3.0f}% ' + bar + ' '
        evolution = evolution.format(name, n_epoch, pct*100)

        values = ''
        if mode == 'train':
            values += '| loss = {:12.6g} '.format(loss.item())
            if losses and self.show_losses:
                values += '|'
                for key, val in losses.items():
                    values += ' {}: {:12.6g} '.format(key, val.item())
        if metrics and (mode == 'eval' or self.show_metrics):
            values += '|'
            for key, val in metrics.items():
                values += ' {}: {:12.6g} '.format(key, val.item())

        print(evolution + values, end='\r', flush=True)
        if last:
            print('')

    def _board(self, mode, epoch, loss, epoch_metrics):
        """Add losses and metrics to tensorboard."""
        if not self.tensorboard:
            return
        tb = self.tensorboard
        tb.add_scalars('loss', {mode: loss.item()}, epoch)
        for tag, value in epoch_metrics.items():
            tb.add_scalars(tag, {mode: value.item()}, epoch)
        tb.flush()

    def add_tensorboard_callback(self, func, mode='train', trigger='epoch'):
        """Register tensorboard callbacks

        Parameters
        ----------
        func : callable
            If trigger 'step', with signature
                `(tb, input, output, epoch, step, loss, losses, metrics)`
            If trigger 'epoch', with signature:
                `(tb, epoch, loss, losses, metrics)`
        mode : {'train', 'eval'}
            Trigger either during a training or evaluation call.
        trigger : {'epoch', 'step'}
            Trigger either at the end of a step or at the end of an epoch.

        """
        if mode not in self._tensorboard_callbacks.keys():
            self._tensorboard_callbacks[mode] = dict()
        if trigger not in self._tensorboard_callbacks[mode].keys():
            self._tensorboard_callbacks[mode][trigger] = list()
        self._tensorboard_callbacks[mode][trigger].append(func)

    def _hello(self, mode):
        """Tell the use what we are going to do (mode, device, dtype, ...)

        Parameters
        ----------
        mode : {'train', 'eval'}

        """
        if self.device.type == 'cuda':
            device = torch.cuda.get_device_name(self.device)
        else:
            assert self.device.type == 'cpu'
            device = 'CPU'
        dtype = str(self.dtype).split('.')[-1]
        if mode == 'train':
            hello = 'Training model {} for {} epochs (steps per epoch: {}) ' \
                    'on {} (dtype = {})'
            hello = hello.format(type(self.model).__name__, self.nb_epoch,
                                 len(self.train_set), device, dtype)
        else:
            hello = 'Evaluating model {} (minibatches: {}) on {} (dtype = {})'
            hello = hello.format(type(self.model).__name__,
                                 len(self.eval_set), device, dtype)
        print(hello, flush=True)

    def _save(self, epoch):
        """Save once"""
        if self.save_model:
            save_model = self._formatfile(self.save_model, epoch)
            dir_model = os.path.dirname(save_model)
            if dir_model:
                os.makedirs(dir_model, exist_ok=True)
            torch.save(self.model.state_dict(), save_model)
        if self.save_optimizer:
            save_optimizer = self._formatfile(self.save_optimizer, epoch)
            dir_optimizer = os.path.dirname(save_optimizer)
            if dir_optimizer:
                os.makedirs(dir_optimizer, exist_ok=True)
            torch.save(self.optimizer.state_dict(), save_optimizer)

    @staticmethod
    def _formatfile(file, epoch):
        """Format filename for an epoch"""
        keys = [tup[1] for tup in string.Formatter().parse(file)
                if tup[1] is not None]
        if len(keys) == 1:
            file = file.format(epoch)
        elif len(keys) > 1:
            raise ValueError('Cannot have more than one format key')
        return file

    def train(self):
        """Launch training"""
        self._hello('train')
        with torch.random.fork_rng(enabled=self.seed is not None):
            if self.seed is not None:
                torch.random.manual_seed(self.seed)
            self.initial_seed = torch.random.initial_seed()
            with benchmark(self.benchmark):
                self.model.to(dtype=self.dtype, device=self.device)
                self.epoch = self.initial_epoch
                self._eval(self.epoch)
                self._save(self.epoch)
                for self.epoch in range(self.epoch+1, self.nb_epoch+1):
                    train_loss = self._train(self.epoch)
                    print('Train loss: {}'.format(train_loss))
                    val_loss = self._eval(self.epoch)
                    self._save(self.epoch)
                    # scheduler
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        sched_loss = val_loss or train_loss
                        self.scheduler.step(sched_loss)
                    elif self.scheduler:
                        self.scheduler.step()

    def eval(self):
        """Launch evaluation"""
        self._hello('eval')
        self.model.to(dtype=self.dtype, device=self.device)
        self._eval()

    def init(self):
        """Initialize the random state + run one evaluation."""
        with torch.random.fork_rng(enabled=self.seed is not None):
            if self.seed is not None:
                torch.random.manual_seed(self.seed)
            self.initial_seed = torch.random.initial_seed()
            self.save_random_state()
            self.epoch = self.initial_epoch
            self.model.to(dtype=self.dtype, device=self.device)
            self._eval(self.epoch)
            self._save(self.epoch)

    def set_random_state(self):
        """Populate the random state using a saved state."""
        if self.random_state:
            cpu_state, *gpu_states = self.random_state
            devices = list(range(torch.cuda.device_count()))
            torch.set_rng_state(self.random_state[0])
            for device, state in zip(devices, gpu_states):
                torch.cuda.set_rng_state(state, device)

    def save_random_state(self):
        """Save the current random state."""
        devices = list(range(torch.cuda.device_count()))
        self.random_state = [torch.get_rng_state()]
        self.random_state.extend(torch.cuda.get_rng_state(device)
                                 for device in devices)

    def train1(self):
        """Train for one epoch."""
        with torch.random.fork_rng():
            self.set_random_state()
            self.model.to(dtype=self.dtype, device=self.device)
            self.epoch += 1
            self._train(self.epoch)
            self._eval(self.epoch)
            self._save(self.epoch)
            self.save_random_state()


class SegGANTrainer:
    """Training class for Seg+CycleGAN model, may need tweaking for general use."""

    _nb_steps = None
    _train_set = None
    _eval_set = None
    _tensorboard = None
    _tensorboard_callbacks = None
    random_state = []

    def __init__(self, model, disc, train_set, eval_set=None,
                 optimizer=None,
                 lambda_gp=10,
                 lambda_domain=1,
                 lambda_cycle=10,
                 lambda_id=0.1,
                 lambda_seg_domain=0.1,
                 lambda_seg_synth=0.3,
                 lambda_seg_adv=0.1,
                 seg_loss=DiceLoss(log=False, implicit=False),
                 domain_loss=torch.nn.CrossEntropyLoss(),
                 cycle_loss=torch.nn.L1Loss(),
                 gen_interval=5,
                 seg_interval=20,
                 adv_seg_start=5,
                 nb_epoch=100,
                 nb_steps=None,
                 *, # the remaining parameters *must be* keywords
                 device=None,
                 dtype=None,
                 initial_epoch=0,
                 log_interval=10,
                 benchmark=False,
                 seed=None,
                 tensorboard=None,
                 save_model=None,
                 save_optimizer=None,
                 load_model=None,
                 load_optimizer=None,
                 show_losses=True,
                 show_metrics=False,
                 scheduler=ReduceLROnPlateau):
        """

        Parameters
        ----------
        model : Module
            (Generative) Model to train.
            Its forward pass should accept a `loss` argument, and take as
            inputs the elements that pop out of the training set.
        disc : Module or sequence[Module]
            Discriminator model(s) for GAN training.
            For cycleSeg model this should contain one for GAN and one for seg.
        train_set : sequence[tensor or tuple[tensor]]
            Training set.
            It should be a finite sequence of tensors or tuple of tensors.
            Should contain tuple of (Source, Target) domains.
        eval_set : sequence[tensor or tuple[tensor]], optional
            Evaluation set.
            It should be a finite sequence of tensors or tuple of tensors.
            Should contain tuple of (Source, Target) domains.
        optimizer : callable, default=Adam
            A function that takes trainable parameters as inputs and
            returns an Optimizer object.
        nb_epoch : int, default=100
            Number of epochs.
        nb_steps : int, default=`len(train_set) or 100`
            Number of steps per epoch.
            If the training set is a finite sequence (i.e., `len` is
            implemented), its length is used. Else, the training set
            is assumed to be infinite and the default number of steps
            is 100.
        scheduler : Scheduler, default=ReduceLROnPlateau

        Other Parameters
        ----------------
        device : torch.device, optional
            Device to use. By default, use the default cuda device if
            any, else use cpu.
        dtype : torch.dtype, optional
            Data type to use. By default use `torch.get_default_dtype`.
        initial_epoch : int, default=0
            First epoch
        log_interval : int, default=10
            Number of steps between screen updates.
        benchmark : bool, default=False
            Use the cudnn benchmarking utility that uses the first forward
            pass to compare different convolution algorithms and select the
            best performing one. You should only use this option if the
            spatial shape of your input data is constant across mini batches.
        seed : int, optional
            Manual seed to use for training. The seed is set when
            training starts. A context manager is used so that the global
            state is kept untouched. If `None`, use the global state.
        tensorboard : str, optional
            A path to the tensorboard log directory.
            If provided, losses and metrics are registered to the board
            by default.
        save_model : str, optional
            A path to save the model at each epoch. Can have a
            formatted component ('mymodel_{}.pth') for the epoch number.
        save_optimizer : str, optional
            A path to save the optimizer at each epoch. Can have a
            formatted component ('myoptim_{}.pth') for the epoch number.
        load_model : str, optional
            Path to saved weights to use to initialize the model.
        load_optimizer : str, optional
            Path to saved state to use to initialize the optimizer.
        show_losses : bool, default=True
            Print values of individual losses
        show_metrics : bool, default=False
            Print values of individual metrics
        """
        self.model = model
        if len(disc) == 2:
            self.disc_gan, self.disc_seg = disc
            self.disc = None
        else:
            self.disc = disc

        self.train_set = train_set
        self.eval_set = eval_set
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5,0.999))
            if self.disc:
                self.optim_d = torch.optim.Adam(self.disc.parameters(), lr=0.0002, betas=(0.5,0.999))
            elif self.disc_gan and self.disc_seg:
                self.optim_d = None
                self.optim_d_gan = torch.optim.Adam(self.disc_gan.parameters(), lr=0.0002, betas=(0.5,0.999))
                self.optim_d_seg = torch.optim.Adam(self.disc_seg.parameters(), lr=0.0002, betas=(0.5,0.999))
        self.optimizer = optimizer
        self.lambda_gp = lambda_gp
        self.lambda_domain = lambda_domain
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.lambda_seg_domain = lambda_seg_domain
        self.lambda_seg_synth = lambda_seg_synth
        self.lambda_seg_adv = lambda_seg_adv
        self.seg_loss = seg_loss
        self.domain_loss = domain_loss
        self.cycle_loss = cycle_loss
        self.gen_interval = gen_interval
        self.seg_interval = seg_interval
        self.adv_seg_start = adv_seg_start
        self.log_interval = log_interval
        self.benchmark = benchmark
        self.seed = seed
        self.initial_seed = seed
        self.tensorboard = tensorboard
        self._tensorboard_callbacks = dict(train=dict(epoch=[], step=[]),
                                           eval=dict(epoch=[], step=[]))
        self.save_model = save_model
        self.save_optimizer = save_optimizer
        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.show_losses = show_losses
        self.show_metrics = show_metrics
        self.nb_epoch = nb_epoch
        self.nb_steps = nb_steps
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.dtype = dtype or torch.get_default_dtype()
        self.scheduler = scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)

        if self.load_model:
            self.model.load_state_dict(torch.load(self.load_model))
        if self.load_optimizer:
            self.optimizer.load_state_dict(torch.load(self.load_optimizer))

    def _update_nb_steps(self):
        def len_or(x, default):
            return len(x) if hasattr(x, '__len__') else default
        self._nb_train = self._nb_steps or len_or(self._train_set, 100)
        self._nb_eval = self._nb_steps or len_or(self._eval_set, 100)

    class _batch_iterator:
        def __init__(self, set, length):
            self.set = set
            self.length = length
        def __len__(self):
            return self.length
        def __iter__(self):
            d = 0
            while d < self.length:
                for batch in self.set:
                    if d >= self.length:
                        return
                    yield batch
                    d += 1

    @property
    def tensorboard(self):
        if self._tensorboard:
            return self._tensorboard
        else:
            return self._tensorboard

    @tensorboard.setter
    def tensorboard(self, val):
        if not val:
            self._tensorboard = val
        else:
            self._tensorboard = SummaryWriter(val)

    @property
    def nb_steps(self):
        return self._nb_steps

    @nb_steps.setter
    def nb_steps(self, val):
        self._nb_steps = val
        self._update_nb_steps()

    @property
    def train_set(self):
        if self._train_set:
            return self._batch_iterator(self._train_set, self._nb_train)
        else:
            return None

    @train_set.setter
    def train_set(self, val):
        self._train_set = val
        self._update_nb_steps()

    @property
    def eval_set(self):
        if self._eval_set:
            return self._batch_iterator(self._eval_set, self._nb_eval)
        else:
            return None

    @eval_set.setter
    def eval_set(self, val):
        self._eval_set = val
        self._update_nb_steps()

    def wass_gp(self, disc, real, fake):
        # Adapted from example provided by @eriklindernoren on GitHub

        # debugging - print device of tensors
        device = real.device
        fake = fake.to(device)

        # assume [B, C, **] -> dim = length of shape excluding B & C
        dim = len(real.shape) - 2
        # random number to scale between real & fake
        shape = [real.shape[0], 1]
        _ = [shape.append(1) for i in range(dim)]
        eps = torch.rand(shape)
        eps = eps.to(device)

        mix = (real * eps + fake * (1 - eps)).requires_grad_(True)

        disc_mix = disc(mix)
        if isinstance(disc_mix, (list, tuple)):
            disc_mix = disc_mix[0]

        fake_ = torch.ones(disc_mix.shape, requires_grad=False)
        fake_ = fake_.to(device)

        grad = torch.autograd.grad(
            outputs=disc_mix,
            inputs=mix,
            grad_outputs=fake_,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        grad = grad[0]
        grad = grad.view(grad.shape[0], -1)

        gp = ((grad.norm(2, dim=1) - 1)**2)
        gp = gp.mean()
        return gp

    def _train_gan(self, epoch=0):
        """Train GAN for one epoch"""
        # TODO: Look at implementing FID metric for val

        self.model.train()
        # check for translation data - need to extend to work for standard generation
        if len(self.train_set) == 2:
            train_s, train_t = self.train_set
            train_set = zip(train_s, train_t)
            nb_steps = len(train_s)
        else:
            train_set = self.train_set
            nb_steps = len(train_set)
        epoch_loss_d_gan = 0.
        epoch_loss_d_seg = 0.
        epoch_loss_g = 0.
        epoch_loss_seg = 0.
        epoch_losses = {}
        epoch_metrics = {}
        nb_batches = 0
        ### TODO: add proper data-logging
        for n_batch, batch in enumerate(train_set):
            nb_d_gan = 0.
            nb_d_seg = 0.
            nb_gan = 0.
            nb_seg = 0.
            losses = {}
            metrics = {}
            loss_d_gan = 0.
            loss_d_seg = 0.
            loss_g = 0.
            loss_seg = 0.
            batch_s, batch_t = batch

            # create batch for source domain
            batch_s = make_tuple(batch_s)
            batch_s = tuple(torch.as_tensor(b, device=self.device) for b in batch_s)
            batch_s = tuple(b.to(dtype=self.dtype)
                          if b.dtype in (torch.half, torch.float, torch.double)
                          else b for b in batch_s)
            batch_s_img, batch_s_ref, batch_s_met = batch_s
            nb_batches += batch_s[0].shape[0]

            # create batch for target domain
            batch_t = make_tuple(batch_t)
            batch_t = tuple(torch.as_tensor(b, device=self.device) for b in batch_t)
            batch_t = tuple(b.to(dtype=self.dtype)
                          if b.dtype in (torch.half, torch.float, torch.double)
                          else b for b in batch_t)
            batch_t_img, batch_t_met = batch_t

            self.optimizer.zero_grad()
            if self.optim_d:
                self.optim_d.zero_grad()
            if self.optim_d_gan:
                self.optim_d_gan.zero_grad()
            if self.optim_d_seg:
                self.optim_d_seg.zero_grad()

            ## training translation discriminator

            # first perform source -> target
            trans_t_img = self.model(image=batch_s_img, meta=batch_s_met,
                                    seg=False, gan=True,
                                    gan_meta=batch_t_met)
            
            # test discriminator on translation
            real_valid, real_class = self.disc_gan(batch_s_img)
            fake_valid, fake_class = self.disc_gan(trans_t_img)

            # calculate wasserstein gradient penalty
            grad_pen = self.wass_gp(self.disc_gan, batch_s_img, trans_t_img)

            # adversarial
            loss_adv_d = -torch.mean(real_valid) + torch.mean(fake_valid) + self.lambda_gp * grad_pen

            # domain
            loss_dom_d = self.domain_loss(real_class.view(-1, real_class.shape[-1]), torch.max(batch_s_met, -1)[1].view(-1))

            # repeat for target -> source
            trans_s_img = self.model(image=batch_t_img, meta=batch_t_met,
                                    seg=False, gan=True,
                                    gan_meta=batch_s_met)
            real_valid, real_class = self.disc_gan(batch_t_img)
            fake_valid, _ = self.disc_gan(trans_s_img)
            grad_pen = self.wass_gp(self.disc_gan, batch_s_img, trans_t_img)
            loss_adv_d += -torch.mean(real_valid) + torch.mean(fake_valid) + self.lambda_gp * grad_pen
            loss_dom_d += self.domain_loss(real_class.view(-1, real_class.shape[-1]), torch.max(batch_t_met, -1)[1].view(-1))

            # calculate overall loss
            loss_d_gan = loss_adv_d + self.lambda_domain * loss_dom_d

            losses['loss_adv_d_gan'] = loss_adv_d
            losses['loss_dom_d_gan'] = loss_dom_d
            losses['loss_d_gan'] = loss_d_gan

            loss_d_gan.backward()
            self.optim_d_gan.step()

            nb_d_gan += 1.

            if epoch > self.adv_seg_start:
                ## training segmentation discriminator

                # segment images
                s_seg = self.model(image=batch_s_img, meta=batch_s_met,
                                    seg=True, gan=False)
                t_seg = self.model(image=batch_t_img, meta=batch_t_met,
                                    seg=True, gan=False)

                # test discriminator on segmentation
                gt_valid, _ = self.disc_seg(batch_s_ref)
                s_valid, s_class = self.disc_seg(s_seg)
                t_valid, t_class = self.disc_seg(t_seg)

                # calculate wasserstein gradient penalty
                grad_pen = 0.5 * (self.wass_gp(self.disc_seg, batch_s_ref, s_seg) + \
                    self.wass_gp(self.disc_seg, batch_s_ref, t_seg))

                # adversarial
                loss_adv_d = -torch.mean(gt_valid) + \
                    0.5 * (torch.mean(s_valid) + torch.mean(t_valid)) + \
                        self.lambda_gp * grad_pen
                # domain
                loss_dom_d = 0.5 * (self.domain_loss(s_class.view(-1, s_class.shape[-1]), torch.max(batch_s_met, -1)[1].view(-1)) + \
                    self.domain_loss(t_class.view(-1, t_class.shape[-1]), torch.max(batch_t_met, -1)[1].view(-1)))

                # calculate overall loss
                loss_d_seg = loss_adv_d + self.lambda_domain * loss_dom_d

                losses['loss_adv_d_seg'] = loss_adv_d
                losses['loss_dom_d_seg'] = loss_dom_d
                losses['loss_d_seg'] = loss_d_seg

                loss_d_seg.backward()
                self.optim_d_seg.step()

                nb_d_seg += 1.

            if n_batch > 0 and n_batch % self.gen_interval == 0:
                ## training translation generator

                # source -> target
                s_t_img = self.model(image=batch_s_img, meta=batch_s_met,
                                    seg=False, gan=True,
                                    gan_meta=batch_t_met)

                fake_valid, fake_class = self.disc_gan(s_t_img)

                loss_g_adv = - torch.mean(fake_valid)

                loss_g_dom = self.domain_loss(fake_class.view(-1, fake_class.shape[-1]), torch.max(batch_t_met, -1)[1].view(-1))

                # target -> source
                t_s_img = self.model(image=batch_s_img, meta=batch_s_met,
                                    seg=False, gan=True,
                                    gan_meta=batch_t_met)

                fake_valid, fake_class = self.disc_gan(t_s_img)

                loss_g_adv += - torch.mean(fake_valid)

                loss_g_dom += self.domain_loss(fake_class.view(-1, fake_class.shape[-1]), torch.max(batch_s_met, -1)[1].view(-1))

                # source -> target -> source

                s_t_s_img = self.model(image=s_t_img, meta=batch_t_met,
                                    seg=False, gan=True,
                                    gan_meta=batch_s_met)

                loss_g_cyc = self.cycle_loss(s_t_s_img, batch_s_img)

                # target -> source -> target

                t_s_t_img = self.model(image=t_s_img, meta=batch_s_met,
                                    seg=False, gan=True,
                                    gan_meta=batch_t_met)

                loss_g_cyc += self.cycle_loss(t_s_t_img, batch_t_img)
                
                # source -> source

                s_s_img = self.model(image=batch_s_img, meta=batch_s_met,
                                    seg=False, gan=True,
                                    gan_meta=batch_s_met)

                loss_g_id = self.cycle_loss(s_s_img, batch_s_img)
                
                # target -> target

                t_t_img = self.model(image=batch_t_img, meta=batch_t_met,
                                    seg=False, gan=True,
                                    gan_meta=batch_t_met)

                loss_g_id += self.cycle_loss(t_t_img, batch_t_img)

                # overall loss

                loss_g = loss_g_adv + self.lambda_domain * loss_g_dom + \
                    self.lambda_cycle * loss_g_cyc + self.lambda_id * loss_g_id

                losses['loss_g_adv'] = loss_g_adv
                losses['loss_g_dom'] = loss_g_dom
                losses['loss_g_cyc'] = loss_g_cyc
                losses['loss_g_id'] = loss_g_id
                losses['loss_g'] = loss_g

                loss_g.backward()
                self.optimizer.step()

                nb_gan += 1.

            if n_batch > 0 and n_batch % self.seg_interval == 0:
                ## training segmentation 'generator' via Dice
                self.optimizer.zero_grad()

                # segment images
                s_seg = self.model(image=batch_s_img, meta=batch_s_met,
                                    seg=True, gan=False)

                # supervised learning of source -> label
                loss_seg_sup = self.seg_loss(s_seg, batch_s_ref)

                if epoch > self.adv_seg_start:

                    t_seg = self.model(image=batch_t_img, meta=batch_t_met,
                                        seg=True, gan=False)

                    s_t_img = self.model(image=batch_s_img, meta=batch_s_met,
                                        seg=False, gan=True, 
                                        gan_meta=batch_t_met)
                    s_t_seg = self.model(image=s_t_img, meta=batch_t_met,
                                        seg=True, gan=False)

                    t_s_img = self.model(image=batch_t_img, meta=batch_t_met,
                                        seg=False, gan=True, 
                                        gan_meta=batch_s_met)
                    t_s_seg = self.model(image=t_s_img, meta=batch_t_met,
                                        seg=True, gan=False)

                    # supervised learning of source -> target -> label
                    loss_seg_synth = self.seg_loss(s_t_seg, batch_s_ref)

                    # test discriminator on segmentation
                    s_valid, s_class = self.disc_seg(s_seg)
                    t_valid, t_class = self.disc_seg(t_seg)
                    s_t_valid, s_t_class = self.disc_seg(s_t_seg)
                    t_s_valid, t_s_class = self.disc_seg(t_s_seg)

                    # adversarial
                    loss_seg_adv = -torch.mean(s_valid)
                    loss_seg_adv += -torch.mean(t_valid)
                    loss_seg_adv += -torch.mean(s_t_valid)
                    loss_seg_adv += -torch.mean(t_s_valid)

                    # domain
                    loss_seg_dom = self.domain_loss(s_class.view(-1, s_class.shape[-1]), torch.max(batch_s_met, -1)[1].view(-1))
                    loss_seg_dom += self.domain_loss(t_class.view(-1, t_class.shape[-1]), torch.max(batch_t_met, -1)[1].view(-1))
                    loss_seg_dom += self.domain_loss(s_t_class.view(-1, s_t_class.shape[-1]), torch.max(batch_t_met, -1)[1].view(-1))
                    loss_seg_dom += self.domain_loss(t_s_class.view(-1, t_s_class.shape[-1]), torch.max(batch_s_met, -1)[1].view(-1))

                    # calculate overall loss
                    loss_seg = loss_seg_sup + self.lambda_seg_synth * loss_seg_synth + \
                        self.lambda_seg_adv * loss_seg_adv + self.lambda_seg_domain * loss_seg_dom

                    losses['loss_seg_sup'] = loss_seg_sup
                    losses['loss_seg_synth'] = loss_seg_synth
                    losses['loss_seg_adv'] = loss_seg_adv
                    losses['loss_seg_domain'] = loss_seg_dom

                else:

                    # only use T1 segmentation loss
                    loss_seg = loss_seg_sup

                    losses['loss_seg_sup'] = loss_seg_sup

                losses['loss_seg'] = loss_seg

                loss_seg.backward()
                self.optimizer.step()

                nb_seg += 1.
            # update average across batches
            with torch.no_grad():
                weight = float(batch_s[0].shape[0])

                epoch_loss_d_gan += loss_d_gan * weight
                epoch_loss_d_seg += loss_d_seg * weight
                epoch_loss_g += loss_g * weight
                epoch_loss_seg += loss_seg * weight

                loss = loss_d_gan + loss_d_seg + loss_g + loss_seg

                update_loss_dict(epoch_losses, losses, weight)
                update_loss_dict(epoch_metrics, metrics, weight)
                # print
                if n_batch % self.log_interval == 0:
                    self._print('train', epoch, n_batch+1, nb_steps,
                                loss, losses, metrics)
                # tb callback
                if self.tensorboard:
                    tbopt = dict(inputs=batch, outputs=output,
                                 epoch=epoch, minibatch=n_batch, mode='train',
                                 loss=loss, losses=losses, metrics=metrics)
                    self.model.board(self.tensorboard, **tbopt)
                    for func in self._tensorboard_callbacks['train']['step']:
                        func(self.tensorboard, **tbopt)
                    del tbopt
        # print summary
        with torch.no_grad():
            if nb_d_gan > 0:
                epoch_loss_d_gan /= nb_d_gan
            if nb_d_seg > 0:
                epoch_loss_d_seg /= nb_d_seg
            if nb_gan > 0:
                epoch_loss_g /= nb_gan
            if nb_seg > 0:
                epoch_loss_seg /= nb_seg

            epoch_loss = epoch_loss_d_gan + epoch_loss_d_seg + epoch_loss_g + epoch_loss_seg

            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('train', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)
            self._board('train', epoch, epoch_loss, epoch_metrics)
            # tb callback
            if self.tensorboard:
                tbopt = dict(epoch=epoch, loss=epoch_loss, mode='train',
                             losses=epoch_losses, metrics=epoch_metrics)
                self.model.board(self.tensorboard, **tbopt)
                for func in self._tensorboard_callbacks['train']['epoch']:
                    func(self.tensorboard, **tbopt)
        print('D_G loss: {}\nD_S loss: {}\nG loss: {}\nS loss: {}'.format(epoch_loss_d_gan, epoch_loss_d_seg, epoch_loss_g, epoch_loss_seg))
        return epoch_loss

    def _train_seg(self, epoch=0):
        """Train segmentation for one epoch"""

        self.model.train()
        epoch_loss = 0.
        epoch_losses = {}
        epoch_metrics = {}
        nb_batches = 0
        nb_steps = len(self.train_set)
        for n_batch, batch in enumerate(self.train_set):
            losses = {}
            metrics = {}
            # forward pass
            batch = make_tuple(batch)
            batch = tuple(torch.as_tensor(b, device=self.device) for b in batch)
            batch = tuple(b.to(dtype=self.dtype)
                          if b.dtype in (torch.half, torch.float, torch.double)
                          else b for b in batch)
            nb_batches += batch[0].shape[0]
            self.optimizer.zero_grad()
            output = self.model(*batch, _loss=losses, _metric=metrics)
            loss = sum(losses.values())
            # backward pass
            loss.backward()
            self.optimizer.step()
            # update average across batches
            with torch.no_grad():
                weight = float(batch[0].shape[0])
                epoch_loss += loss * weight
                update_loss_dict(epoch_losses, losses, weight)
                update_loss_dict(epoch_metrics, metrics, weight)
                # print
                if n_batch % self.log_interval == 0:
                    self._print('train', epoch, n_batch+1, nb_steps,
                                loss, losses, metrics)
                # tb callback
                if self.tensorboard:
                    tbopt = dict(inputs=batch, outputs=output,
                                 epoch=epoch, minibatch=n_batch, mode='train',
                                 loss=loss, losses=losses, metrics=metrics)
                    self.model.board(self.tensorboard, **tbopt)
                    for func in self._tensorboard_callbacks['train']['step']:
                        func(self.tensorboard, **tbopt)
                    del tbopt
        # print summary
        with torch.no_grad():
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('train', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)
            self._board('train', epoch, epoch_loss, epoch_metrics)
            # tb callback
            if self.tensorboard:
                tbopt = dict(epoch=epoch, loss=epoch_loss, mode='train',
                             losses=epoch_losses, metrics=epoch_metrics)
                self.model.board(self.tensorboard, **tbopt)
                for func in self._tensorboard_callbacks['train']['epoch']:
                    func(self.tensorboard, **tbopt)

        return epoch_loss

    def _eval(self, epoch=0):
        """Evaluate once"""
        if self.eval_set is None:
            return

        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_losses = {}
            epoch_metrics = {}
            nb_batches = 0
            nb_steps = len(self.eval_set)
            for n_batch, batch in enumerate(self.eval_set):
                losses = {}
                metrics = {}
                # forward pass
                batch = make_tuple(batch)
                batch = tuple(torch.as_tensor(b, device=self.device) for b in batch)
                batch = tuple(b.to(dtype=self.dtype)
                              if b.dtype in (torch.half, torch.float, torch.double)
                              else b for b in batch)
                nb_batches += batch[0].shape[0]
                self.optimizer.zero_grad()
                output = self.model(*batch, _loss=losses, _metric=metrics)
                loss = sum(losses.values())
                # update average across batches
                weight = float(batch[0].shape[0])
                epoch_loss += loss * weight
                update_loss_dict(epoch_losses, losses, weight)
                update_loss_dict(epoch_metrics, metrics, weight)
                # print
                if n_batch % self.log_interval == 0:
                    self._print('eval', epoch, n_batch + 1, nb_steps,
                                loss, losses, metrics)
                # tb callback
                if self.tensorboard:
                    tbopt = dict(inputs=batch, outputs=output,
                                 epoch=epoch, minibatch=n_batch, mode='eval',
                                 loss=loss, losses=losses, metrics=metrics)
                    self.model.board(self.tensorboard, **tbopt)
                    for func in self._tensorboard_callbacks['eval']['step']:
                        func(self.tensorboard, **tbopt)

            # print summary
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('eval', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)
            self._board('eval', epoch, epoch_loss, epoch_metrics)
            # tb callback
            if self.tensorboard:
                tbopt = dict(epoch=epoch, loss=epoch_loss, mode='eval',
                             losses=epoch_losses, metrics=epoch_metrics)
                self.model.board(self.tensorboard, **tbopt)
                for func in self._tensorboard_callbacks['eval']['epoch']:
                    func(self.tensorboard, **tbopt)

        return epoch_loss

    def _print(self, mode, n_epoch, n_batch, nb_steps, loss,
               losses=None, metrics=None, last=False):
        """Pretty printing

        Parameters
        ----------
        mode : {'train', 'eval'}
        n_epoch : int
            Index of current epoch (starts at one)
        n_batch : int
            Index of current batch (starts at one)
        nb_steps : int
            Total number of batches
        loss : () tensor
            Loss for this batch
        losses : dict[str: () tensor]
            Loss components for this batch
        metrics : dict[str: () tensor]
            Metrics for this batch
        last : bool, default=False
            Is this the end of the batch?
            If True, loss/losses/metrics should contain the average loss
            across all batches.

        """
        name = 'Train' if mode == 'train' else 'Eval '
        if last:
            pct = 1
            bar = '[' + '=' * 10 + ']'
        else:
            pct = n_batch/nb_steps
            len_arrow = min(math.floor(pct*10 + 0.5), 9)
            bar = '[' + '=' * len_arrow + '>' + ' ' * (9-len_arrow) + ']'

        lepoch = str(len(str(self.nb_epoch)))
        evolution = '{:s} | {:' + lepoch + 'd} | {:3.0f}% ' + bar + ' '
        evolution = evolution.format(name, n_epoch, pct*100)

        values = ''
        if mode == 'train':
            values += '| loss = {:12.6g} '.format(loss.item())
            if losses and self.show_losses:
                values += '|'
                for key, val in losses.items():
                    values += ' {}: {:12.6g} '.format(key, val.item())
        if metrics and (mode == 'eval' or self.show_metrics):
            values += '|'
            for key, val in metrics.items():
                values += ' {}: {:12.6g} '.format(key, val.item())

        print(evolution + values, end='\r', flush=True)
        if last:
            print('')

    def _board(self, mode, epoch, loss, epoch_metrics):
        """Add losses and metrics to tensorboard."""
        if not self.tensorboard:
            return
        tb = self.tensorboard
        tb.add_scalars('loss', {mode: loss.item()}, epoch)
        for tag, value in epoch_metrics.items():
            tb.add_scalars(tag, {mode: value.item()}, epoch)
        tb.flush()

    def add_tensorboard_callback(self, func, mode='train', trigger='epoch'):
        """Register tensorboard callbacks

        Parameters
        ----------
        func : callable
            If trigger 'step', with signature
                `(tb, input, output, epoch, step, loss, losses, metrics)`
            If trigger 'epoch', with signature:
                `(tb, epoch, loss, losses, metrics)`
        mode : {'train', 'eval'}
            Trigger either during a training or evaluation call.
        trigger : {'epoch', 'step'}
            Trigger either at the end of a step or at the end of an epoch.

        """
        if mode not in self._tensorboard_callbacks.keys():
            self._tensorboard_callbacks[mode] = dict()
        if trigger not in self._tensorboard_callbacks[mode].keys():
            self._tensorboard_callbacks[mode][trigger] = list()
        self._tensorboard_callbacks[mode][trigger].append(func)

    def _hello(self, mode):
        """Tell the use what we are going to do (mode, device, dtype, ...)

        Parameters
        ----------
        mode : {'train', 'eval'}

        """
        if self.device.type == 'cuda':
            device = torch.cuda.get_device_name(self.device)
        else:
            assert self.device.type == 'cpu'
            device = 'CPU'
        dtype = str(self.dtype).split('.')[-1]
        if mode == 'train':
            hello = 'Training model {} for {} epochs (steps per epoch: {}) ' \
                    'on {} (dtype = {})'
            hello = hello.format(type(self.model).__name__, self.nb_epoch,
                                 len(self.train_set), device, dtype)
        else:
            hello = 'Evaluating model {} (minibatches: {}) on {} (dtype = {})'
            hello = hello.format(type(self.model).__name__,
                                 len(self.eval_set), device, dtype)
        print(hello, flush=True)

    def _save(self, epoch):
        """Save once"""
        if self.save_model:
            save_model = self._formatfile(self.save_model, epoch)
            dir_model = os.path.dirname(save_model)
            if dir_model:
                os.makedirs(dir_model, exist_ok=True)
            torch.save(self.model.state_dict(), save_model)
        if self.save_optimizer:
            save_optimizer = self._formatfile(self.save_optimizer, epoch)
            dir_optimizer = os.path.dirname(save_optimizer)
            if dir_optimizer:
                os.makedirs(dir_optimizer, exist_ok=True)
            torch.save(self.optimizer.state_dict(), save_optimizer)

    @staticmethod
    def _formatfile(file, epoch):
        """Format filename for an epoch"""
        keys = [tup[1] for tup in string.Formatter().parse(file)
                if tup[1] is not None]
        if len(keys) == 1:
            file = file.format(epoch)
        elif len(keys) > 1:
            raise ValueError('Cannot have more than one format key')
        return file

    def train(self):
        """Launch training"""
        self._hello('train')
        with torch.random.fork_rng(enabled=self.seed is not None):
            if self.seed is not None:
                torch.random.manual_seed(self.seed)
            self.initial_seed = torch.random.initial_seed()
            with benchmark(self.benchmark):
                self.model.to(dtype=self.dtype, device=self.device)
                self.epoch = self.initial_epoch
                self._eval(self.epoch)
                self._save(self.epoch)
                for self.epoch in range(self.epoch+1, self.nb_epoch+1):
                    train_loss = self._train(self.epoch)
                    print('Train loss: {}'.format(train_loss))
                    val_loss = self._eval(self.epoch)
                    self._save(self.epoch)
                    # scheduler
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        sched_loss = val_loss or train_loss
                        self.scheduler.step(sched_loss)
                    elif self.scheduler:
                        self.scheduler.step()

    def eval(self):
        """Launch evaluation"""
        self._hello('eval')
        self.model.to(dtype=self.dtype, device=self.device)
        self._eval()

    def init(self):
        """Initialize the random state + run one evaluation."""
        with torch.random.fork_rng(enabled=self.seed is not None):
            if self.seed is not None:
                torch.random.manual_seed(self.seed)
            self.initial_seed = torch.random.initial_seed()
            self.save_random_state()
            self.epoch = self.initial_epoch
            self.model.to(dtype=self.dtype, device=self.device)
            self._eval(self.epoch)
            self._save(self.epoch)

    def set_random_state(self):
        """Populate the random state using a saved state."""
        if self.random_state:
            cpu_state, *gpu_states = self.random_state
            devices = list(range(torch.cuda.device_count()))
            torch.set_rng_state(self.random_state[0])
            for device, state in zip(devices, gpu_states):
                torch.cuda.set_rng_state(state, device)

    def save_random_state(self):
        """Save the current random state."""
        devices = list(range(torch.cuda.device_count()))
        self.random_state = [torch.get_rng_state()]
        self.random_state.extend(torch.cuda.get_rng_state(device)
                                 for device in devices)

    def train1(self):
        """Train for one epoch."""
        with torch.random.fork_rng():
            self.set_random_state()
            self.model.to(dtype=self.dtype, device=self.device)
            self.epoch += 1
            self._train(self.epoch)
            self._eval(self.epoch)
            self._save(self.epoch)
            self.save_random_state()
