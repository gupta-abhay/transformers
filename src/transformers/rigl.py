import torch
import torch.nn as nn
import numpy as np
from .rigl_utils import (
    extract_number,
    get_all_layers_with_weight,
    get_mask_random,
    get_mask_lamp,
    get_mask_topk,
    get_global_mask_topk,
    get_uniform_sparsity_distribution,
    get_erk_sparsity_distribution,
    calc_sparsity,
)
# from .rigl_sparsity_utils import get_model_complexity_info  # noqa
import pdb, sys  # noqa
from packaging import version

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimizer(object):
    """Implementation of optimizer.
        Basic class of optimizer.
        This optimizer wraps a regular optimizer and performs static masks
        according to the default sparsity.
        Attributes:
            model: torch.nn
            optimizer: torch.Optimizer
            default_sparsity: float, the sparsity of nn
            mask_init_method: str, of method for initialization
    """

    def __init__(
        self,
        model,
        optimizer,
        default_sparsity=0.3,
        mask_init_method='random',
    ):
        super(Optimizer, self).__init__()

        self._optimizer = optimizer
        self._default_sparsity = default_sparsity
        self._mask_init_method = mask_init_method

    def step(self, inputs=None, lr_scheduler=None):
        # update parameters
        self._optimizer.step()
        self._optimizer.zero_grad()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)


class SparseOptimiser:
    def __init__(self, sparse_optimizer, step_number=0, **opt_params):
        self.step_number = step_number
        self.opt_params = opt_params
        self._sparse_opt = sparse_optimizer
        self._opt = sparse_optimizer._optimizer

    def param_values(self):
        return {
            k: v(self.step_number) if callable(v) else v
            for k, v in self.opt_params.items()
        }

    def step(self, inputs=None, lr_scheduler=None):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._sparse_opt.step(inputs, lr_scheduler)

    @property
    def param_groups(self):
        return self._opt.param_groups

    def __repr__(self):
        return repr(self._sparse_opt)

    def state_dict(self):
        return {
            "step_number": self.step_number,
            **self._sparse_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step_number = state_dict.pop("step_number")
        self._sparse_opt.load_state_dict(state_dict)


class SparseSETOptimizer(Optimizer):
    """Implementation of dynamic sparsity optimizers.
        Implementation of SET.
        See https://www.nature.com/articles/s41467-018-04316-3
        This optimizer wraps a regular optimizer and performs updates on the
        masks according to schedule given.
        Attributes:
            model: torch.nn
            optimizer: torch.Optimizer
            begin_iteration: float, first iteration where masks are updated.
            end_iteration: float, iteration after which no mask is updated.
            train_loader: dataset, contain samples of each batch
            frequency: int, of mask update operations.
            default_sparsity: float, the sparsity of nn
            mask_init_method: str, of method for initialization
            drop_fraction: float, of connections to drop during each update.
            drop_fraction_anneal: str or None, if supplied used to anneal the
              drop fraction.
            grow_init: str, name of the method used to initialize new
              connections.
    """

    def __init__(
        self,
        model,
        optimizer,
        begin_iteration,
        end_iteration,
        mask_distribution,
        erk_power_scale,
        frequency,
        default_sparsity=0.3,
        mask_init_method='random',
        drop_fraction=0.1,
        drop_fraction_anneal='constant',
        grow_init='zeros',
        grad_accum_steps=1,
    ):
        super(SparseSETOptimizer, self).__init__(
            model,
            optimizer,
            default_sparsity=default_sparsity,
            mask_init_method=mask_init_method,
        )

        self._grow_init = grow_init
        self._drop_fraction_anneal = drop_fraction_anneal
        self._drop_fraction_initial_value = drop_fraction
        self._drop_fraction = drop_fraction
        self._begin_iteration = begin_iteration
        self._end_iteration = end_iteration
        self._frequency = frequency
        self._frequency_val = frequency
        self._mask_distribution = mask_distribution
        self._erk_power_scale = erk_power_scale
        self._grad_accum_steps = grad_accum_steps
        self.global_step = 0

        # get the names and parameters
        layers = get_all_layers_with_weight(model)
        names = [
            name
            for name, module in model.named_modules()
            if 'weight' in module._parameters or hasattr(module, "weight")
        ]

        masks = []
        final_layers = []
        final_names = []
        total_params = 0
        for i, layer in enumerate(layers):
            # remove the embedding, normalization, linear layers not in encoder
            layer_name = layer.__class__.__name__
            name = names[i]
            if ('head' in name) or ('pooler' in name):
                continue
            elif layer_name == 'Embedding':
                continue
            elif layer_name == 'LayerNorm':
                continue
            else:
                weight = layer.weight
                masks.append(
                    torch.ones(weight.size(), dtype=torch.bool, device=device)
                )
                final_layers.append(layer)
                final_names.append(name)
                total_params += weight.numel()

        if self._mask_distribution == "uniform":
            sparsity_dict = get_uniform_sparsity_distribution(
                final_names, self._default_sparsity
            )
        elif self._mask_distribution == "erk":
            sparsity_dict = get_erk_sparsity_distribution(
                final_names,
                masks,
                total_params,
                density=1.0 - self._default_sparsity,
                erk_power_scale=self._erk_power_scale,
            )

        self.layers = final_layers
        self.names = final_names
        self.masks = masks
        self.sparsity_dict = sparsity_dict
        self.setup_graph()
        self.masks_applied = False

    def state_dict(self):
        return {
            **super(SparseSETOptimizer, self).state_dict(),
            "global_step": self.global_step,
            "masks": self.masks,
            "sparsity_dict": self.sparsity_dict,
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict.pop("global_step")
        self.masks = state_dict.pop("masks")
        self.sparsity_dict = state_dict.pop("sparsity_dict")
        super(SparseSETOptimizer, self).load_state_dict(state_dict)

        # initialize masked weight
        for i, layer in enumerate(self.layers):
            assert torch.allclose(layer.weight, layer.weight * self.masks[i])

    def setup_graph(self):
        # initialize the masks
        if self._mask_init_method == 'random':
            for i, (name, mask) in enumerate(zip(self.names, self.masks)):
                size = mask.size()
                self.masks[i] = torch.tensor(
                    get_mask_random(size, self.sparsity_dict.get(name, 0.0)),
                    dtype=mask.dtype,
                ).to(device)
        elif self._mask_init_method == 'lamp':
            for i, (name, layer) in enumerate(zip(self.names, self.layers)):
                self.masks[i] = get_mask_lamp(
                    layer, self.sparsity_dict.get(name, 0.0)
                ).to(device)
        elif self._mask_init_method == 'topk':
            for i, (name, layer) in enumerate(zip(self.names, self.layers)):
                self.masks[i] = get_mask_topk(
                    layer, self.sparsity_dict.get(name, 0.0)
                ).to(device)
        elif self._mask_init_method == 'global_topk':
            self.masks = get_global_mask_topk(
                self.layers, self._default_sparsity
            )
            self.masks = [mask.to(device) for mask in self.masks]

        # initialize masked weight
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                layer.weight.mul_(self.masks[i])

    def apply_masks(self):
        # Applies the current masks to the weights. This needs to be called
        # after the optimizer is initialized and any state dicts loaded. This
        # used to be called in setup_graph(), but this did not work with state
        # dict loading. The model weights would be loaded, then, when the
        # optimizer was initialized, setup_graph() would be called and random
        # masks applied. This "double-masked" the weights!
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                layer.weight.mul_(self.masks[i])
        self.masks_applied = True

    def step(self, inputs=None, lr_scheduler=None):
        assert self.masks_applied, (
            "step() called before apply_masks();"
            "you need to apply the masks after initializaing and loading!"
        )
        # update parameters
        self._optimizer.step()
        self._optimizer.zero_grad()

        # update fraction
        if (
            self._begin_iteration <= self.global_step < self._end_iteration
            and self.global_step % self._frequency == 0
        ):
            self.update_topology()

        self.global_step += 1

    def update_topology(self):
        # update fraction
        self.update_drop_fraction()

        masks = self.masks
        new_masks = []

        # update topology
        for i, layer in enumerate(self.layers):
            prev_mask = masks[i]
            masks[i], new_mask = self.update_layer_mask(layer, masks[i])
            # hamming_distance = distance.hamming(
            #     np.ravel(prev_mask.float().view(-1).cpu()),
            #     np.ravel(masks[i].float().view(-1).cpu()),
            # )
            # if (self.global_step + 1) % 1000:
            #     self._writer.add_scalar(
            #         'hamming_distance/' + self.names[i],
            #         hamming_distance,
            #         self.global_step,
            #     )
            new_masks.append(new_mask)

        # clear the masked momentum gradient
        self.reset_momentum(masks, new_masks)

    def update_drop_fraction(self):
        """Returns a constant or annealing drop_fraction op."""
        cur_iter = self.global_step
        if self._drop_fraction_anneal == 'constant':
            self._drop_fraction = self._drop_fraction_initial_value
        elif self._drop_fraction_anneal == 'cosine':
            decay_iteratons = self._end_iteration - self._begin_iteration
            self._drop_fraction = self._drop_fraction_initial_value * (
                np.cos(
                    np.pi * (cur_iter - self._begin_iteration) / decay_iteratons
                )
                * 0.5
                + 0.5
            )
        elif self._drop_fraction_anneal.startswith('exponential'):
            decay_iteratons = self._end_iteration - self._begin_iteration
            exponent = extract_number(self._drop_fraction_anneal)
            self._drop_fraction = self._drop_fraction_initial_value * np.exp(
                -exponent * (cur_iter - self._begin_iteration) / decay_iteratons
            )
        else:
            raise ValueError(
                f'{self._drop_fraction_anneal} is not a valid annealer'
            )

    def update_layer_mask(self, layer, layer_mask, noise_std=1e-5):
        layer_weight = layer.weight

        # Add noise for slight bit of randomness and drop
        masked_weights = layer_mask * layer_weight
        score_drop = masked_weights.abs() + self._random_normal(
            layer_weight.size(), std=noise_std
        )
        layer_mask_dropped, n_prune = self.drop_minimum(score_drop, layer_mask)

        # Randomly revive n_prune many connections from non-existing
        # connections.
        score_grow = self._random_uniform(layer_weight.size()) * (
            ~layer_mask_dropped
        )
        layer_mask, new_mask = self.grow_maximum(
            score_grow, layer_mask_dropped, n_prune
        )

        # update the weight
        with torch.no_grad():
            layer.weight.mul_(layer_mask)

        return layer_mask, new_mask

    def drop_minimum(self, layer_score, layer_mask):
        """
        drop the weights with minimum score (weight or the gradient of weight)
        """
        n_ones = int(layer_mask.sum())
        n_prune = int(self._drop_fraction * n_ones)
        n_keep = n_ones - n_prune

        _, indices = layer_score.view([-1]).sort(descending=True)
        indices = indices[:n_keep]

        layer_mask_drop = torch.zeros(
            layer_mask.size(), dtype=layer_mask.dtype
        ).to(device)
        mask_vec = layer_mask_drop.view([-1])
        mask_vec[indices] = True

        return layer_mask_drop, n_prune

    def grow_maximum(self, layer_score, layer_mask, n_prune):
        """
        grow the weights with maximum score (weight or the gradient of weight)
        """
        _, indices = layer_score.view([-1]).sort(descending=True)
        indices = indices[:n_prune]

        mask_vec = layer_mask.view([-1])
        mask_vec[indices] = True

        new_mask = torch.zeros(layer_mask.size(), dtype=layer_mask.dtype).to(
            device
        )
        new_mask.view([-1])[indices] = True

        return layer_mask, new_mask

    def reset_momentum(self, masks, new_masks):
        optimizer = self._optimizer
        group = optimizer.param_groups[0]
        beta1, beta2 = group["betas"]
        if beta1 != 0 and beta2 != 0:
            for i, layer in enumerate(self.layers):
                weight_orig = layer.weight_orig
                param_state = optimizer.state[weight_orig]
                param_state['exp_avg'].mul_(masks[i])
                param_state['exp_avg'].mul_(~new_masks[i])
                param_state['exp_avg_sq'].mul_(masks[i])
                param_state['exp_avg_sq'].mul_(~new_masks[i])

    def get_grow_tensor(self, weights, method):
        """Different ways to initialize new connections.
        Args:
          weights: torch.Tensor or Variable.
          method: str, available options: 'zeros', 'random_normal',
            'random_uniform' and 'initial_value'
        Returns:
          torch.Tensor same shape and type as weights.
        Raises:
          ValueError, when the method is not valid.
        """
        if not isinstance(method, str):
            raise ValueError('Grow-Init: {} is not a string'.format(method))

        if method == 'zeros':
            grow_tensor = torch.zeros(weights, dtype=weights.dtype)
        elif method.startswith('initial_dist'):
            original_size = weights.size()
            divisor = extract_number(method)
            grow_tensor = (
                torch.reshape(
                    weights.view([-1])[torch.randperm(len(weights.view([-1])))],
                    original_size,
                )
                / divisor
            )
        elif method.startswith('random_normal'):
            original_size = weights.size()
            std = weights.std()
            divisor = extract_number(method)
            grow_tensor = (
                self._random_normal(original_size, std=std, dtype=weights.dtype)
                / divisor
            )
        elif method.startswith('random_uniform'):
            original_size = weights.size()
            mean = weights.abs().mean()
            divisor = extract_number(method)
            grow_tensor = (
                self._random_uniform(
                    original_size,
                    minval=-mean,
                    maxval=mean,
                    dtype=weights.dtype,
                )
                / divisor
            )
        else:
            raise ValueError('Grow-Init: %s is not a valid option.' % method)
        return grow_tensor

    def _random_uniform(self, size, minval=None, maxval=None, dtype=None):
        if minval and maxval:
            return (
                torch.rand(size, dtype=dtype)
                .mul(maxval - minval)
                .add(minval)
                .to(device)
            )
        else:
            return torch.rand(size, dtype=dtype).to(device)

    def _random_normal(self, size, std, dtype=None):
        return torch.randn(size, dtype=dtype).mul(std).to(device)


class SparseRigLOptimizer(SparseSETOptimizer):
    """Sparse optimizer that grows connections with the pre-removal gradients.
    Implementation of RigL
    https://arxiv.org/abs/1911.11134
    Attributes:
        model: torch.nn
        optimizer: torch.Optimizer
        begin_iteration: float, first iteration where masks are updated.
        end_iteration: float, iteration after which no mask is updated.
        train_loader: dataset, contain samples of each batch.
        frequency: int, of mask update operations.
        drop_fraction: float, of connections to drop during each update.
        drop_fraction_anneal: str or None, if supplied used to anneal the drop
          fraction.
        grow_init: str, name of the method used to initialize new connections.
        initial_acc_scale: float, used to scale the gradient when initializing
          the, momentum values of new connections. We hope this will improve
          training, compare to starting from 0 for the new connections. Set
          this to something between 0 and 1 / (1 - momentum). This is because
          in the current implementation of MomentumOptimizer, aggregated values
          converge to 1 / (1 - momentum) with constant gradients.
    """

    def __init__(
        self,
        model,
        optimizer,
        begin_iteration,
        end_iteration,
        mask_distribution='uniform',
        mask_init_method='random',
        erk_power_scale=1.0,
        frequency=50,
        default_sparsity=0.3,
        drop_fraction=0.1,
        drop_fraction_anneal='constant',
        grow_init='zeros',
        initial_acc_scale=0.0,
        grad_accum_steps=1,
    ):
        super(SparseRigLOptimizer, self).__init__(
            model,
            optimizer,
            begin_iteration,
            end_iteration,
            mask_distribution=mask_distribution,
            mask_init_method=mask_init_method,
            erk_power_scale=erk_power_scale,
            frequency=frequency,
            default_sparsity=default_sparsity,
            drop_fraction=drop_fraction,
            drop_fraction_anneal=drop_fraction_anneal,
            grow_init=grow_init,
            grad_accum_steps=grad_accum_steps,
        )
        self._initial_acc_scale = initial_acc_scale
        self.model = model
        self.redensify = False
        self.grad_dict = None
        self.prev_iteration = None

    def state_dict(self):
        return {
            **super(SparseRigLOptimizer, self).state_dict(),
            "grad_dict": self.grad_dict,
        }

    def load_state_dict(self, state_dict):
        self.grad_dict = state_dict.pop("grad_dict")
        super(SparseRigLOptimizer, self).load_state_dict(state_dict)

    def step(self, inputs=None, lr_scheduler=None):
        assert self.masks_applied, (
            "step() called before apply_masks();"
            "you need to apply the masks after initializaing and loading!"
        )
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = (
                # backward compatibility for pytorch schedulers
                lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else lr_scheduler.get_lr()[0]
            )

        # update parameters
        self._optimizer.step()
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                layer.weight.mul_(self.masks[i])
        self._optimizer.zero_grad()

        # update fraction
        if (
            self._begin_iteration <= self.global_step < self._end_iteration
            and self.global_step % self._frequency == 0
        ):
            self.compute_gradients(inputs)
            self.update_topology()
            self._optimizer.zero_grad()

        self.global_step += 1

    def compute_gradients(self, inputs):
        """Wraps the compute gradient of passed optimizer."""

        # fwd, bwd and calculate gradients
        for i, _input in enumerate(inputs):
            outputs = self.model(**_input)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if self._grad_accum_steps > 1:
                loss = loss / self._grad_accum_steps
            loss.backward()

        # get gradient
        gradient = []
        for layer in self.layers:
            gradient.append(layer.weight.grad.data)

        self.grad_dict = dict(zip(self.layers, gradient))

    def update_layer_mask(
        self, layer, layer_mask, noise_std=1e-5, grow_noise_std=1e-6
    ):
        layer_weight = layer.weight

        # Add noise for slight bit of randomness and drop
        masked_weights = layer_mask * layer_weight
        score_drop = masked_weights.abs() + self._random_normal(
            layer_weight.size(), std=noise_std
        )
        layer_mask_dropped, n_prune = self.drop_minimum(score_drop, layer_mask)

        # Randomly revive n_prune many connections from
        # non-existing connections.
        score_grow = (
            self.grad_dict[layer].abs()
            + self._random_uniform(layer_weight.size()) * grow_noise_std
        ) * (~layer_mask_dropped)
        layer_mask, new_mask = self.grow_maximum(
            score_grow, layer_mask_dropped, n_prune
        )

        # update the weight
        with torch.no_grad():
            layer.weight.mul_(layer_mask)

        return layer_mask, new_mask

    def reset_momentum(self, masks, new_masks):
        optimizer = self._optimizer
        for group in optimizer.param_groups:
            beta1, beta2 = group["betas"]
            if beta1 != 0 and beta2 != 0:
                for i, layer in enumerate(self.layers):
                    weight = layer.weight
                    param_state = optimizer.state[weight]
                    new_mask = new_masks[i]
                    param_state['exp_avg'].mul_(masks[i])
                    param_state['exp_avg'][new_mask] = (
                        self.grad_dict[layer][new_mask]
                        * self._initial_acc_scale
                    )
                    param_state['exp_avg_sq'].mul_(masks[i])
                    param_state['exp_avg_sq'][new_mask] = (
                        self.grad_dict[layer][new_mask]
                        * self._initial_acc_scale
                    )
