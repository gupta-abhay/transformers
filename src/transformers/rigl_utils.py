import re
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
not_device = torch.device("cpu" if torch.cuda.is_available() else "cuda")


def sub_modules(module):
    """get the sub modules of certain module, if no sub modules, return None
    Args:
        module
    Returns:
        sub_modules
    """
    return list(module.children())


def get_all_layers_with_weight(module, layer_list=None):
    """get all layers contain weight
    Args:
        module (nn.module)
        layers (list of modules)
    Returns:
        layers (list of modules)
    """
    if layer_list is None:
        layer_list = []

    if 'weight' in module._parameters:
        layer_list.append(module)
    elif hasattr(module, "weight"):
        layer_list.append(module)

    sub_layers = sub_modules(module)
    if len(sub_layers) > 0:
        for layer in sub_layers:
            layer_list = get_all_layers_with_weight(layer, layer_list)

    return layer_list


def get_n_zeros(size, sparsity):
    return int(np.ceil(sparsity * size))


def get_mask_random(shape, sparsity):
    """Creates a random sparse mask with deterministic sparsity.
    Args:
        shape: torch.Tensor, used to obtain shape of the random mask.
        sparsity: float, between 0 and 1.
        dtype: torch.dtype, type of the return value.
    Returns:
        numpy.ndarray
    """
    flat_ones = np.ones(shape).flatten()
    n_zeros = get_n_zeros(flat_ones.size, sparsity)
    flat_ones[:n_zeros] = 0

    np.random.shuffle(flat_ones)
    new_mask = flat_ones.reshape(shape)
    return new_mask


def extract_number(token):
    """Strips the number from the end of the token if it exists.
    Args:
        token: str, s or s_d where d is a number: a float or int.
        `foo_.5`, `foo_foo.5`, `foo_0.5`, `foo_4` are all valid strings.
    Returns:
    float, d if exists otherwise 1.
    """
    regexp = re.compile(r'.*_(\d*\.?\d*)$')
    if regexp.search(token):
        return float(regexp.search(token).group(1))
    else:
        return 1.0


def density(tensor):
    """Computes the density of a tensor.
    Density is the fraction of non-zero elements in a tensor.
    If a tensor has a density of 1.0, then it has no zero elements.
    Args:
        tensor: the tensor for which we compute the density.
    Returns:
        density (float)
    """
    # Using torch.nonzero(tensor) can lead to memory exhaustion on
    # very large tensors, so we count zeros "manually".
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)


def calc_sparsity(tensor):
    """Computes the sparsity of a tensor.
    Sparsity is the fraction of zero elements in a tensor.
    If a tensor has a density of 0.0, then it has all zero elements.
    Sparsity and density are complementary.
    Args:
        tensor: the tensor for which we compute the density.
    Returns:
        sparsity (float)
    """
    return 1.0 - density(tensor)


def sparsity(model, eps=1e-8):
    return weight_statstics(model, eps) / parameter_total(model)


def parameter_total(model):
    f = 0
    for p in model.parameters():
        f += np.prod(p.size()).item()
    return f


def weight_statstics(model, eps=1e-8):
    layers = get_all_layers_with_weight(model)
    f = 0
    for layer in layers:
        if 'weight' in layer._parameters:
            f += (layer._parameters['weight'].data.abs() <= eps).sum().item()
        elif hasattr(layer, "weight"):
            f += (layer.weight.data.abs() <= eps).sum().item()
    return f


def get_erk_sparsity_distribution(
    names, masks, total_params, density=1.0, erk_power_scale=1.0
):
    """

    Given the method, returns the sparsity of individual layers as a dict.
    It ensures that the non-custom layers have a total parameter count as the
    one with uniform sparsities. In other words for the layers which are not in
    the custom_sparsity_map the following equation should be satisfied.
    # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
    Args:
      names: Names of the different layers
      masks: pre-defined masks (for shapes)
      density: float, between 0 and 1.
      erk_power_scale: float, if given used to take power of the ratio. Use
        scale<1 to make the erdos_renyi softer.
    Returns:
      density_dict, dict of where keys() are equal to names and individiual
        masks are mapped to the their densities.
    """
    is_epsilon_valid = False
    dense_layers = set()

    while not is_epsilon_valid:
        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in zip(names, masks):
            n_param = np.prod(mask.shape)
            n_zeros = n_param * (1.0 - density)
            n_ones = n_param * density

            if name in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_probabilities[name] = (
                    np.sum(mask.shape) / np.prod(mask.shape)
                ) ** erk_power_scale
                divisor += raw_probabilities[name] * n_param

        epsilon = rhs / divisor
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    sparsity_dict = {}
    total_nonzero = 0.0
    for name, mask in zip(names, masks):
        n_param = np.prod(mask.shape)
        if name in dense_layers:
            sparsity_dict[name] = 0.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            sparsity_dict[name] = 1.0 - probability_one
        total_nonzero += sparsity_dict[name] * mask.numel()
    return sparsity_dict


def get_uniform_sparsity_distribution(names, sparsity):
    sparsity_dict = {}
    for name in names:
        sparsity_dict[name] = sparsity
    return sparsity_dict


def get_mask_lamp(layer, sparsity):
    weight_orig = layer.weight
    weight_shape = weight_orig.shape

    normalizer = weight_orig.norm() ** 2
    sorted_weight, sorted_idx = (
        weight_orig.abs().view(-1).sort(descending=False)
    )

    weight_cumsum_temp = (sorted_weight ** 2).cumsum(dim=0)
    weight_cumsum = torch.zeros(weight_cumsum_temp.shape)
    weight_cumsum[1:] = weight_cumsum_temp[: len(weight_cumsum_temp) - 1]
    weight_cumsum = weight_cumsum.to(device)

    sorted_weight /= (normalizer - weight_cumsum).sqrt()
    sorted_weight = sorted_weight.to(not_device)

    score = torch.zeros(weight_cumsum.shape)
    score[sorted_idx] = sorted_weight

    score = score.view(weight_shape)

    score = score.view(-1)
    density = 1.0 - sparsity
    topk_val = int(np.ceil(density * score.numel()))
    topk_scores, indices = torch.topk(score, topk_val)
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask = mask.scatter(0, indices, True)
    mask = mask.view(weight_shape)

    return mask


def get_mask_topk(layer, sparsity):
    weight_orig = layer.weight
    weight_shape = weight_orig.shape

    weight_orig = weight_orig.abs().view(-1)
    density = 1.0 - sparsity
    topk_val = int(np.ceil(density * weight_orig.numel()))
    topk_scores, indices = torch.topk(weight_orig, topk_val)
    mask = torch.zeros_like(weight_orig, dtype=torch.bool)
    mask = mask.scatter(0, indices, True)
    mask = mask.view(weight_shape)

    return mask


def get_global_mask_topk(layers, sparsity):
    # Get all orig parameters
    weight_orig = [layer.weight for layer in layers]
    # Get all shapes
    weight_shape = [weight.shape for weight in weight_orig]
    # Get orig parameters flattened
    weight_orig_flatten = [weight.abs().view(-1) for weight in weight_orig]
    aggregate_weight_orig_flatten = torch.cat(weight_orig_flatten)
    aggregate_global_mask_flatten = torch.zeros_like(
        aggregate_weight_orig_flatten, dtype=torch.bool
    )
    # Get orig params count.
    weight_orig_params = [weight.numel() for weight in weight_orig_flatten]
    # Get masks of shape orig params
    masks_flatten = [
        torch.zeros_like(weight, dtype=torch.bool)
        for weight in weight_orig_flatten
    ]
    density = 1.0 - sparsity
    topk_val = int(np.ceil(density * len(aggregate_weight_orig_flatten)))
    topk_scores, indices = torch.topk(aggregate_weight_orig_flatten, topk_val)
    aggregate_global_mask_flatten = aggregate_global_mask_flatten.scatter(
        0, indices, True
    )
    pointer = 0
    masks = []
    for i, num_params in enumerate(weight_orig_params):
        _mask = aggregate_global_mask_flatten[
            pointer : pointer + num_params
        ].view_as(weight_orig[i])
        masks.append(_mask)
        pointer += num_params

    return masks
