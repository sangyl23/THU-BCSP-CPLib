import torch
from thop import profile


def _unwrap_model(model):
    # DataParallel(model adapter(model)) -> original forecasting model.
    while isinstance(model, torch.nn.DataParallel):
        model = model.module
    if model.__class__.__name__ == '_DataParallelOutputAdapter':
        model = model.model
    return model


def _count_gpt2_conv1d(module, inputs, output):
    # inputs[0]: (B, token_num, input_features).
    input_features, output_features = module.weight.shape
    operations = inputs[0].numel() // input_features * input_features * output_features
    _add_profile_ops(module, operations)


def _count_gpt2_attention(module, inputs, output):
    # inputs[0]: (B, token_num, hidden_size).
    batch_size, sequence_length, hidden_size = inputs[0].shape
    operations = 2 * batch_size * sequence_length * sequence_length * hidden_size
    _add_profile_ops(module, operations)


def _add_profile_ops(module, operations):
    current_ops = module.total_ops
    if torch.is_tensor(current_ops):
        increment = torch.as_tensor(
            [operations], device=current_ops.device, dtype=current_ops.dtype
        )
        module.total_ops = current_ops + increment
    else:
        module.total_ops += operations


def _gpt2_profile_ops(model):
    # THOP does not provide complete GPT-2 Conv1D/attention counters by default.
    gpt2 = getattr(model, 'gpt2', None)
    if gpt2 is None:
        return {}

    from transformers.pytorch_utils import Conv1D

    custom_ops = {Conv1D: _count_gpt2_conv1d}
    custom_ops.update({
        type(block.attn): _count_gpt2_attention for block in gpt2.h
    })
    return custom_ops


def model_statistics(model, inputs, batch_size):
    # Count the original model instead of the outer DataParallel wrapper.
    model = _unwrap_model(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    gpt2 = getattr(model, 'gpt2', None)
    gpt2_parameters = (
        sum(parameter.numel() for parameter in gpt2.parameters())
        if gpt2 is not None else 0
    )
    gpt2_parameter_ids = (
        {id(parameter) for parameter in gpt2.parameters()}
        if gpt2 is not None else set()
    )
    chronos_model = getattr(model, 'chronos_model', None)
    chronos_parameters = (
        sum(parameter.numel() for parameter in chronos_model.parameters())
        if chronos_model is not None else None
    )
    non_gpt_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in gpt2_parameter_ids
    )

    # Each tensor in inputs has batch dimension first, e.g. x_enc: (B, seq_len, c_in).
    # Profile one item, then restore the channel expansion introduced by preprocessing.
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    first_batched_input = next(
        value for value in inputs
        if torch.is_tensor(value) and value.ndim > 0
    )
    transformed_batch_size = first_batched_input.shape[0]
    channel_sample_multiplier = transformed_batch_size / batch_size
    profile_inputs = tuple(
        value[:1]
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == transformed_batch_size
        )
        else value
        for value in inputs
    )
    profiling_buffer_names = ('total_ops', 'total_params')
    existing_profiling_buffers = {
        (id(module), name)
        for module in model.modules()
        for name in profiling_buffer_names
        if name in module._buffers
    }
    try:
        macs, _ = profile(
            model,
            inputs=profile_inputs,
            custom_ops=_gpt2_profile_ops(model),
            verbose=False,
        )
    finally:
        # THOP can leave CPU bookkeeping buffers on modules without counting hooks.
        # Such buffers make the next DataParallel forward fail its device check.
        for module in model.modules():
            for name in profiling_buffer_names:
                if (
                    (id(module), name) not in existing_profiling_buffers
                    and name in module._buffers
                ):
                    module._buffers.pop(name)

    # One multiply-accumulate operation is reported as two floating-point operations.
    return {
        'total_parameters': total_parameters,
        'trainable_parameters': trainable_parameters,
        'gpt2_parameters': gpt2_parameters,
        'chronos_parameters': chronos_parameters,
        'non_gpt_trainable_parameters': non_gpt_trainable_parameters,
        'mflops_per_sample': 2 * macs * channel_sample_multiplier / 1e6,
    }


def print_model_statistics(model, inputs, batch_size):
    # Parameter counts are printed in K and forward complexity in MFLOPs/sample.
    statistics = model_statistics(model, inputs, batch_size)
    print(f"Total parameters: {statistics['total_parameters'] / 1e3:.5f} K")
    print(f"Trainable parameters: {statistics['trainable_parameters'] / 1e3:.5f} K")
    print(f"GPT-2 parameters: {statistics['gpt2_parameters'] / 1e3:.5f} K")
    if statistics['chronos_parameters'] is not None:
        print(
            'Chronos-2 parameters: '
            f"{statistics['chronos_parameters'] / 1e3:.5f} K"
        )
    print(
        'Non-GPT trainable parameters: '
        f"{statistics['non_gpt_trainable_parameters'] / 1e3:.5f} K"
    )
    print(
        f"FLOPs: {statistics['mflops_per_sample']:.5f} MFLOPs/sample "
        '(2 FLOPs/MAC)'
    )
