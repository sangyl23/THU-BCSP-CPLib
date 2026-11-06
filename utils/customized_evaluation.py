import torch


def _flatten_test_specs(specs):
    # flatten two-dimensional test specifications in speed-major order
    return [spec for speed_specs in specs for spec in speed_specs]


def _condition_indices(speed_norm, time_interval_norm, specs):
    # normalized conditions are one-based indices of the two test_specs dimensions
    speed_indices = torch.round(speed_norm).to(torch.long) - 1
    interval_indices = torch.round(time_interval_norm).to(torch.long) - 1

    # reject non-integer condition labels instead of assigning them to a nearby category
    if not torch.allclose(speed_norm, speed_indices.to(speed_norm.dtype) + 1):
        raise ValueError("speed_norm must contain one-based integer indices")
    if not torch.allclose(time_interval_norm, interval_indices.to(time_interval_norm.dtype) + 1):
        raise ValueError("time_interval_norm must contain one-based integer indices")

    # row lengths and offsets also support different interval counts for different speeds
    row_lengths = torch.tensor(
        [len(speed_specs) for speed_specs in specs],
        device=speed_indices.device,
        dtype=torch.long,
    )
    if torch.any(speed_indices < 0) or torch.any(speed_indices >= len(specs)):
        raise ValueError("speed_norm contains an index outside test_specs")
    if torch.any(interval_indices < 0) or torch.any(
        interval_indices >= row_lengths[speed_indices]
    ):
        raise ValueError("time_interval_norm contains an index outside test_specs")

    # convert test_specs[x - 1][y - 1] to its index in the flattened specification list
    row_offsets = torch.cat([
        torch.zeros(1, device=speed_indices.device, dtype=torch.long),
        torch.cumsum(row_lengths, dim=0)[:-1],
    ])
    return row_offsets[speed_indices] + interval_indices


def _to_complex_channel(channel):
    # convert a real-imaginary pair in the last dimension to a complex tensor
    if torch.is_complex(channel):
        return channel
    if channel.ndim >= 1 and channel.shape[-1] == 2:
        return torch.complex(channel[..., 0], channel[..., 1])
    raise TypeError("Expected a complex channel tensor or a real/imaginary pair")


def _to_real_channel(channel):
    # convert a complex tensor to a real-imaginary pair in the last dimension
    if torch.is_complex(channel):
        return torch.stack([channel.real, channel.imag], dim=-1)
    if channel.ndim >= 1 and channel.shape[-1] == 2:
        return channel
    raise TypeError("Expected a complex channel tensor or a real/imaginary pair")


def _print_test_summary(
    nmse_eval_epoch,
    specs,
    print_specs,
    summary_spec=('60kmh', '0p5en3'),
):
    # locate print_specs in the full flattened list while retaining its configured order
    flat_specs = _flatten_test_specs(specs)
    flat_print_specs = _flatten_test_specs(print_specs)
    mean_by_set = nmse_eval_epoch.mean(
        axis=tuple(range(nmse_eval_epoch.ndim - 1))
    )
    print_indices = []
    for spec in flat_print_specs:
        if spec not in flat_specs:
            raise ValueError(f"print_specs entry is not present in test_specs: {spec}")
        print_indices.append(flat_specs.index(spec))

    if summary_spec not in flat_specs:
        raise ValueError(f"summary_spec entry is not present in test_specs: {summary_spec}")
    summary_index = flat_specs.index(summary_spec)

    # print only the conditions selected by print_specs
    print('Test-set NMSE: ' + ' '.join(
        f'v={speed},interval={interval} {mean_by_set[index]:.6g}'
        for (speed, interval), index in zip(flat_print_specs, print_indices)
    ))

    # retain the prediction-step dimension and average any run/epoch dimensions
    summary_nmse = nmse_eval_epoch[..., summary_index]
    if summary_nmse.ndim > 1:
        summary_nmse = summary_nmse.mean(axis=tuple(range(1, summary_nmse.ndim)))
    speed, interval = summary_spec
    print(
        f'Prediction-step NMSE for v={speed},interval={interval}: '
        + ' '.join(f'{value:.6g}' for value in summary_nmse)
    )

    # average the NMSE equally over all categories configured in test_specs
    print(f'All {len(flat_specs)} test sets NMSE mean={mean_by_set.mean():.6g}')


def print_model_structure(model, indent=0):
    # recursively print each submodule with indentation representing its depth
    for name, module in model.named_children():
        print(" " * indent + f"{name}: {module.__class__.__name__}")
        print_model_structure(module, indent + 2)
