"""Auditable initialization and exact trained-checkpoint loading."""
from pathlib import Path


def read_checkpoint(path):
    import torch
    # No fallback to arbitrary pickle execution or silent random initialization.
    return torch.load(str(path), map_location="cpu", weights_only=True)


def tensor_state(checkpoint):
    import torch
    state = checkpoint
    while isinstance(state, dict):
        nested = next((state[key] for key in ("state_dict", "model")
                       if key in state and isinstance(state[key], dict)), None)
        if nested is None:
            break
        state = nested
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain a nonempty tensor state")
    result = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint state contains a non-tensor entry")
        name = key
        while name.startswith(("module.", "model.")):
            name = name.split(".", 1)[1]
        if name in result:
            raise ValueError("Duplicate checkpoint key after prefix removal: " + name)
        result[name] = value
    return result


def initialize(model, spec, cfg):
    policy = cfg.MODEL.INITIALIZATION
    if policy == "random":
        if cfg.MODEL.PRETRAINED:
            raise ValueError("Random initialization cannot have a pretrained path")
        return {"policy": "random", "loaded_keys": []}
    path = Path(cfg.MODEL.PRETRAINED)
    if not path.is_file():
        raise FileNotFoundError("Required pretrained file is missing: " + str(path))
    checkpoint = read_checkpoint(path)
    if isinstance(checkpoint, dict) and any(k in checkpoint for k in ("data_protocol", "experiment_protocol", "optimizer")):
        raise ValueError("A trained experiment checkpoint cannot be used as external initialization")
    source = tensor_state(checkpoint)
    family = spec.family
    if family in ("deeplabv3plus", "pspnet"):
        target, prefix = model.backbone, "backbone."
    elif family == "bisenet":
        target, prefix = model.cp.resnet, "cp.resnet."
    else:
        target, prefix = model, ""
    # Support a documented backbone wrapper, without suffix/fuzzy matching.
    if prefix and any(key.startswith(prefix) for key in source):
        source = {key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)}
    target_state = target.state_dict()
    excluded = ("final_layer.", "seghead_p.", "seghead_d.") if family == "pidnet" else (
        ("last_layer.",) if family == "hrnet" else ())
    eligible = {key: value for key, value in target_state.items() if not key.startswith(excluded)}
    matched = {key: source[key] for key, value in eligible.items()
               if key in source and source[key].shape == value.shape}
    missing = sorted(key for key in eligible if key not in source)
    mismatched = sorted(key for key in eligible if key in source and source[key].shape != eligible[key].shape)
    if family == "pidnet":
        # Altered modules cannot consume the original kernels. All unchanged stem
        # and early residual tensors are mandatory; every other omission is logged.
        required = [key for key in eligible if key.startswith(("conv1.", "layer1.", "layer2."))]
        load_policy = "pidnet_compatible_backbone_with_required_stem_v1"
    else:
        required = list(eligible)
        load_policy = "complete_backbone_v1"
    required = [key for key in required if not key.endswith("num_batches_tracked")]
    failed = [key for key in required if key not in matched]
    if not matched or failed:
        raise ValueError("Pretrained backbone mismatch (%s): %s" % (spec.name, failed[:15]))
    target.load_state_dict(matched, strict=False)
    parameters = dict(target.named_parameters())
    eligible_params = {key: value for key, value in parameters.items() if key in eligible}
    loaded_numel = sum(value.numel() for key, value in eligible_params.items() if key in matched)
    total_numel = sum(value.numel() for value in eligible_params.values())
    return {
        "policy": "external", "load_policy": load_policy, "source_path": str(path),
        "target_prefix": prefix,
        "loaded_keys": sorted(matched), "missing_keys": missing,
        "shape_mismatches": mismatched, "excluded_task_heads": sorted(set(target_state) - set(eligible)),
        "unused_source_keys": sorted(set(source) - set(matched)),
        "loaded_parameter_elements": loaded_numel, "eligible_parameter_elements": total_numel,
        "loaded_parameter_fraction": loaded_numel / total_numel if total_numel else 0.0,
        "source_note": "Project-supplied external pretraining.",
    }


def load_trained_state(model, checkpoint):
    if not isinstance(checkpoint, dict) or "experiment_protocol" not in checkpoint:
        raise ValueError("Trained checkpoint lacks an embedded comparison protocol")
    # Full architecture equality is required for evaluation and resume.
    model.load_state_dict(tensor_state(checkpoint["state_dict"]), strict=True)
