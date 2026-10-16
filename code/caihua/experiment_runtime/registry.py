"""Exact model identities and output contracts; never guess from substrings."""
from dataclasses import asdict, dataclass
import importlib
import math
import re


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module: str
    family: str
    size: str = ""
    semantic_outputs: int = 1
    main_index: int = 0
    boundary_index: int = -1

    def record(self):
        return asdict(self)


PIDNET_VARIANTS = {
    "": "pidnet", "ca": "pidnet_CA", "ca_dsc": "pidnet_CA_DSC",
    "ca_dcu": "pidnet_CA_DCU", "ca_dsc_dcu": "pidnet_CA_DSC_DCU",
}
OTHER_MODELS = {
    "bisenet": ("bisenet", 2), "pspnet": ("pspnet", 2),
    "deeplabv3plus": ("deeplabv3plus", 1), "unet": ("unet", 1),
    "hrnet_os8": ("seg_hrnet_OS8", 1, "hrnet"),
}


def resolve_model(name):
    name = str(name).lower()
    match = re.fullmatch(r"pidnet_(m)(?:_(.+))?", name)
    if match and (match.group(2) or "") in PIDNET_VARIANTS:
        module = PIDNET_VARIANTS[match.group(2) or ""]
        return ModelSpec(name, module, "pidnet", match.group(1), 2, 1, 2)
    if name in OTHER_MODELS:
        entry = OTHER_MODELS[name]
        module, outputs = entry[0], entry[1]
        family = entry[2] if len(entry) > 2 else name
        return ModelSpec(name, module, family, semantic_outputs=outputs)
    raise ValueError("Unregistered model name: %s. No fallback model is permitted." % name)


def validate_contract(cfg):
    spec = resolve_model(cfg.MODEL.NAME)
    if cfg.MODEL.NUM_OUTPUTS != spec.semantic_outputs:
        raise ValueError("MODEL.NUM_OUTPUTS disagrees with the registered model")
    if cfg.TEST.OUTPUT_INDEX != spec.main_index:
        raise ValueError("TEST.OUTPUT_INDEX must select the registered main semantic head")
    weights = list(cfg.LOSS.BALANCE_WEIGHTS)
    if len(weights) != spec.semantic_outputs or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("One nonnegative loss weight is required per semantic output")
    if weights[spec.main_index] <= 0:
        raise ValueError("The main semantic output must receive supervision")
    if spec.boundary_index < 0 and cfg.LOSS.SB_WEIGHTS != 0:
        raise ValueError("Semantic-boundary supervision requires a PIDNet boundary head")
    return spec


def build_model(cfg):
    """Construct without any implicit download or pretrained-loading side effect."""
    spec = validate_contract(cfg)
    module = importlib.import_module("models." + spec.module)
    classes = cfg.DATASET.NUM_CLASSES
    if spec.family == "pidnet":
        # Searching for 's' in a name containing 'dsc' incorrectly selected PIDNet-S.
        dims = (2, 3, 64, 96, 128)
        m, n, planes, ppm, head = dims
        model = module.PIDNet(m=m, n=n, num_classes=classes, planes=planes,
                             ppm_planes=ppm, head_planes=head, augment=True)
    elif spec.family == "bisenet":
        model = module.BiSeNetV1(n_classes=classes, augment=True, pretrained=False)
    elif spec.family == "pspnet":
        model = module.PSPNet(num_classes=classes, downsample_factor=16,
                              augment=True, aux_branch=True, pretrained=False)
    elif spec.family == "deeplabv3plus":
        model = module.DeepLabV3Plus(num_classes=classes, downsample_factor=16,
                                     augment=True, pretrained=False)
    elif spec.family == "hrnet":
        model = module.HighResolutionNet(cfg)
        model.init_weights("")
    elif spec.family == "unet":
        model = module.UNet(n_channels=3, n_classes=classes, bilinear=False)
    else:
        raise AssertionError(spec.family)
    return model, spec


def semantic_outputs(outputs, spec, training):
    """Return semantic heads and optional boundary head in a checked order."""
    values = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    expected = spec.semantic_outputs + int(spec.boundary_index >= 0)
    # BiSeNet and PSPNet omit auxiliary outputs in eval mode.
    if not training and len(values) == 1 and spec.family != "pidnet":
        semantic, main, boundary = values, 0, None
    elif len(values) == expected:
        semantic = values[:spec.semantic_outputs]
        main = spec.main_index
        boundary = values[spec.boundary_index] if spec.boundary_index >= 0 else None
    else:
        raise ValueError("%s returned %d outputs; expected %d" % (spec.name, len(values), expected))
    if any(value is None or value.ndim != 4 for value in semantic):
        raise ValueError("Semantic outputs must be NCHW tensors")
    if boundary is not None and (boundary.ndim != 4 or boundary.shape[1] != 1):
        raise ValueError("The PIDNet boundary output must have one channel")
    return semantic, main, boundary
