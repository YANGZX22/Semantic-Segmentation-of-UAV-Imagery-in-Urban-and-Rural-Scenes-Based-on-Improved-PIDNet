"""Load the existing architectures, without training or implicit downloads."""
import importlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("unet", "deeplabv3plus", "hrnet", "pspnet", "bisenet", "pidnet_m",
          "pidnet_m_ca", "pidnet_m_ca_dsc", "pidnet_m_ca_dcu", "pidnet_m_ca_dsc_dcu")
PID_MODULES = dict(zip(MODELS[5:], ("pidnet", "pidnet_CA", "pidnet_CA_DSC",
                                   "pidnet_CA_DCU", "pidnet_CA_DSC_DCU")))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
                          + "\n", encoding="utf-8")


def environment():
    import torch
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), cpu_threads=torch.get_num_threads(),
                cpu_interop_threads=torch.get_num_interop_threads())


def activate(dataset):
    """The legacy packages use global imports; never mix datasets in one process."""
    path = ROOT / "code" / dataset
    for name in ("models", "configs", "experiment_runtime"):
        mod = sys.modules.get(name)
        if mod is not None and not Path(mod.__file__).resolve().is_relative_to(path):
            raise RuntimeError("Use a fresh process for each dataset")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def configuration(dataset, name, fold):
    activate(dataset)
    if dataset == "caihua":
        from configs import config
        cfg = config.clone()
        cfg.defrost()
        internal = "hrnet_os8" if name == "hrnet" else name
        cfg.merge_from_file(str(ROOT / "configs/caihua" / internal / f"rs{fold}.yaml"))
        cfg.freeze()
        return cfg
    from experiment_runtime.support import config_for
    return config_for(name)


def build_model(dataset, name, cfg):
    if dataset == "caihua":
        from experiment_runtime.registry import build_model as build
        return build(cfg)[0]
    module = importlib.import_module("models." + PID_MODULES.get(
        name, "seg_hrnet_OS8" if name == "hrnet" else name))
    if name.startswith("pidnet_"):
        return module.PIDNet(m=2, n=3, num_classes=5, planes=64,
                             ppm_planes=96, head_planes=128, augment=True)
    if name == "hrnet":
        model = module.HighResolutionNet(cfg)
        model.init_weights("")
        return model
    return module.get_seg_model(cfg, imgnet_pretrained=False)


def checkpoint_state(checkpoint):
    """Accept raw weights, Caihua packages, and UDD FullModel checkpoints."""
    import torch
    state = checkpoint
    while isinstance(state, dict):
        nested = next((state[k] for k in ("state_dict", "model")
                       if isinstance(state.get(k), dict)), None)
        if nested is None:
            break
        state = nested
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing checkpoint tensor state")
    result, ignored = {}, []
    # UDD resume weights also contain criterion buffers. Select the explicit
    # FullModel.model namespace, allowing only the two known loss namespaces.
    unwrapped = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Non-tensor entry in checkpoint state")
        while key.startswith("module."):
            key = key[7:]
        if key in unwrapped:
            raise ValueError("Duplicate key after DataParallel prefix removal: " + key)
        unwrapped[key] = value
    wrapper = any(k.startswith("model.") for k in unwrapped)
    for key, value in unwrapped.items():
        if wrapper and key.startswith(("sem_loss.", "bd_loss.")):
            ignored.append(key)
            continue
        if wrapper and not key.startswith("model."):
            raise ValueError("Unexpected FullModel checkpoint key: " + key)
        while key.startswith(("module.", "model.")):
            key = key.split(".", 1)[1]
        if key in result:
            raise ValueError("Duplicate checkpoint key: " + key)
        result[key] = value
    return result, ignored


def load_weights(model, path, dataset, name, cfg, manifest_path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state dictionary or checkpoint dictionary")
    contract = checkpoint.get("experiment_protocol", {})
    identity = contract.get("model", {})
    expected = "hrnet_os8" if dataset == "caihua" and name == "hrnet" else name
    if identity and identity.get("name") != expected:
        raise ValueError("Checkpoint model identity differs from --model")
    inference = contract.get("inference", {})
    for key, value in {"num_classes": cfg.DATASET.NUM_CLASSES,
                       "align_corners": cfg.MODEL.ALIGN_CORNERS,
                       "ignore_label": cfg.TRAIN.IGNORE_LABEL}.items():
        if key in inference and inference[key] != value:
            raise ValueError("Checkpoint inference contract differs: " + key)
    stamps = [s for s in (checkpoint.get("data_protocol"), contract.get("data")) if s]
    manifest = read_json(manifest_path)
    for stamp in stamps:
        if (stamp.get("dataset"), stamp.get("split_id")) != (manifest["dataset"], manifest["split_id"]):
            raise ValueError("Checkpoint belongs to a different split/fold")
    state, ignored = checkpoint_state(checkpoint)
    # Check exact key coverage too: PyTorch otherwise tolerates missing old BN counters.
    expected_keys = set(model.state_dict())
    if set(state) != expected_keys:
        raise ValueError(f"Checkpoint architecture mismatch: missing {sorted(expected_keys-set(state))[:10]}, "
                         f"unexpected {sorted(set(state)-expected_keys)[:10]}")
    model.load_state_dict(state, strict=True)
    return dict(path=str(Path(path).resolve()), strict_state_load=True,
                epoch=checkpoint.get("epoch"), ignored_training_loss_buffers=ignored,
                embedded_model_identity_checked=bool(identity),
                embedded_split_identity_checked=bool(stamps),
                identity_note="Raw weights have no embedded split/seed identity; the supplied model/fold is used.")


def logits(model, x, name, align_corners=True):
    import torch
    from torch.nn import functional as F
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[1 if name.startswith("pidnet_") else 0]
    if not isinstance(y, torch.Tensor) or y.ndim != 4:
        raise ValueError("Expected NCHW semantic logits")
    if y.shape[-2:] != x.shape[-2:]:
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=align_corners)
    return y


def strip_training_heads(model, name):
    # Same deployment path as the historical profiler. Only this in-memory copy
    # is changed; the loaded checkpoint and architecture source are never edited.
    if name.startswith("pidnet_"):
        model.augment = False
        names = ("seghead_p", "seghead_d")
    elif name == "pspnet":
        model.augment = model.aux_branch = False
        names = ("auxiliary_branch",)
    elif name == "bisenet":
        model.augment = False
        names = ("conv_out16",)
    else:
        names = ()
    removed = {}
    for key in names:
        removed[key] = sum(p.numel() for p in getattr(model, key).parameters())
        delattr(model, key)
    return removed
