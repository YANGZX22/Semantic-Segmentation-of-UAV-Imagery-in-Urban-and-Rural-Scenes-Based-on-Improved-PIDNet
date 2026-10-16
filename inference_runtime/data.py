"""Read frozen list membership and the exact inference transforms."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .common import read_json


def split_root(data_root, dataset, fold):
    return Path(data_root) / (f"caihua/rs{fold}" if dataset == "caihua" else "udd")


def normalization(manifest, dataset, name):
    stats = manifest["training_statistics"]
    if dataset == "udd" and name.startswith("pidnet_"):
        return dict(mean=stats["mean_rgb"], std=stats["std_rgb"], source="fold training pixels")
    return dict(manifest["training_statistics"]["normalization"])


class FrozenEvaluationSet(Dataset):
    def __init__(self, data_root, dataset, fold, role, name):
        self.root = split_root(data_root, dataset, fold)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = m = read_json(self.manifest_path)
        self.normalization = normalization(m, dataset, name)
        self.mapping = dict(enumerate(range(8))) if dataset == "caihua" else {108: 0, 119: 1, 90: 2, 16: 3, 0: 4}
        data_root = Path(data_root).resolve()
        self.rows = []
        expected = {}
        for sample in m["samples"]:
            if sample["split"] != role:
                continue
            pair = []
            for key in ("image", "label"):
                parts = Path(sample[key]).parts
                if parts[0] != "datasets_clean":
                    raise ValueError("Unknown source root in manifest")
                pair.append(data_root.joinpath(*parts[1:]).resolve())
            expected[tuple(pair)] = sample
        seen = set()
        for line in (self.root / m["lists"][role]["path"]).read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                raise ValueError("Evaluation requires image and label columns")
            pair = tuple((self.root / p).resolve() for p in parts)
            if pair not in expected or pair in seen or not all(p.is_relative_to(data_root) for p in pair):
                raise ValueError("List membership does not match the frozen partition")
            seen.add(pair)
            self.rows.append((pair, expected[pair]))
        if seen != set(expected) or not seen:
            raise ValueError("Incomplete or empty evaluation list")
        self.record = dict(split=role, split_id=m["split_id"], samples=len(self.rows),
                           normalization=self.normalization, input="RGB / 255, then (x-mean)/std; no augmentation",
                           mapping=self.mapping)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        pair, _ = self.rows[index]
        decoded = []
        for path, flag in zip(pair, (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE)):
            content = path.read_bytes()
            value = cv2.imdecode(np.frombuffer(content, np.uint8), flag)
            if value is None or value.shape[:2] != (512, 512):
                raise ValueError("Expected a decodable 512x512 tile: " + str(path))
            decoded.append(value)
        image, raw = decoded
        image = image.astype(np.float32)[:, :, ::-1] / 255.0
        image -= self.normalization["mean"]
        image /= self.normalization["std"]
        label = np.full(raw.shape, 255, np.uint8)
        for gray, target in self.mapping.items():
            label[raw == gray] = target
        return image.transpose(2, 0, 1).copy(), label.copy()
