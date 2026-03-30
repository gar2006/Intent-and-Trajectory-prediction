from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

MANUAL_MINI_VAL_SCENES = {
    "scene-0553",  # crossing-heavy crosswalk scene
    "scene-0916",  # waiting-heavy scene
    "scene-1100",  # night scene with more stationary VRUs
}
EXCLUDED_MINI_SCENES = {
    "scene-1077",  # contains no pedestrian annotations
}
CYCLIST_CATEGORIES = {
    "vehicle.bicycle",
    "vehicle.motorcycle",
}
AGENT_TYPE_TO_ID = {
    "pedestrian": 0,
    "cyclist": 1,
}


@dataclass(frozen=True)
class TrajectorySample:
    agent_id: str
    agent_type: str
    sample_token: str
    map_token: str
    origin_xy: np.ndarray
    history_xy: np.ndarray
    future_xy: np.ndarray
    history_features: np.ndarray
    neighbor_features: np.ndarray
    neighbor_mask: np.ndarray
    intent_label: int


class MapPatchExtractor:
    """
    Lightweight map raster extractor using the semantic prior PNGs shipped with nuScenes.

    nuScenes map images are georeferenced in the official devkit. To keep Phase 2
    runnable without that dependency, this extractor estimates a linear mapping
    from global annotation coordinates to image pixels using the observed
    coordinate range for each map in the dataset split.
    """

    def __init__(
        self,
        data_root: Path,
        map_records: List[dict],
        coord_ranges: Dict[str, dict],
        patch_size: int = 100,
        map_padding_m: float = 20.0,
    ) -> None:
        self.data_root = data_root
        self.patch_size = patch_size
        self.coord_ranges = coord_ranges
        self.map_padding_m = map_padding_m
        self.map_meta = {
            row["token"]: {
                "path": self.data_root / row["filename"],
            }
            for row in map_records
        }
        self.image_cache: Dict[str, np.ndarray] = {}

    def extract_patch(self, map_token: str, x: float, y: float) -> np.ndarray:
        image = self._load_image(map_token)
        h, w = image.shape
        ranges = self.coord_ranges[map_token]

        x_min = ranges["x_min"] - self.map_padding_m
        x_max = ranges["x_max"] + self.map_padding_m
        y_min = ranges["y_min"] - self.map_padding_m
        y_max = ranges["y_max"] + self.map_padding_m

        px = (x - x_min) / max(x_max - x_min, 1e-6) * (w - 1)
        py = (1.0 - (y - y_min) / max(y_max - y_min, 1e-6)) * (h - 1)

        half = self.patch_size // 2
        px = int(round(px))
        py = int(round(py))

        x0 = px - half
        x1 = px + half
        y0 = py - half
        y1 = py + half

        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        src_x0 = max(0, x0)
        src_x1 = min(w, x1)
        src_y0 = max(0, y0)
        src_y1 = min(h, y1)

        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        if src_x1 > src_x0 and src_y1 > src_y0:
            patch[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

        return patch[None, ...]

    def _load_image(self, map_token: str) -> np.ndarray:
        cached = self.image_cache.get(map_token)
        if cached is not None:
            return cached

        image_path = self.map_meta[map_token]["path"]
        image = Image.open(image_path).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        self.image_cache[map_token] = array
        return array


class NuScenesPedestrianDataset(Dataset):
    """
    Minimal nuScenes vulnerable-road-user trajectory dataset.

    This loader reads the raw JSON tables directly from the downloaded nuScenes
    folder, extracts pedestrian/cyclist tracks with enough history and future context,
    and returns:
    - history_features: [past_steps, 6] = (x, y, vx, vy, ax, ay)
    - future_xy: [future_steps, 2] = future positions in the agent-local frame
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        past_steps: int = 4,
        future_steps: int = 12,
        dt: float = 0.5,
        val_ratio: float = 0.2,
        split_seed: int = 42,
        cache_dir: Optional[str | Path] = None,
        use_cache: bool = True,
        include_map: bool = False,
        map_patch_size: int = 100,
        include_social: bool = False,
        social_radius: float = 10.0,
        max_neighbors: int = 8,
        include_intent: bool = False,
        heading_aligned: bool = False,
        include_cyclists: bool = True,
        use_manual_mini_split: bool = True,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.data_root = Path(data_root)
        self.table_root = self._resolve_table_root(self.data_root)
        self.split = split
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.dt = dt
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.include_map = include_map
        self.map_patch_size = map_patch_size
        self.include_social = include_social
        self.social_radius = social_radius
        self.max_neighbors = max_neighbors
        self.include_intent = include_intent
        self.heading_aligned = heading_aligned
        self.include_cyclists = include_cyclists
        self.use_manual_mini_split = use_manual_mini_split
        self.map_extractor: Optional[MapPatchExtractor] = None

        cache_base = Path(cache_dir) if cache_dir is not None else Path("cache")
        self.cache_dir = cache_base
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = (
            self.cache_dir
            / (
                f"nuscenes_{self.table_root.name}_{split}_p{past_steps}_f{future_steps}"
                f"_map{int(include_map)}_social{int(include_social)}"
                f"_intent{int(include_intent)}"
                f"_iv4"
                f"_ha{int(heading_aligned)}"
                f"_cy{int(include_cyclists)}"
                f"_ms{int(use_manual_mini_split)}"
                f"_r{int(social_radius * 10)}_n{max_neighbors}_seed{split_seed}.pkl"
            )
        )

        if use_cache and self.cache_path.exists():
            with self.cache_path.open("rb") as fp:
                cached_payload = pickle.load(fp)
            if isinstance(cached_payload, dict):
                self.samples = cached_payload["samples"]
                coord_ranges = cached_payload.get("coord_ranges")
            else:
                self.samples = cached_payload
                coord_ranges = None
        else:
            self.samples, coord_ranges = self._build_samples()
            if use_cache:
                with self.cache_path.open("wb") as fp:
                    pickle.dump(
                        {
                            "samples": self.samples,
                            "coord_ranges": coord_ranges,
                        },
                        fp,
                    )

        if self.include_map and coord_ranges is None:
            self.samples, coord_ranges = self._build_samples()
            if use_cache:
                with self.cache_path.open("wb") as fp:
                    pickle.dump(
                        {
                            "samples": self.samples,
                            "coord_ranges": coord_ranges,
                        },
                        fp,
                    )

        if self.include_social and self.samples and not hasattr(self.samples[0], "neighbor_features"):
            self.samples, coord_ranges = self._build_samples()
            if use_cache:
                with self.cache_path.open("wb") as fp:
                    pickle.dump(
                        {
                            "samples": self.samples,
                            "coord_ranges": coord_ranges,
                        },
                        fp,
                    )

        if self.include_intent and self.samples and not hasattr(self.samples[0], "intent_label"):
            self.samples, coord_ranges = self._build_samples()
            if use_cache:
                with self.cache_path.open("wb") as fp:
                    pickle.dump(
                        {
                            "samples": self.samples,
                            "coord_ranges": coord_ranges,
                        },
                        fp,
                    )

        if not self.samples:
            raise RuntimeError(
                "No usable pedestrian trajectory samples were found. "
                "Check that the dataset path is correct and contains the nuScenes tables."
            )

        if self.include_map:
            map_records = self._load_table("map")
            if coord_ranges is None:
                raise RuntimeError("Map patches requested, but map coordinate ranges were not built.")
            self.map_extractor = MapPatchExtractor(
                data_root=self.data_root,
                map_records=map_records,
                coord_ranges=coord_ranges,
                patch_size=self.map_patch_size,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        item = {
            "history": torch.from_numpy(sample.history_features).float(),
            "future": torch.from_numpy(sample.future_xy).float(),
            "history_xy": torch.from_numpy(sample.history_xy).float(),
            "sample_token": sample.sample_token,
            "agent_id": sample.agent_id,
            "agent_type": sample.agent_type,
            "agent_type_id": torch.tensor(
                AGENT_TYPE_TO_ID.get(sample.agent_type, 0),
                dtype=torch.long,
            ),
        }
        if self.include_map and self.map_extractor is not None:
            global_x = float(sample.origin_xy[0])
            global_y = float(sample.origin_xy[1])
            patch = self.map_extractor.extract_patch(sample.map_token, global_x, global_y)
            item["map_patch"] = torch.from_numpy(patch).float()
        if self.include_social:
            item["neighbors"] = torch.from_numpy(sample.neighbor_features).float()
            item["neighbor_mask"] = torch.from_numpy(sample.neighbor_mask).bool()
        if self.include_intent:
            item["intent"] = torch.tensor(sample.intent_label, dtype=torch.long)
        return item

    @staticmethod
    def _resolve_table_root(data_root: Path) -> Path:
        direct = data_root / data_root.name
        if direct.exists():
            return direct

        nested = data_root / "v1.0-mini"
        if nested.exists():
            return nested

        raise FileNotFoundError(
            f"Could not find nuScenes JSON tables inside {data_root}. "
            "Expected a nested 'v1.0-mini' directory."
        )

    def _load_table(self, name: str) -> List[dict]:
        path = self.table_root / f"{name}.json"
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _build_samples(self) -> tuple[List[TrajectorySample], Dict[str, dict]]:
        scenes = self._load_table("scene")
        samples = self._load_table("sample")
        annotations = self._load_table("sample_annotation")
        instances = self._load_table("instance")
        categories = self._load_table("category")
        maps = self._load_table("map")

        ann_by_token = {row["token"]: row for row in annotations}
        instance_by_token = {row["token"]: row for row in instances}
        category_by_token = {row["token"]: row["name"] for row in categories}
        scene_by_token = {row["token"]: row for row in scenes}
        log_to_map = {
            log_token: map_row["token"]
            for map_row in maps
            for log_token in map_row["log_tokens"]
        }
        selected_scene_tokens = self._get_scene_split(scenes)
        selected_sample_tokens = {
            row["token"] for row in samples if row["scene_token"] in selected_scene_tokens
        }
        tracked_ann_tokens_by_sample = self._build_target_index(
            annotations=annotations,
            selected_sample_tokens=selected_sample_tokens,
            instance_by_token=instance_by_token,
            category_by_token=category_by_token,
        )
        sample_to_map_token = {
            sample["token"]: log_to_map[scene_by_token[sample["scene_token"]]["log_token"]]
            for sample in samples
        }
        coord_ranges = self._build_coord_ranges(
            annotations=annotations,
            selected_sample_tokens=selected_sample_tokens,
            sample_to_map_token=sample_to_map_token,
        )

        built_samples: List[TrajectorySample] = []
        for ann in annotations:
            if ann["sample_token"] not in selected_sample_tokens:
                continue

            instance = instance_by_token[ann["instance_token"]]
            category_name = category_by_token[instance["category_token"]]
            if not self._is_target_agent_category(category_name):
                continue

            history_tokens = self._collect_history_tokens(ann, ann_by_token)
            future_tokens = self._collect_future_tokens(ann, ann_by_token)
            if history_tokens is None or future_tokens is None:
                continue

            history_xy = np.asarray(
                [ann_by_token[token]["translation"][:2] for token in history_tokens],
                dtype=np.float32,
            )
            future_xy = np.asarray(
                [ann_by_token[token]["translation"][:2] for token in future_tokens],
                dtype=np.float32,
            )

            origin = history_xy[-1].copy()
            history_rel = history_xy - origin
            future_rel = future_xy - origin
            rotation = self._make_heading_rotation(history_rel) if self.heading_aligned else None
            if rotation is not None:
                history_rel = history_rel @ rotation.T
                future_rel = future_rel @ rotation.T
            history_features = self._make_kinematic_features(history_rel)
            neighbor_features, neighbor_mask = self._build_neighbor_features(
                ego_ann=ann,
                ann_by_token=ann_by_token,
                target_ann_tokens=tracked_ann_tokens_by_sample[ann["sample_token"]],
                origin_xy=origin,
                rotation=rotation,
            )
            intent_label = self._classify_intent(history_rel=history_rel, future_rel=future_rel)

            built_samples.append(
                TrajectorySample(
                    agent_id=ann["instance_token"],
                    agent_type=self._to_agent_type(category_name),
                    sample_token=ann["sample_token"],
                    map_token=sample_to_map_token[ann["sample_token"]],
                    origin_xy=origin,
                    history_xy=history_rel,
                    future_xy=future_rel,
                    history_features=history_features,
                    neighbor_features=neighbor_features,
                    neighbor_mask=neighbor_mask,
                    intent_label=intent_label,
                )
            )

        return built_samples, coord_ranges

    def _get_scene_split(self, scenes: List[dict]) -> set[str]:
        if self.use_manual_mini_split and self.table_root.name == "v1.0-mini":
            valid_scenes = [scene for scene in scenes if scene["name"] not in EXCLUDED_MINI_SCENES]
            if self.split == "train":
                selected_names = {
                    scene["name"] for scene in valid_scenes if scene["name"] not in MANUAL_MINI_VAL_SCENES
                }
            else:
                selected_names = MANUAL_MINI_VAL_SCENES
            return {scene["token"] for scene in valid_scenes if scene["name"] in selected_names}

        ordered_names = sorted(scene["name"] for scene in scenes)
        rng = random.Random(self.split_seed)
        rng.shuffle(ordered_names)

        val_count = max(1, int(round(len(ordered_names) * self.val_ratio)))
        val_names = set(ordered_names[:val_count])

        if self.split == "train":
            selected_names = {name for name in ordered_names if name not in val_names}
        else:
            selected_names = val_names

        return {scene["token"] for scene in scenes if scene["name"] in selected_names}

    def _collect_history_tokens(
        self, ann: dict, ann_by_token: Dict[str, dict]
    ) -> Optional[List[str]]:
        tokens = [ann["token"]]
        cursor = ann
        for _ in range(self.past_steps - 1):
            prev_token = cursor["prev"]
            if not prev_token:
                return None
            tokens.append(prev_token)
            cursor = ann_by_token[prev_token]
        tokens.reverse()
        return tokens

    def _collect_future_tokens(
        self, ann: dict, ann_by_token: Dict[str, dict]
    ) -> Optional[List[str]]:
        tokens: List[str] = []
        cursor = ann
        for _ in range(self.future_steps):
            next_token = cursor["next"]
            if not next_token:
                return None
            tokens.append(next_token)
            cursor = ann_by_token[next_token]
        return tokens

    def _make_kinematic_features(self, history_xy: np.ndarray) -> np.ndarray:
        velocity = np.zeros_like(history_xy, dtype=np.float32)
        velocity[1:] = (history_xy[1:] - history_xy[:-1]) / self.dt
        velocity[0] = velocity[1] if len(history_xy) > 1 else 0.0

        acceleration = np.zeros_like(history_xy, dtype=np.float32)
        acceleration[1:] = (velocity[1:] - velocity[:-1]) / self.dt
        acceleration[0] = acceleration[1] if len(history_xy) > 1 else 0.0

        return np.concatenate([history_xy, velocity, acceleration], axis=-1).astype(np.float32)

    def _build_coord_ranges(
        self,
        annotations: List[dict],
        selected_sample_tokens: set[str],
        sample_to_map_token: Dict[str, str],
    ) -> Dict[str, dict]:
        grouped: Dict[str, List[np.ndarray]] = {}
        for ann in annotations:
            if ann["sample_token"] not in selected_sample_tokens:
                continue
            map_token = sample_to_map_token[ann["sample_token"]]
            grouped.setdefault(map_token, []).append(np.asarray(ann["translation"][:2], dtype=np.float32))

        coord_ranges: Dict[str, dict] = {}
        for map_token, coords in grouped.items():
            stacked = np.stack(coords, axis=0)
            coord_ranges[map_token] = {
                "x_min": float(stacked[:, 0].min()),
                "x_max": float(stacked[:, 0].max()),
                "y_min": float(stacked[:, 1].min()),
                "y_max": float(stacked[:, 1].max()),
            }
        return coord_ranges

    def _build_target_index(
        self,
        annotations: List[dict],
        selected_sample_tokens: set[str],
        instance_by_token: Dict[str, dict],
        category_by_token: Dict[str, str],
    ) -> Dict[str, List[str]]:
        target_ann_tokens_by_sample: Dict[str, List[str]] = {}
        for ann in annotations:
            if ann["sample_token"] not in selected_sample_tokens:
                continue
            instance = instance_by_token[ann["instance_token"]]
            category_name = category_by_token[instance["category_token"]]
            if not self._is_target_agent_category(category_name):
                continue
            target_ann_tokens_by_sample.setdefault(ann["sample_token"], []).append(ann["token"])
        return target_ann_tokens_by_sample

    def _build_neighbor_features(
        self,
        ego_ann: dict,
        ann_by_token: Dict[str, dict],
        target_ann_tokens: List[str],
        origin_xy: np.ndarray,
        rotation: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        neighbor_sequences: List[np.ndarray] = []
        ego_pos = np.asarray(ego_ann["translation"][:2], dtype=np.float32)

        for neighbor_token in target_ann_tokens:
            neighbor_ann = ann_by_token[neighbor_token]
            if neighbor_ann["instance_token"] == ego_ann["instance_token"]:
                continue

            neighbor_pos = np.asarray(neighbor_ann["translation"][:2], dtype=np.float32)
            if np.linalg.norm(neighbor_pos - ego_pos) > self.social_radius:
                continue

            history_tokens = self._collect_history_tokens(neighbor_ann, ann_by_token)
            if history_tokens is None:
                continue

            neighbor_history_xy = np.asarray(
                [ann_by_token[token]["translation"][:2] for token in history_tokens],
                dtype=np.float32,
            )
            neighbor_history_rel = neighbor_history_xy - origin_xy
            if rotation is not None:
                neighbor_history_rel = neighbor_history_rel @ rotation.T
            neighbor_features = self._make_kinematic_features(neighbor_history_rel)
            neighbor_sequences.append(neighbor_features)

        neighbor_sequences.sort(
            key=lambda seq: float(np.linalg.norm(seq[-1, :2])),
        )
        neighbor_sequences = neighbor_sequences[: self.max_neighbors]

        padded = np.zeros(
            (self.max_neighbors, self.past_steps, 6),
            dtype=np.float32,
        )
        mask = np.zeros((self.max_neighbors,), dtype=np.bool_)
        for idx, seq in enumerate(neighbor_sequences):
            padded[idx] = seq
            mask[idx] = True
        return padded, mask

    def _make_heading_rotation(self, history_rel: np.ndarray) -> np.ndarray | None:
        motion = history_rel[-1] - history_rel[-2]
        speed = float(np.linalg.norm(motion))
        if speed < 1e-5:
            return None

        heading = motion / speed
        cos_theta = heading[0]
        sin_theta = heading[1]
        return np.asarray(
            [
                [cos_theta, sin_theta],
                [-sin_theta, cos_theta],
            ],
            dtype=np.float32,
        )

    def _classify_intent(self, history_rel: np.ndarray, future_rel: np.ndarray) -> int:
        """
        Heuristic intent labels used for Phase 4 training:
        0 = crossing, 1 = waiting, 2 = turning, 3 = walking_straight
        """

        observed_motion = history_rel[-1] - history_rel[-2]
        observed_speed = float(np.linalg.norm(observed_motion))
        if observed_speed < 1e-3:
            observed_heading = np.array([1.0, 0.0], dtype=np.float32)
        else:
            observed_heading = observed_motion / observed_speed
        perp_heading = np.array([-observed_heading[1], observed_heading[0]], dtype=np.float32)

        final_disp = future_rel[-1]
        total_dist = float(np.linalg.norm(final_disp))

        start = np.zeros((1, 2), dtype=np.float32)
        full_path = np.concatenate([start, future_rel.astype(np.float32)], axis=0)
        step_vectors = np.diff(full_path, axis=0)
        step_lengths = np.linalg.norm(step_vectors, axis=1)
        path_length = float(step_lengths.sum())
        horizon_seconds = max(len(step_vectors) * self.dt, 1e-6)
        avg_speed = path_length / horizon_seconds

        headings: List[np.ndarray] = []
        for vec, length in zip(step_vectors, step_lengths):
            if length > 1e-4:
                headings.append(vec / length)

        heading_changes: List[float] = []
        for i in range(len(headings) - 1):
            cos_theta = float(np.clip(np.dot(headings[i], headings[i + 1]), -1.0, 1.0))
            heading_changes.append(float(np.degrees(np.arccos(cos_theta))))

        total_heading_change = float(np.sum(heading_changes)) if heading_changes else 0.0
        max_heading_change = float(np.max(heading_changes)) if heading_changes else 0.0
        sinuosity = path_length / max(total_dist, 1e-6)

        if avg_speed < 0.45:
            return 1

        path_lateral = future_rel @ perp_heading
        path_longitudinal = future_rel @ observed_heading
        peak_lateral = float(np.max(np.abs(path_lateral))) if len(path_lateral) else 0.0
        final_lateral = float(abs(path_lateral[-1])) if len(path_lateral) else 0.0
        final_longitudinal = float(path_longitudinal[-1]) if len(path_longitudinal) else 0.0

        if total_heading_change > 55.0 or (max_heading_change > 35.0 and sinuosity > 1.08):
            return 2

        if (
            peak_lateral > 1.5
            and final_lateral > 1.0
            and final_longitudinal > 0.3
            and total_heading_change <= 60.0
        ):
            return 0

        longitudinal = float(np.dot(final_disp, observed_heading))
        lateral = float(np.dot(final_disp, perp_heading))
        final_dir = final_disp / max(total_dist, 1e-6)
        heading_cos = float(np.clip(np.dot(observed_heading, final_dir), -1.0, 1.0))
        heading_delta_deg = float(np.degrees(np.arccos(heading_cos)))
        if total_dist < 1.0:
            return 1
        if abs(lateral) > max(1.25, abs(longitudinal) * 0.4) and heading_delta_deg <= 55.0:
            return 0
        if heading_delta_deg > 30.0:
            return 2
        return 3

    def _is_target_agent_category(self, category_name: str) -> bool:
        if category_name.startswith("human.pedestrian"):
            return True
        if self.include_cyclists and category_name in CYCLIST_CATEGORIES:
            return True
        return False

    def _to_agent_type(self, category_name: str) -> str:
        if category_name.startswith("human.pedestrian"):
            return "pedestrian"
        if category_name in CYCLIST_CATEGORIES:
            return "cyclist"
        return "unknown"
