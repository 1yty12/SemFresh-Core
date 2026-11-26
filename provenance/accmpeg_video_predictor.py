import argparse
import io
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


DEFAULT_ACCMPEG_ROOT = r"F:\paper\reference_code\AccMPEG-MLSys\AccMPEG-MLSys"
DEFAULT_MASKGEN_REL = r"maskgen\SSD\accmpegmodel.py"
DEFAULT_WEIGHT_REL = (
    r"maskgen_pths\COCO_detection_FPN_SSD_withconfidence_allclasses_new_"
    r"unfreezebackbone_withoutclasscheck.pth.best"
)
DEFAULT_TILE_SIZE = 16
DEFAULT_OUTPUT_SIZE = (720, 1280)


def parse_video_inputs(video_inputs):
    if video_inputs is None:
        return []
    if isinstance(video_inputs, (list, tuple)):
        return [str(item).strip() for item in video_inputs if str(item).strip()]
    return [item.strip() for item in str(video_inputs).split(",") if item.strip()]


def cache_path_for_video(video_path, cache_dir):
    video_abs = str(Path(video_path).resolve())
    digest = hashlib.md5(video_abs.encode("utf-8")).hexdigest()[:12]
    stem = Path(video_path).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return Path(cache_dir) / f"{safe_stem}_{digest}.npz"


def ensure_importance_caches(
    video_inputs,
    cache_dir,
    accmpeg_root=DEFAULT_ACCMPEG_ROOT,
    maskgen_file=None,
    weight_path=None,
    tile_size=DEFAULT_TILE_SIZE,
):
    video_paths = parse_video_inputs(video_inputs)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = []
    for video_path in video_paths:
        cache_path = cache_path_for_video(video_path, cache_dir)
        if not cache_path.exists():
            generate_importance_cache(
                video_path=video_path,
                output_path=cache_path,
                accmpeg_root=accmpeg_root,
                maskgen_file=maskgen_file,
                weight_path=weight_path,
                tile_size=tile_size,
            )
        elif not _cache_has_video_rate_fields(cache_path):
            _upgrade_cache_metadata(video_path, cache_path)
        cache_paths.append(str(cache_path))
    return cache_paths


def generate_importance_cache(
    video_path,
    output_path,
    accmpeg_root=DEFAULT_ACCMPEG_ROOT,
    maskgen_file=None,
    weight_path=None,
    tile_size=DEFAULT_TILE_SIZE,
    batch_size=1,
):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image

    video_path = Path(video_path)
    output_path = Path(output_path)
    accmpeg_root = Path(accmpeg_root)
    maskgen_file = Path(maskgen_file) if maskgen_file else accmpeg_root / DEFAULT_MASKGEN_REL
    weight_path = Path(weight_path) if weight_path else accmpeg_root / DEFAULT_WEIGHT_REL

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not maskgen_file.exists():
        raise FileNotFoundError(f"Mask generator file not found: {maskgen_file}")
    if not weight_path.exists():
        raise FileNotFoundError(f"Mask generator weight not found: {weight_path}")

    old_cwd = os.getcwd()
    old_path = list(os.sys.path)
    try:
        os.chdir(str(accmpeg_root))
        if str(accmpeg_root) not in os.sys.path:
            os.sys.path.insert(0, str(accmpeg_root))

        spec = importlib.util.spec_from_file_location("accmpeg_maskgen", str(maskgen_file))
        maskgen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(maskgen)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = maskgen.FCN()
        try:
            model.load(str(weight_path))
        except RuntimeError as exc:
            if "Attempting to deserialize object on a CUDA device" not in str(exc):
                raise
            state_dict = torch.load(str(weight_path), map_location=device)
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        frames = _read_video_frames(video_path)
        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        transform = T.Compose([T.Resize(DEFAULT_OUTPUT_SIZE), T.ToTensor()])
        importance_maps = []
        frame_sizes = []
        batch = []

        with torch.no_grad():
            for frame in frames:
                image = Image.fromarray(frame).convert("RGB")
                width, height = image.size
                frame_sizes.append([height, width])
                batch.append(transform(image))

                if len(batch) >= batch_size:
                    _append_predictions(model, batch, importance_maps, device)
                    batch = []

            if batch:
                _append_predictions(model, batch, importance_maps, device)

        importance_maps = np.asarray(importance_maps, dtype=np.float32)
        importance_maps = np.clip(importance_maps, 0.0, 1.0)
        frame_sizes = np.asarray(frame_sizes, dtype=np.int32)
        frame_data_sizes, frame_data_size_source = _frame_data_sizes_from_video(video_path, frames)
        fps = np.asarray(_read_video_fps(video_path), dtype=np.float32)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            importance_maps=importance_maps,
            frame_sizes=frame_sizes,
            frame_data_sizes=frame_data_sizes,
            frame_data_size_source=np.asarray(frame_data_size_source),
            fps=fps,
            video_paths=np.asarray([str(video_path.resolve())]),
            tile_size=np.asarray(tile_size, dtype=np.int32),
            source_resolution=np.asarray(DEFAULT_OUTPUT_SIZE, dtype=np.int32),
        )
    finally:
        os.chdir(old_cwd)
        os.sys.path[:] = old_path


def _append_predictions(model, batch, importance_maps, device):
    import torch

    tensor = torch.stack(batch, dim=0).to(device)
    logits = model(tensor)
    probs = logits.softmax(dim=1)[:, 1, :, :]
    importance_maps.extend(probs.detach().cpu().numpy())


def _read_video_frames(video_path):
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        return [np.asarray(frame)[:, :, :3] for frame in iio.imiter(str(video_path))]
    except Exception as exc:
        raise RuntimeError(
            "Unable to decode video. Install opencv-python or imageio, or provide an existing cache."
        ) from exc


def _read_video_fps(video_path):
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps > 0:
            return fps
    except Exception:
        pass
    return 1.0


def _frame_data_sizes_from_video(video_path, frames):
    frame_count = len(frames)
    packet_sizes = _read_ffprobe_packet_sizes_bits(video_path, frame_count)
    if packet_sizes is not None:
        return packet_sizes.astype(np.float32), "ffprobe_packet_size_bits"

    estimated_sizes = _estimate_jpeg_frame_sizes_bits(frames)
    return estimated_sizes.astype(np.float32), "jpeg_q90_estimated_bits"


def _read_ffprobe_packet_sizes_bits(video_path, expected_frames):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_size",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
    except Exception:
        return None

    sizes = []
    for frame in payload.get("frames", []):
        pkt_size = frame.get("pkt_size")
        if pkt_size is None:
            continue
        try:
            pkt_size = float(pkt_size)
        except (TypeError, ValueError):
            continue
        if pkt_size > 0:
            sizes.append(pkt_size * 8.0)

    if not sizes:
        return None

    sizes = np.asarray(sizes, dtype=np.float32)
    if len(sizes) >= expected_frames:
        return sizes[:expected_frames]

    padded = np.empty(expected_frames, dtype=np.float32)
    padded[: len(sizes)] = sizes
    padded[len(sizes) :] = float(np.mean(sizes))
    return padded


def _estimate_jpeg_frame_sizes_bits(frames, quality=90):
    try:
        import cv2

        sizes = []
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            sizes.append(float(encoded.size * 8))
        return np.asarray(sizes, dtype=np.float32)
    except Exception:
        pass

    from PIL import Image

    sizes = []
    for frame in frames:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=int(quality))
        sizes.append(float(len(buffer.getvalue()) * 8))
    return np.asarray(sizes, dtype=np.float32)


def _cache_has_video_rate_fields(cache_path):
    try:
        data = np.load(cache_path, allow_pickle=True)
        return "frame_data_sizes" in data.files and "fps" in data.files
    except Exception:
        return False


def _upgrade_cache_metadata(video_path, cache_path):
    video_path = Path(video_path)
    cache_path = Path(cache_path)
    data = np.load(cache_path, allow_pickle=True)
    arrays = {key: data[key] for key in data.files}
    num_frames = len(arrays["importance_maps"])
    if "frame_data_sizes" not in arrays:
        frames = _read_video_frames(video_path)
        arrays["frame_data_sizes"], source = _frame_data_sizes_from_video(video_path, frames)
        arrays["frame_data_size_source"] = np.asarray(source)
    elif "frame_data_size_source" not in arrays:
        arrays["frame_data_size_source"] = np.asarray("legacy_average_bits")
    if "fps" not in arrays:
        arrays["fps"] = np.asarray(_read_video_fps(video_path), dtype=np.float32)
    np.savez_compressed(cache_path, **arrays)


def refresh_cache_frame_data_sizes(video_path, cache_path):
    video_path = Path(video_path)
    cache_path = Path(cache_path)
    data = np.load(cache_path, allow_pickle=True)
    arrays = {key: data[key] for key in data.files}
    frames = _read_video_frames(video_path)
    frame_data_sizes, source = _frame_data_sizes_from_video(video_path, frames)
    num_maps = len(arrays["importance_maps"])
    if len(frame_data_sizes) != num_maps:
        raise ValueError(
            f"Decoded frame count ({len(frame_data_sizes)}) does not match cache importance maps ({num_maps}) for {cache_path}."
        )
    arrays["frame_data_sizes"] = frame_data_sizes.astype(np.float32)
    arrays["frame_data_size_source"] = np.asarray(source)
    np.savez_compressed(cache_path, **arrays)
    return source


def _main():
    parser = argparse.ArgumentParser(description="Generate AccMPEG importance-map caches.")
    parser.add_argument("--video_inputs", required=True, help="Comma-separated video paths.")
    parser.add_argument("--cache_dir", default=r"data\accmpeg_importance_cache")
    parser.add_argument("--accmpeg_root", default=DEFAULT_ACCMPEG_ROOT)
    parser.add_argument("--maskgen_file", default=None)
    parser.add_argument("--weight_path", default=None)
    parser.add_argument("--refresh_frame_data_sizes", action="store_true")
    args = parser.parse_args()

    paths = ensure_importance_caches(
        args.video_inputs,
        args.cache_dir,
        accmpeg_root=args.accmpeg_root,
        maskgen_file=args.maskgen_file,
        weight_path=args.weight_path,
    )
    for path in paths:
        if args.refresh_frame_data_sizes:
            video_paths = parse_video_inputs(args.video_inputs)
            for video_path, cache_path in zip(video_paths, paths):
                source = refresh_cache_frame_data_sizes(video_path, cache_path)
                print(f"Refreshed frame_data_sizes in {cache_path} from {source}.")
            break
        data = np.load(path)
        maps = data["importance_maps"]
        frame_data_sizes = np.asarray(data["frame_data_sizes"], dtype=np.float32)
        source = str(np.asarray(data["frame_data_size_source"]).reshape(-1)[0]) if "frame_data_size_source" in data.files else "unknown"
        print(
            f"{path}: importance_maps={maps.shape}, range=({maps.min():.4f}, {maps.max():.4f}), "
            f"frame_data_sizes=({frame_data_sizes.min():.1f}, {frame_data_sizes.max():.1f}), source={source}"
        )


if __name__ == "__main__":
    _main()
