"""
Generate BD/HC EC from subject-wise block-to-next-block CausalFormer models.

Workflow:
    for each subject s:
        load subject s checkpoint
        build non-overlapping observation blocks from subject s
        perturb the last observed position of each source ROI
        average future prediction changes into an EC matrix

This keeps the subject-wise surrogate logic:
    subject s model -> perturb subject s data -> EC_s
and finally stacks all EC_s into ec_all_subjects.npy.
"""

import argparse
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing

import model.model as module_arch
from utils import prepare_device


def parse_subjects(raw_subjects, start_subject, end_subject, n_subjects):
    if raw_subjects:
        subjects = []
        for part in raw_subjects.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = [int(x) for x in part.split("-", 1)]
                subjects.extend(range(start, end + 1))
            else:
                subjects.append(int(part))
    else:
        start = 0 if start_subject is None else int(start_subject)
        end = n_subjects - 1 if end_subject is None else int(end_subject)
        subjects = list(range(start, end + 1))

    bad = [s for s in subjects if s < 0 or s >= n_subjects]
    if bad:
        raise ValueError(f"Subject indices out of range [0, {n_subjects - 1}]: {bad}")
    return subjects


def checkpoint_epoch(path):
    match = re.search(r"checkpoint-epoch(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(checkpoint_root, run_id_prefix, subject_idx, checkpoint_name, allow_latest):
    subject_dir = checkpoint_root / f"{run_id_prefix}_subject_{subject_idx:03d}"
    checkpoint_path = subject_dir / checkpoint_name
    if checkpoint_path.exists():
        return checkpoint_path

    if allow_latest:
        candidates = sorted(subject_dir.glob("checkpoint-epoch*.pth"), key=checkpoint_epoch)
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(f"No checkpoint found for subject {subject_idx}: {checkpoint_path}")


def load_subject_model(checkpoint_path, gpu):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if gpu is not None:
        config["n_gpu"] = 1

    model = getattr(module_arch, config["arch"]["type"])(config, **config["arch"]["args"])
    state_dict = checkpoint["state_dict"]
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("module.", "") if key.startswith("module.") else key
        if key.endswith(".base"):
            continue
        cleaned[key] = value
    model.load_state_dict(cleaned)

    device, _ = prepare_device(config["n_gpu"])
    model = model.to(device).eval()
    data_args = config["data_loader"]["args"]
    return (
        model,
        device,
        int(data_args["time_step"]),
        int(data_args["output_window"]),
        int(data_args["feature_dim"]),
        int(data_args["series_num"]),
    )


def make_block_windows(subj_data, time_step, output_window, series_num, feature_dim):
    scaler = preprocessing.MinMaxScaler(feature_range=(0.5, 1))
    subj_data = scaler.fit_transform(subj_data).astype("float32")

    pair_len = time_step + output_window
    if pair_len > len(subj_data):
        raise ValueError(
            f"time_step + output_window ({pair_len}) must be <= subject length ({len(subj_data)})"
        )

    windows = [
        subj_data[start:start + time_step].reshape(time_step, series_num, feature_dim)
        for start in range(0, len(subj_data) - pair_len + 1, time_step)
    ]
    if not windows:
        raise ValueError("No complete block windows were generated")
    return np.stack(windows, axis=0).astype("float32")


@torch.no_grad()
def forward_in_batches(model, device, windows, batch_size):
    outputs = []
    for start in range(0, windows.shape[0], batch_size):
        end = min(start + batch_size, windows.shape[0])
        xb = torch.from_numpy(windows[start:end]).to(device)
        outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def compute_subject_ec(
    model,
    device,
    subj_data,
    time_step,
    output_window,
    series_num,
    feature_dim,
    pert_strength,
    batch_size,
    last_output_only,
):
    windows = make_block_windows(subj_data, time_step, output_window, series_num, feature_dim)
    unperturbed = forward_in_batches(model, device, windows, batch_size)

    ec = np.zeros((series_num, series_num), dtype=np.float64)
    for source in range(series_num):
        torch.cuda.empty_cache()
        diff_sum = np.zeros(series_num, dtype=np.float64)
        denom = 0
        for start in range(0, windows.shape[0], batch_size):
            end = min(start + batch_size, windows.shape[0])
            xb = torch.from_numpy(windows[start:end].copy()).to(device)
            xb[:, -1, source, :] += pert_strength

            perturbed = model(xb).cpu().numpy()
            diff = perturbed - unperturbed[start:end]
            selected_diff = diff[:, -1:, :, :] if last_output_only else diff

            diff_sum += selected_diff.sum(axis=(0, 1, 3))
            denom += selected_diff.shape[0] * selected_diff.shape[1] * selected_diff.shape[3]

        ec[source, :] = diff_sum / denom

    np.fill_diagonal(ec, 0)
    return ec.astype("float32"), windows.shape[0]


def keep_signed_topk(ec, top_k):
    if top_k <= 0:
        return ec.copy()

    filtered = ec.copy()
    series_num = filtered.shape[0]
    for row in range(series_num):
        row_data = filtered[row].copy()

        pos_idx = np.where(row_data > 0)[0]
        pos_keep = (
            pos_idx[np.argsort(row_data[pos_idx])[-top_k:]]
            if len(pos_idx) > 0 else np.array([], dtype=int)
        )

        neg_idx = np.where(row_data < 0)[0]
        neg_keep = (
            neg_idx[np.argsort(row_data[neg_idx])[:top_k]]
            if len(neg_idx) > 0 else np.array([], dtype=int)
        )

        mask = np.zeros(series_num, dtype=bool)
        mask[pos_keep] = True
        mask[neg_keep] = True
        filtered[row, ~mask] = 0

    return filtered


def normalize_signed_ec_per_subject(ec_all):
    ec_all = np.nan_to_num(ec_all, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    ec_norm = np.zeros_like(ec_all, dtype="float32")
    ranges = []
    for idx in range(ec_all.shape[0]):
        subject_ec = ec_all[idx]
        x_min = float(subject_ec.min())
        x_max = float(subject_ec.max())
        ranges.append((x_min, x_max))
        if x_min < 0:
            neg_mask = subject_ec < 0
            ec_norm[idx][neg_mask] = subject_ec[neg_mask] / abs(x_min)
        if x_max > 0:
            pos_mask = subject_ec > 0
            ec_norm[idx][pos_mask] = subject_ec[pos_mask] / x_max
    return ec_norm, np.asarray(ranges, dtype="float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/BD_HC/BD_HC_sig.npy")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="saved/models/BD_HC Subjectwise Block32 Predict32 Causality Learning",
    )
    parser.add_argument("--run_id_prefix", type=str, default="subjectwise_block32_pred32")
    parser.add_argument("--checkpoint_name", type=str, default="model_best.pth")
    parser.add_argument("--allow_latest_checkpoint", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output/BD_HC_block32_subjectwise_EC")
    parser.add_argument("--subjects", type=str, default=None, help="e.g. 0,1,2 or 0-20")
    parser.add_argument("--start_subject", type=int, default=None)
    parser.add_argument("--end_subject", type=int, default=None)
    parser.add_argument("--pert_strength", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--last_output_only", action="store_true")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_norm", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data = np.load(args.data_path)
    n_subjects = data.shape[0]
    subjects = parse_subjects(args.subjects, args.start_subject, args.end_subject, n_subjects)

    checkpoint_root = Path(args.checkpoint_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {run_dir}")
    print(f"Data: {args.data_path} shape={data.shape}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Subjects: {len(subjects)}/{n_subjects}")
    print(
        f"EC aggregation: block windows, output_steps="
        f"{'last' if args.last_output_only else 'all'}, top_k={args.top_k}"
    )

    first_checkpoint = resolve_checkpoint(
        checkpoint_root, args.run_id_prefix, subjects[0],
        args.checkpoint_name, args.allow_latest_checkpoint
    )
    _, _, _, _, _, series_num = load_subject_model(first_checkpoint, args.gpu)

    ec_dense_all = np.zeros((n_subjects, series_num, series_num), dtype="float32")
    ec_all = np.zeros((n_subjects, series_num, series_num), dtype="float32")
    processed_mask = np.zeros(n_subjects, dtype=bool)
    summary_rows = []

    for order, subject_idx in enumerate(subjects, start=1):
        print(f"[{order}/{len(subjects)}] Subject {subject_idx} ... ", end="", flush=True)
        try:
            checkpoint_path = resolve_checkpoint(
                checkpoint_root, args.run_id_prefix, subject_idx,
                args.checkpoint_name, args.allow_latest_checkpoint
            )
            model, device, time_step, output_window, feature_dim, model_series_num = load_subject_model(
                checkpoint_path, args.gpu
            )
            if model_series_num != series_num:
                raise ValueError(f"Inconsistent series_num: {model_series_num} vs {series_num}")

            ec_dense, n_windows = compute_subject_ec(
                model=model,
                device=device,
                subj_data=data[subject_idx],
                time_step=time_step,
                output_window=output_window,
                series_num=series_num,
                feature_dim=feature_dim,
                pert_strength=args.pert_strength,
                batch_size=args.batch_size,
                last_output_only=args.last_output_only,
            )
            ec = keep_signed_topk(ec_dense, args.top_k)

            ec_dense_all[subject_idx] = ec_dense
            ec_all[subject_idx] = ec
            processed_mask[subject_idx] = True
            summary_rows.append(
                {
                    "subject_idx": subject_idx,
                    "status": "ok",
                    "checkpoint": str(checkpoint_path),
                    "n_windows": n_windows,
                    "dense_min": float(ec_dense.min()),
                    "dense_max": float(ec_dense.max()),
                    "topk_min": float(ec.min()),
                    "topk_max": float(ec.max()),
                    "error": "",
                }
            )
            print(
                f"done windows={n_windows} dense_range=[{ec_dense.min():.6f}, {ec_dense.max():.6f}] "
                f"topk_range=[{ec.min():.6f}, {ec.max():.6f}]"
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "subject_idx": subject_idx,
                    "status": "error",
                    "checkpoint": "",
                    "n_windows": "",
                    "dense_min": "",
                    "dense_max": "",
                    "topk_min": "",
                    "topk_max": "",
                    "error": repr(exc),
                }
            )
            print(f"error: {exc}")
        finally:
            torch.cuda.empty_cache()

    np.save(run_dir / "ec_all_subjects_dense.npy", ec_dense_all)
    np.save(run_dir / "ec_all_subjects_notopk.npy", ec_dense_all)
    np.save(run_dir / "ec_all_subjects_topk.npy", ec_all)
    np.save(run_dir / "ec_all_subjects.npy", ec_all)
    np.save(run_dir / "processed_subject_mask.npy", processed_mask)
    np.save(run_dir / "processed_subject_indices.npy", np.asarray(subjects, dtype=np.int64))

    if not args.no_norm:
        ec_norm, ranges = normalize_signed_ec_per_subject(ec_all)
        dense_norm, dense_ranges = normalize_signed_ec_per_subject(ec_dense_all)
        np.save(run_dir / "ec_all_subjects_norm.npy", ec_norm)
        np.save(run_dir / "ec_all_subjects_dense_norm.npy", dense_norm)
        np.save(run_dir / "ec_subject_ranges.npy", ranges)
        np.save(run_dir / "ec_dense_subject_ranges.npy", dense_ranges)

    try:
        import pandas as pd

        pd.DataFrame(summary_rows).to_csv(run_dir / "subject_ec_summary.csv", index=False)
    except Exception:
        pass

    print(f"\nAll done. Saved to {run_dir}/")
    print(f"Processed subjects: {int(processed_mask.sum())}/{len(subjects)} selected")
    print(f"EC shape: {ec_all.shape}")


if __name__ == "__main__":
    main()
