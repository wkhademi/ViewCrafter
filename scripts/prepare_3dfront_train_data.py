"""Build (video, video_cond) training pairs for ViewCrafter sparse-view
finetuning on 3D-FRONT.

Consumes the camera-path clips rendered by BlenderProc-3DFront's
generate_train_path.py (25 real GT frames per clip, interpolated between two
anchor views) plus each room's original pre-existing renders. For every clip:

  - video      = the 25 real rendered frames along the path (true GT).
  - video_cond = a point cloud backprojected from anchor_a (frame 0) +
                 anchor_b (last frame, both freshly re-rendered by Part 0) +
                 K random extra views from the room's original render pool,
                 rendered into all 25 path poses with the same PyTorch3D
                 renderer used at inference/eval time. Frame 0 is overwritten
                 with the true anchor_a RGB.

Output: self-contained `videos/*.mp4` pairs + metadata_train.csv /
metadata_val.csv in --out_dir, ready to be copied to a training machine and
read by lvdm/data/threedfront.py.
"""
import argparse
import csv
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from utils.pvd_utils import generate_traj, run_render, save_video

DEPTH_INVALID_THRESHOLD = 1000.0  # meters; BlenderProc writes ~1e10 for no-hit ("sky") pixels


def blender2opencv_cam(cam_T):
    # blender camera to opencv camera: flip local Y and Z only (keep local X).
    # Negating local X here mirrors the render left-right - verified empirically
    # by measuring optical shift against the true rendered trajectory.
    y_z_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    cam_T = y_z_swap.dot(cam_T)
    cam_T[:3, 1] *= -1
    cam_T[:3, 2] *= -1
    return cam_T


def backproject(depth, extrinsics, K, device):
    u, v = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
    u = u.reshape([1, -1])[0]
    v = v.reshape([1, -1])[0]
    z = depth[v, u]
    x = (u - K[0][2]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]
    point_cam = np.vstack([x, y, z]).T
    R, T = extrinsics[:3, :3], extrinsics[:3, 3]
    point_world = point_cam.dot(R.T) + T
    return point_world.reshape(depth.shape[0], depth.shape[1], 3).astype(np.float32)


def load_view(room_dir, view_id):
    with h5py.File(os.path.join(room_dir, view_id), "r") as f:
        colors = np.array(f["colors"]).astype(np.float32) / 255.0
        depth = np.array(f["depth"]).astype(np.float32)
        cam_T = blender2opencv_cam(np.array(f["cam_Ts"]).astype(np.float64))
    mask = (depth > 0) & (depth < DEPTH_INVALID_THRESHOLD)
    return colors, depth, cam_T, mask


def build_video_cond(room_dir, clip_dir, n_frames, path_colors, path_cam_Ts, K, H, W,
                      k_extra_min, k_extra_max, rng, device):
    # anchors: frame 0 and last frame of the freshly-rendered path itself
    anchor_colors = [path_colors[0], path_colors[-1]]
    anchor_depths = []
    anchor_masks = []
    for idx in (0, n_frames - 1):
        with h5py.File(os.path.join(clip_dir, f"{idx}.hdf5"), "r") as f:
            depth = np.array(f["depth"]).astype(np.float32)
        anchor_depths.append(depth)
        anchor_masks.append((depth > 0) & (depth < DEPTH_INVALID_THRESHOLD))
    anchor_cam_Ts = [path_cam_Ts[0], path_cam_Ts[-1]]

    # K extra views from the room's original render pool
    pool = [os.path.basename(p) for p in glob.glob(os.path.join(room_dir, "*.hdf5"))]
    k_extra = rng.randint(k_extra_min, k_extra_max)
    extra_ids = rng.sample(pool, min(k_extra, len(pool)))

    imgs, pts3d, masks = [], [], []
    for colors, depth, cam_T, mask in zip(anchor_colors, anchor_depths, anchor_cam_Ts, anchor_masks):
        imgs.append(colors)
        pts3d.append(backproject(depth, cam_T, K, device))
        masks.append(mask)
    for view_id in extra_ids:
        colors, depth, cam_T, mask = load_view(room_dir, view_id)
        imgs.append(colors)
        pts3d.append(backproject(depth, cam_T, K, device))
        masks.append(mask)

    focals = torch.tensor([[K[0, 0], K[1, 1]]], device=device).expand(n_frames, -1).float()
    principal_points = torch.tensor([[K[0, 2], K[1, 2]]], device=device).expand(n_frames, -1).float()
    path_c2ws = torch.from_numpy(np.stack(path_cam_Ts)).to(device).float()
    camera_traj, num_views = generate_traj(path_c2ws, H, W, focals, principal_points, device)

    render_results, _ = run_render(pts3d, imgs, masks, H, W, camera_traj, num_views, device)
    return render_results  # [n_frames, H, W, 3] in [0, 1]


def process_clip(job, args, rng, device):
    video_path = f"videos/video_{job['clip_id']}.mp4"
    video_cond_path = f"videos/video_cond_{job['clip_id']}.mp4"
    row = {"name": "", "oripath": video_path, "videopath": video_cond_path, "frameid": 0}

    if os.path.exists(os.path.join(args.out_dir, video_path)) and \
            os.path.exists(os.path.join(args.out_dir, video_cond_path)):
        print(f"Skipping {job['clip_id']}: already processed.")
        return row  # already processed in a previous run, skip re-rendering

    room_dir = os.path.join(args.renderings_dir, job["scene_id"], job["room_id"])
    clip_dir = os.path.join(room_dir, "train_paths", job["clip_id"])
    hdf5_files = sorted(glob.glob(os.path.join(clip_dir, "*.hdf5")),
                        key=lambda p: int(os.path.basename(p).split(".")[0]))
    if len(hdf5_files) < args.n_frames:
        print(f"Skipping {job['clip_id']}: only {len(hdf5_files)}/{args.n_frames} frames rendered.")
        return None

    path_colors, path_cam_Ts = [], []
    H = W = None
    for i in range(args.n_frames):
        with h5py.File(os.path.join(clip_dir, f"{i}.hdf5"), "r") as f:
            colors = np.array(f["colors"]).astype(np.float32) / 255.0
            cam_T = blender2opencv_cam(np.array(f["cam_Ts"]).astype(np.float64))
        path_colors.append(colors)
        path_cam_Ts.append(cam_T)
        H, W = colors.shape[0], colors.shape[1]

    K = np.load(os.path.join(args.renderings_dir, "cam_K.npy"))

    video_cond = build_video_cond(room_dir, clip_dir, args.n_frames, path_colors, path_cam_Ts,
                                   K, H, W, args.k_extra_min, args.k_extra_max, rng, device)

    video = torch.from_numpy(np.stack(path_colors)).to(device).float()  # [n_frames, H, W, 3]

    video = F.interpolate(video.permute(0, 3, 1, 2), size=(args.height, args.width),
                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    video_cond = F.interpolate(video_cond.permute(0, 3, 1, 2), size=(args.height, args.width),
                               mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    video_cond[0] = video[0]  # frame 0 (anchor_a) is always the real GT frame

    os.makedirs(os.path.join(args.out_dir, "videos"), exist_ok=True)
    save_video(video.clamp(0, 1), os.path.join(args.out_dir, video_path))
    save_video(video_cond.clamp(0, 1), os.path.join(args.out_dir, video_cond_path))

    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs_json", type=str, required=True,
                        help="Job list from sample_train_pairs.py.")
    parser.add_argument("--renderings_dir", type=str,
                        default="/media/wkhademi/BackupDrive/Datasets/3D-FRONT-renderings")
    parser.add_argument("--out_dir", type=str,
                        default="/media/wkhademi/BackupDrive/Datasets/3D-FRONT-train-pairs")
    parser.add_argument("--n_frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--k_extra_min", type=int, default=0,
                        help="Min random extra views merged into the point cloud beyond the two anchors. "
                             "Lower = sparser/more holes in video_cond, closer to a true two-view input.")
    parser.add_argument("--k_extra_max", type=int, default=1,
                        help="Max random extra views merged into the point cloud beyond the two anchors.")
    parser.add_argument("--n_val_rooms", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    import json
    with open(args.jobs_json, "r") as f:
        jobs = json.load(f)

    rooms = sorted({(j["scene_id"], j["room_id"]) for j in jobs})
    rng.shuffle(rooms)
    val_rooms = set(rooms[:args.n_val_rooms])

    os.makedirs(args.out_dir, exist_ok=True)
    train_rows, val_rows = [], []
    for job in jobs:
        try:
            row = process_clip(job, args, rng, device)
        except Exception as e:
            print(f"Skipping {job['clip_id']}: {type(e).__name__}: {e}")
            row = None
        if row is None:
            continue
        if (job["scene_id"], job["room_id"]) in val_rooms:
            val_rows.append(row)
        else:
            train_rows.append(row)

    for name, rows in (("metadata_train.csv", train_rows), ("metadata_val.csv", val_rows)):
        with open(os.path.join(args.out_dir, name), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "oripath", "videopath", "frameid"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {name}")


if __name__ == "__main__":
    main()
