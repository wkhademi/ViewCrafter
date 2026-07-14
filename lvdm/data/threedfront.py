import os
import pandas as pd
from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset


class ThreeDFront(Dataset):
    """
    3D-FRONT sparse-view finetuning dataset.

    Assumes data_dir is the self-contained output of
    scripts/prepare_3dfront_train_data.py:
        videos/video_<clip_id>.mp4       (real GT frames -> 'video')
        videos/video_cond_<clip_id>.mp4  (point-cloud-rendered conditioning frames -> 'video_cond')
    and meta_path is metadata_train.csv / metadata_val.csv with columns:
        name (-> caption), oripath (-> video), videopath (-> video_cond), frameid
    """
    def __init__(self, meta_path, data_dir, video_length=25, resolution=(576, 1024), fps=10):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps = fps
        self._load_metadata()

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path)
        metadata['caption'] = metadata['name']
        del metadata['name']
        metadata['caption'] = metadata['caption'].fillna('')
        self.metadata = metadata

    def _get_video_path(self, sample):
        video_fp = os.path.join(self.data_dir, sample['oripath'])
        cond_video_fp = os.path.join(self.data_dir, sample['videopath'])
        return video_fp, cond_video_fp

    def __getitem__(self, index):
        sample = self.metadata.iloc[index]
        video_path, cond_video_path = self._get_video_path(sample)
        caption = sample['caption']
        frameid = int(sample['frameid'])

        video_reader = VideoReader(video_path, ctx=cpu(0))
        cond_video_reader = VideoReader(cond_video_path, ctx=cpu(0))
        assert len(video_reader) == self.video_length and len(cond_video_reader) == self.video_length, \
            f"expected {self.video_length} frames, got {len(video_reader)}/{len(cond_video_reader)} for {video_path}"

        frame_indices = list(range(self.video_length))
        frames = video_reader.get_batch(frame_indices)
        frames_cond = cond_video_reader.get_batch(frame_indices)

        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
        frames_cond = torch.tensor(frames_cond.asnumpy()).permute(3, 0, 1, 2).float()

        assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), \
            f"frames={frames.shape}, expected resolution={self.resolution}"

        frames = (frames / 255 - 0.5) * 2
        frames_cond = (frames_cond / 255 - 0.5) * 2

        # guard against h264 re-encoding drift between the two independently
        # compressed mp4s: the reference frame must match 'video' exactly.
        frames_cond[:, frameid, :, :] = frames[:, frameid, :, :]

        data = {
            'video': frames,
            'caption': caption,
            'path': video_path,
            'fps': self.fps,
            'frame_stride': 1,
            'video_cond': frames_cond,
            'frameid': frameid,
        }
        return data

    def __len__(self):
        return len(self.metadata)
