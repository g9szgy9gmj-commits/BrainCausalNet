from base import BaseDataLoader
import numpy as np
from sklearn import preprocessing


class SubjectBlockFutureNpyTimeseriesDataLoader(BaseDataLoader):
    """
    Subject-specific block-to-next-block future prediction.

    For one selected subject, each sample is:

        X_m = subj_data[start : start + L]
        Y_m = subj_data[start + L : start + 2L]

    The stride is L, so observation blocks are consecutive and non-overlapping.
    This loader is intended for training one temporal surrogate per subject.
    """
    def __init__(self, data_dir, subject_idx, batch_size, time_step, output_window,
                 feature_dim, output_dim, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True):
        self.data_dir = data_dir
        raw = np.load(self.data_dir).astype('float32')

        self.subject_idx = int(subject_idx)
        if self.subject_idx < 0 or self.subject_idx >= raw.shape[0]:
            raise ValueError(f"subject_idx must be in [0, {raw.shape[0] - 1}], got {subject_idx}")

        self.batch_size = batch_size
        self.time_step = time_step
        self.output_window = output_window
        self.series_num = raw.shape[2]
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        if self.output_window != self.time_step:
            raise ValueError(
                "SubjectBlockFutureNpyTimeseriesDataLoader expects output_window == time_step "
                "for same-length block-to-next-block prediction")

        subj_data = raw[self.subject_idx]
        scaler = preprocessing.MinMaxScaler(feature_range=(0.5, 1))
        subj_data = scaler.fit_transform(subj_data)

        n_timepoints = subj_data.shape[0]
        pair_len = self.time_step + self.output_window
        assert pair_len <= n_timepoints, \
            f"2 * time_step ({pair_len}) must be <= data length ({n_timepoints}) for subject {self.subject_idx}"

        self.dataset = []
        for start_idx in range(0, n_timepoints - pair_len + 1, self.time_step):
            inp_start = start_idx
            inp_end = start_idx + self.time_step
            tgt_start = inp_end
            tgt_end = tgt_start + self.output_window

            inp = subj_data[inp_start:inp_end].reshape(
                self.time_step, self.series_num, self.feature_dim)
            tgt = subj_data[tgt_start:tgt_end].reshape(
                self.output_window, self.series_num, self.output_dim)
            self.dataset.append((inp, tgt))

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
