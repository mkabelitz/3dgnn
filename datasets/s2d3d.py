from torch.utils.data import Dataset
import glob
import numpy as np
import cv2


class Dataset(Dataset):
    def __init__(self, path, areas, rate=1.0, flip_prob=None, crop_type=None, crop_size=None):

        self.path = path
        self.areas = areas
        self.flip_prob = flip_prob
        self.crop_type = crop_type
        self.crop_size = crop_size

        self.rgb_full_paths = []
        for area_idx, area in enumerate(self.areas):
            self.rgb_full_paths.append([file for file in glob.glob(path + area + '/data/preprocessed/rgb/*.png')])
        self.rgb_full_paths = [item for sublist in self.rgb_full_paths for item in sublist]
        self.rgb_full_paths_fast = [file.replace('work', 'fastwork') for file in self.rgb_full_paths]
        self.depth_full_paths = [file.replace('rgb', 'depth') for file in self.rgb_full_paths]
        self.hha_full_paths = [(file.replace('rgb', 'hha').replace('.png', '16.npy')) for file in self.rgb_full_paths]
        self.hha_full_paths_fast = [file.replace('work', 'fastwork') for file in self.hha_full_paths]
        self.xy_full_paths = [(file.replace('rgb', 'xy').replace('.png', '16.npy')) for file in self.rgb_full_paths]
        self.xy_full_paths_fast = [file.replace('work', 'fastwork') for file in self.xy_full_paths]
        self.label_full_paths = [file.replace('rgb', 'label_area_interpolated') for file in self.rgb_full_paths]
        self.label_full_paths_fast = [file.replace('rgb', 'label_area_interpolated').replace('work', 'fastwork') for
                                      file in self.rgb_full_paths]
        self.json_full_paths = [(file.replace('preprocessed/rgb', 'pose').replace('rgb.png', 'pose.json')) for file in
                                self.rgb_full_paths]

        self.rgb_full_paths = np.array(self.rgb_full_paths[:int(len(self.rgb_full_paths) * rate)])
        self.rgb_full_paths_fast = np.array(self.rgb_full_paths_fast[:int(len(self.rgb_full_paths_fast) * rate)])
        self.depth_full_paths = np.array(self.depth_full_paths[:int(len(self.depth_full_paths) * rate)])
        self.hha_full_paths = np.array(self.hha_full_paths[:int(len(self.hha_full_paths) * rate)])
        self.hha_full_paths_fast = np.array(self.hha_full_paths_fast[:int(len(self.hha_full_paths_fast) * rate)])
        self.xy_full_paths = np.array(self.xy_full_paths[:int(len(self.xy_full_paths) * rate)])
        self.xy_full_paths_fast = np.array(self.xy_full_paths_fast[:int(len(self.xy_full_paths_fast) * rate)])
        self.label_full_paths = np.array(self.label_full_paths[:int(len(self.label_full_paths) * rate)])
        self.label_full_paths_fast = np.array(self.label_full_paths_fast[:int(len(self.label_full_paths_fast) * rate)])

        self.json_full_paths = np.array(self.json_full_paths[:int(len(self.json_full_paths) * rate)])

        assert len(self.rgb_full_paths) == len(self.depth_full_paths)
        assert len(self.rgb_full_paths) == len(self.rgb_full_paths_fast)
        assert len(self.rgb_full_paths) == len(self.label_full_paths)
        assert len(self.rgb_full_paths) == len(self.label_full_paths_fast)
        assert len(self.rgb_full_paths) == len(self.hha_full_paths)
        assert len(self.rgb_full_paths) == len(self.xy_full_paths)
        assert len(self.rgb_full_paths) == len(self.json_full_paths)

    def __len__(self):
        return len(self.rgb_full_paths)

    def __getitem__(self, idx):
        rgb = cv2.cvtColor(cv2.imread(self.rgb_full_paths_fast[idx]), cv2.COLOR_RGB2BGR)
        hha = np.load(self.hha_full_paths_fast[idx]).astype(np.float32)
        rgb_hha = np.concatenate([rgb, hha], axis=2).astype(np.float32)
        label = cv2.imread(self.label_full_paths_fast[idx], cv2.IMREAD_ANYDEPTH)
        xy = np.load(self.xy_full_paths_fast[idx]).astype(np.float32)

        # random crop
        if self.crop_type is not None:
            max_margin = rgb_hha.shape[0] - self.crop_size
            if max_margin == 0:  # crop is original size, so nothing to crop
                self.crop_type = None
            elif self.crop_type == 'Center':
                rgb_hha = rgb_hha[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2, :]
                label = label[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2]
                xy = xy[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2, :]
            elif self.crop_type == 'Random':
                x_ = np.random.randint(0, max_margin)
                y_ = np.random.randint(0, max_margin)
                rgb_hha = rgb_hha[y_:y_ + self.crop_size, x_:x_ + self.crop_size, :]
                label = label[y_:y_ + self.crop_size, x_:x_ + self.crop_size]
                xy = xy[y_:y_ + self.crop_size, x_:x_ + self.crop_size, :]
            else:
                print('Bad crop')  # TODO make this more like, you know, good software
                exit(0)

        # random flip
        if self.flip_prob is not None:
            if np.random.random() > self.flip_prob:
                rgb_hha = np.fliplr(rgb_hha).copy()
                label = np.fliplr(label).copy()
                xy = np.fliplr(xy).copy()

        return rgb_hha, label, xy
