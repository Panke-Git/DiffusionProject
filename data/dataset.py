import os
from PIL import Image
from torch.utils.data import Dataset
import data.util as Util
from model.ColorChannelCompensation import three_c as t_c

class UIEDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split
        self.resolution = resolution

        # DiffWater default expects:
        #   <dataroot>/input_<resolution> and <dataroot>/target_<resolution>
        # To better support common paired UIE datasets (e.g., LSUI) without changing
        # training settings, we also accept:
        #   <dataroot>/input and <dataroot>/GT (or target/gt)
        def _pick_existing_dir(root, candidates):
            for rel in candidates:
                p = os.path.join(root, rel)
                if os.path.isdir(p):
                    return p
            return None

        input_dir = _pick_existing_dir(
            dataroot,
            [f'input_{resolution}', 'input', 'Input']
        )
        target_dir = _pick_existing_dir(
            dataroot,
            [f'target_{resolution}', 'target', 'GT', 'gt', 'Target']
        )
        assert input_dir is not None, (
            f"Cannot find input folder under dataroot: {dataroot}. "
            f"Tried: input_{resolution}/, input/, Input/"
        )
        assert target_dir is not None, (
            f"Cannot find target/GT folder under dataroot: {dataroot}. "
            f"Tried: target_{resolution}/, target/, GT/, gt/, Target/"
        )

        self.input_path = Util.get_paths_from_images(input_dir)
        self.target_path = Util.get_paths_from_images(target_dir)

        # Basic sanity check for paired datasets.
        assert len(self.input_path) == len(self.target_path), (
            f"Paired dataset size mismatch: inputs={len(self.input_path)} targets={len(self.target_path)}. "
            f"input_dir={input_dir} target_dir={target_dir}"
        )

        self.dataset_len = len(self.target_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        target = Image.open(self.target_path[index]).convert("RGB")
        input = Image.open(self.input_path[index]).convert("RGB")

        # On-the-fly resize to match configured resolution (e.g., 256x256).
        # This keeps input/GT spatially aligned for paired training.
        if self.resolution is not None:
            target = target.resize((self.resolution, self.resolution), resample=Image.BICUBIC)
            input = input.resize((self.resolution, self.resolution), resample=Image.BICUBIC)

        input = t_c(input)

        [input, target] = Util.transform_augment([input, target], split=self.split, min_max=(-1, 1))

        return {'target': target, 'input': input, 'Index': index}



