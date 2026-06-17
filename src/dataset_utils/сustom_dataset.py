from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

class BuildingsDataset(Dataset):
    def __init__(self,root_dir,samples=None,image_dir="images",mask_dir="binary_masks",image_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),transform=None,):
        self.root_dir = Path(root_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_exts = image_exts
        self.transform = transform

        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"No image-mask pairs found in {self.root_dir}")

    def _collect_samples(self):
        samples = []

        city_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]

        for city_dir in city_dirs:
            images_dir = city_dir / self.image_dir
            masks_dir = city_dir / self.mask_dir

            if not images_dir.exists() or not masks_dir.exists():
                continue

            image_paths = []
            for ext in self.image_exts:
                image_paths.extend(sorted(images_dir.glob(f"*{ext}")))

            for img_path in image_paths:
                mask_path = masks_dir / img_path.name

                if not mask_path.exists():
                    matches = list(masks_dir.glob(img_path.stem + ".*"))
                    if len(matches) == 0:
                        print(f"Warning: mask not found for {img_path}")
                        continue
                    mask_path = matches[0]

                samples.append((img_path, mask_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        mask = (mask > 0).astype("float32")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            mask = mask.float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
