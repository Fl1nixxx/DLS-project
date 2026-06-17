import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transformer = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-45, 45), mode=0, p=0.5),
    A.Resize(672, 672),

    A.Perspective(scale=(0.05, 0.08), keep_size=True, p=0.3),

    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ToTensorV2()])

val_transformer = A.Compose([
    A.Resize(672, 672),
    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ToTensorV2()])
