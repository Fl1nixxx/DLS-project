import torch
from torch.utils.data import DataLoader, random_split

from сustom_dataset import BuildingsDataset

def get_loaders(root_dir,train_transform=None,val_transform=None,val_size=0.2,batch_size=16,seed=42):

    full_dataset = BuildingsDataset(root_dir=root_dir,transform=None)
    total_size = len(full_dataset)
    val_len = int(total_size * val_size)
    train_len = total_size - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset,[train_len, val_len],generator=generator)

    train_samples = [full_dataset.samples[i] for i in train_subset.indices]
    val_samples = [full_dataset.samples[i] for i in val_subset.indices]

    train_dataset = BuildingsDataset(root_dir=root_dir,samples=train_samples,transform=train_transformer)

    val_dataset = BuildingsDataset(root_dir=root_dir,samples=val_samples,transform=val_transformer)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    print(f"Total samples: {total_size}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    return train_loader, val_loader
