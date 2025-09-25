from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from utils.model_select import set_seed
from config import config
from sklearn.model_selection import train_test_split
import numpy as np

class ProgressiveImageFolderDataset:
    def __init__(self, image_path, img_size=(224, 224), batch_size=32, valid_ratio=0.2, seed=42, milestones=None):
        self.image_path = image_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.seed = seed
        self.milestones = milestones if milestones else [5, 10, 20]

        # transforms
        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # full dataset
        self.full_dataset = datasets.ImageFolder(self.image_path, transform=self.transforms)
        self.classes = self.full_dataset.classes
        self.num_classes = len(self.classes)

        # split train/valid
        targets = self.full_dataset.targets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.valid_ratio, random_state=self.seed)
        train_idx, valid_idx = next(sss.split(range(len(self.full_dataset)), targets))
        self.valid_dataset = Subset(self.full_dataset, valid_idx)

        # shuffle training datasets and split it
        rng = np.random.default_rng(self.seed)
        rng.shuffle(train_idx)
        self.train_splits = np.array_split(train_idx, len(self.milestones)+1)
        self.current_train_idx = []
        self.train_dataset = None

    def set_epoch(self, epoch):
        # epoch=0, part of training dataset
        if epoch == 0 and not self.current_train_idx:
            self.current_train_idx.extend(self.train_splits[0])

        # putting more dataset while touch milestone epoch
        for i, m in enumerate(self.milestones):
            if epoch == m:
                self.current_train_idx.extend(self.train_splits[i+1])
                break

        self.train_dataset = Subset(self.full_dataset, self.current_train_idx)
        print(f"[Epoch {epoch}] Using {len(self.train_dataset)} training samples")

    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def valid_dataloader(self):
        return self.get_dataloader(self.valid_dataset, shuffle=False)




# ImageFolderDataset.set_global_seed(42)

# data_module = ProgressiveImageFolderDataset(
#                         image_path=config.image_path,
#                         img_size=(config.image_size, config.image_size),
#                         batch_size=config.training_batch_size,
#                         valid_ratio=0.2,
#                         seed=42,
#                         milestones=[10, 15, 20, 25]
#                     )

# data_module.split_dataset()
# print(f'Classes Number: {data_module.classes}')
