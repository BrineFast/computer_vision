import numpy as np
import torch
import os
import cv2
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import resnet50, ResNet
from sklearn.model_selection import train_test_split
from typing import NoReturn

train_path: str = "photos/six/train"
test_path: str = "photos/six/test"

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


class ImageDataSet(Dataset):
    def __init__(self, filenames: list) -> NoReturn:
        self._filenames: list = filenames

    @staticmethod
    def add_pad(img: np.array, shape: np.array) -> np.array:
        padded_img: np.array = img[0][0] * np.ones(shape + img.shape[2:3], dtype=np.uint8)
        x_offset: int = int((padded_img.shape[0] - img.shape[0]) / 2)
        y_offset: int = int((padded_img.shape[1] - img.shape[1]) / 2)
        padded_img[x_offset:x_offset + img.shape[0], y_offset:y_offset + img.shape[1]] = img
        return padded_img

    @staticmethod
    def resize(img: np.array, shape: np.array) -> np.array:
        scale: int = min(shape[0] * 1.0 / img.shape[0], shape[1] * 1.0 / img.shape[1])
        return cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


class TrainDataSet(ImageDataSet):
    def __init__(self, filenames: list, labels: list = None) -> NoReturn:
        super().__init__(filenames)
        self._labels: list = labels

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, idx: int) -> tuple:
        img: np.array = self.resize(cv2.imread(self._filenames[idx]), (224, 224))
        img: np.array = self.add_pad(img, (224, 224))
        img: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img: np.array = torch.tensor(img, dtype=torch.float).permute(2, 0, 1) / 255.
        return img, self._labels[idx]


class TestDataSet(ImageDataSet):
    def __init__(self, filenames: list) -> NoReturn:
        super().__init__(filenames)

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, idx: int) -> np.array:
        img: np.array = self.resize(cv2.imread(self._filenames[idx]), (224, 224))
        img: np.array = self.add_pad(img, (224, 224))
        img: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img: np.array = torch.tensor(img, dtype=torch.float).permute(2, 0, 1) / 255.
        return img


class Model():
    def __init__(self, path_to_train: str) -> NoReturn:
        self._mapping: dict = dict()
        self.train_files, self.train_labels = self._get_partitioned_file_names(path_to_train)
        self._classes_count: int = len(os.listdir(path_to_train))
        self._model: ResNet = resnet50(pretrained=True)
        self._prepare_model()

    def train(self, epoch_count: int) -> NoReturn:
        train_filenames, test_filenames, train_labels, test_labels = \
            train_test_split(self.train_files,
                             self.train_labels,
                             test_size=0.2,
                             random_state=42)

        train: DataLoader = DataLoader(TrainDataSet(train_filenames, train_labels),
                                       shuffle=True,
                                       batch_size=25,
                                       num_workers=0)
        test: DataLoader = DataLoader(TrainDataSet(test_filenames, test_labels),
                                      shuffle=True,
                                      batch_size=25,
                                      num_workers=0)
        for x in tqdm(range(epoch_count)):
            for batch in train:
                self.evaluate(batch,
                              CrossEntropyLoss(),
                              Adam(self._model.parameters(), lr=0.0005))

            self._model.eval()
            test_accuracy: list = list()
            test_real: list = list()
            with torch.no_grad():
                for batch_x, batch_y in tqdm(test):
                    outputs: np.array = (self._model(batch_x.to('cuda'))
                                         .detach()
                                         .cpu()
                                         .numpy())
                    test_accuracy.append(outputs)
                    test_real.append(batch_y
                                     .detach()
                                     .cpu()
                                     .numpy())
            self._model.train()

    def test(self, test_img_count: float, test_img: str) -> NoReturn:
        test_file_names: list = list(map(lambda x: os.path.join(test_img, x),
                                         os.listdir(test_img)))
        test_filenames = random.sample(test_file_names, test_img_count)
        test_ds: DataLoader = DataLoader(TestDataSet(test_filenames), batch_size=25, num_workers=0)
        path_num: int = 0
        for batch in test_ds:
            label_pred = self._model(batch.to('cuda'))
            for i in label_pred:
                img_class: str = self._mapping[torch.argmax(i).item()]
                plt.figure(path_num)
                plt.title(img_class)
                plt.imshow(cv2.cvtColor(cv2.imread(test_ds.dataset._filenames[path_num]), cv2.COLOR_BGR2RGB))
                plt.savefig(f"classified_{img_class}_{path_num}")
                path_num += 1
        plt.show()

    def _get_partitioned_file_names(self, path_to_ds: str) -> tuple:
        result: list = list()
        classes: list = list()
        for cl, train_dir in enumerate(os.listdir(path_to_ds)):
            self._mapping[cl] = train_dir
            for file in os.listdir(os.path.join(path_to_ds, train_dir)):
                result.append(os.path.join(path_to_ds, train_dir, file))
                classes.append(cl)
        return result, classes

    def _prepare_model(self) -> NoReturn:
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.fc = torch.nn.Linear(self._model.fc.in_features,
                                         self._classes_count)
        self._model.to('cuda')
        ct: int = 0
        for child in self._model.children():
            ct += 1
            if ct < 47:
                for param in child.parameters():
                    param.requires_grad = True

    def evaluate(self, batch, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam) -> NoReturn:
        optimizer.zero_grad()
        image, label = batch
        label_pred = self._model(image.to('cuda'))
        loss = criterion(label_pred, label.to('cuda'))
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    model: Model = Model(train_path)
    model.train(50)

    model.test(10, test_path)