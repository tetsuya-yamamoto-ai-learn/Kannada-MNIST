import torch
from torch.utils.data import Dataset, DataLoader


class Train_value_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    # ダミーの訓練データを用意して機能を満たしているか確認
    data_num = 10  # ダミーの訓練データのデータ数
    X_train = torch.ones(size=(data_num, 784))
    y_train = torch.ones(size=(data_num,))
    train_dataset = Train_value_dataset(X=X_train, y=y_train)

    # __len__の返り値の確認
    assert data_num == len(train_dataset)

    # __getitem__の返り値の確認
    X, y = next(iter(train_dataset))
    assert torch.Size([784]) == X.size()
    assert torch.float32 == X.dtype
    assert 1.0 == y

    # DataLoaderに使用可能か確認
    BATCH_SIZE = 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for images, labels in train_loader:
        print(images.size(), labels.size())


if __name__ == '__main__':
    main()
