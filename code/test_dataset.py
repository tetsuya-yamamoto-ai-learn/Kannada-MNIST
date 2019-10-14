import torch
from torch.utils.data import Dataset, DataLoader


class Test_Dataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx])


def main():
    # ダミーデータの準備
    data_num = 100
    X = torch.ones([data_num, 784])

    # テストデータセットインスタンスの作成
    test_dataset = Test_Dataset(X)

    # lenの確認
    assert len(test_dataset) == torch.Size([data_num, 784])

    # getitemの確認
    X = next(iter(test_dataset))
    assert X.shape == torch.Size([784])

    # DataLoaderの確認
    test_dataloader = DataLoader(test_dataset)
    for images, labels in test_dataset:
        print(labels.size(), images.size())


if __name__ == '__main__':
    main()
