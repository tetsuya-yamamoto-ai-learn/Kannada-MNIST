import torch
from torch import nn


class MySimplenet(nn.Module):
    def __init__(self):
        super().__init__()

        # ニューラルネットの構造を定義
        self.fc1 = nn.Linear(in_features=784, out_features=196)
        self.fc2 = nn.Linear(in_features=196, out_features=49)
        self.fc3 = nn.Linear(in_features=49, out_features=10)

    def forward(self, x):
        # ニューラルネットの順伝搬の処理を定義
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def main():
    # 定義したニューラルネットの作成
    model = MySimplenet()

    # ダミーデータの作成
    data_num = 100
    X_dummy = torch.randn([data_num, 784])

    # 順伝搬の実施と出力結果の確認
    outputs = model.forward(X_dummy)
    assert outputs.shape == torch.Size([data_num, 10])


if __name__ == '__main__':
    main()
