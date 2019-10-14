import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from code.train_value_dataset import Train_value_dataset


def run():
    # =========================================================== #
    # 0. 下準備
    # =========================================================== #

    torch.manual_seed(0)  # torchの初期化
    BATCH_SIZE = 100  # バッチ数

    # =========================================================== #
    # 1. データセットの準備(DatasetとDataloader)
    # =========================================================== #

    # DatasetとDataloaderを作成するために訓練データを読み込む
    train = pd.read_csv('../input/train.csv', nrows=1000)

    # ラベルとデータに分割する(.valuesでndarrayにすることが重要！！)
    X = train.iloc[:, 1:].values
    y = train.iloc[:, 0].values

    # 訓練データと検証データに分割する
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)

    # データセットの作成
    train_dataset = Train_value_dataset(X_train, y_train)
    valid_dataset = Train_value_dataset(X_valid, y_valid)

    # データローダーの準備
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for images, labels in train_loader:
        print(images.size(), labels.size())

    # =========================================================== #
    # 2. ネットワークの準備(My_simple_net)
    # =========================================================== #

    # =========================================================== #
    # 3. ネットワークの学習(Model, nn.model)
    # =========================================================== #

    # =========================================================== #
    # 4. テストデータの識別
    # =========================================================== #

    # =========================================================== #
    # 5. 識別結果の出力
    # =========================================================== #


if __name__ == '__main__':
    run()
