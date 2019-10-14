import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def run():
    # =========================================================== #
    # 0. 下準備
    # =========================================================== #

    torch.manual_seed(0)  # torchの初期化
    BATCH_SIZE = 1000  # バッチ数

    # =========================================================== #
    # 1. データセットの準備(DatasetとDataloader)
    # =========================================================== #

    # DatasetとDataloaderを作成するために訓練データを読み込む
    train = pd.read_csv('../input/train.csv', nrows=100)

    # ラベルとデータに分割する
    X = train.iloc[:, 1:]
    y = train.iloc[:, 0]

    # 訓練データと検証データに分割する
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)
    assert (70, 784) == X_train.shape
    assert (70,) == y_train.shape

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
