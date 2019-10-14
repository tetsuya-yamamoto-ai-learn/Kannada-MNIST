import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from code.MyNet import MySimplenet
from code.plot_methods import loss_plot
from code.test_dataset import Test_Dataset
from code.train_value_dataset import Train_value_dataset


def run():
    # =========================================================== #
    # 0. 下準備
    # =========================================================== #

    torch.manual_seed(0)  # torchの初期化
    BATCH_SIZE = 10000  # バッチ数

    # =========================================================== #
    # 1. データセットの準備(DatasetとDataloader)
    # =========================================================== #

    # DatasetとDataloaderを作成するために訓練データを読み込む
    train = pd.read_csv('../input/train.csv')

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

    # =========================================================== #
    # 2. ネットワークの準備(My_simple_net)
    # =========================================================== #

    # 自分で定義したMySimplenetをインスタンス化
    model: nn.Module = MySimplenet()

    # =========================================================== #
    # 3. ネットワークの学習(Model, nn.model)
    # =========================================================== #

    # 最適化手法の設定(ネットワークのパラメータを渡す必要あり)
    optimizer = torch.optim.Adam(model.parameters())
    # print(len(list(model.parameters())[0])) # (784, 196)...一層目の重み？
    # print(len(list(model.parameters())[1])) # (196, )...一層目のバイアス？
    # print(len(list(model.parameters())[2])) # (196, 49)
    # print(len(list(model.parameters())[3])) # (49, )
    # print(len(list(model.parameters())[4])) # (49, 10)
    # print(len(list(model.parameters())[5])) # (10, )

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    # 学習の実施
    NUM_EPOCHS = 10
    model.train()
    loss_list = []
    for epochs in range(1, NUM_EPOCHS + 1):
        for images, labels in train_loader:
            # 勾配初期化
            optimizer.zero_grad()

            # 順伝搬
            outputs = model.forward(images)

            # 損失計算
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # 逆伝搬(勾配計算)
            loss.backward()

            # パラメータ更新
            optimizer.step()

    # 損失関数の変化をグラフ表示
    loss_plot(loss_list)

    # =========================================================== #
    # (検証データを用いて精度を確認)
    # =========================================================== #
    model.eval()
    bool = np.array([])
    for images, labels in valid_loader:
        y_valid_pred = model(images)
        _, y_valid_pred_labels = torch.max(y_valid_pred, dim=1)

        bool = np.append(bool, y_valid_pred_labels.numpy() == labels.numpy())

    df_bool = pd.DataFrame(bool)
    print(df_bool.sum() / len(df_bool))

    # =========================================================== #
    # 4. テストデータの識別
    # =========================================================== #

    # テストデータの読み込み
    test = pd.read_csv('../input/test.csv')

    # データidとテストデータの分割
    id_test = test.iloc[:, 0]
    X_test = test.iloc[:, 1:].values

    # テストデータセット
    test_dataset = Test_Dataset(X_test)

    # テストデータセットローダー
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 推測の実施
    model.eval()
    predictions = np.array([], dtype=int)
    for images in test_loader:
        # データの予測
        y_pred = model(images)

        # 最大値のラベルの取得
        _, y_pred_label = torch.max(y_pred, dim=1)
        predictions = np.append(predictions, y_pred_label.numpy())

    # =========================================================== #
    # 5. 識別結果の出力
    # =========================================================== #
    test_csv = pd.DataFrame(id_test)
    test_csv['Label'] = predictions
    test_csv.to_csv('../output/01_Simple_submission', index=False)


if __name__ == '__main__':
    run()
