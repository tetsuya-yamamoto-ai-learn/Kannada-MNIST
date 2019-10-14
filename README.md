# 【Kaggleコンテスト】Kannada データセット

## コンテストの目標、背景

Kannada語数字（0~9）の画像データから、その画像が、どの数字を示しているか判別する識別器を作成する事.

Kannada数字については以下の画像参照.

![Kannada数字](https://storage.googleapis.com/kaggle-media/competitions/Kannada-MNIST/kannada.png)

Kannada語は、インドのカルナータカ州の人が使用する言語.

## コンテストファイルの説明

- train.csv
- test.csv
- sample_submission.csv
- Dig-MNIST.csv

## データセットの内容の説明

- train.csv[訓練用データ:]  
  1列目: label(どの数字か[実測値])  
  2~785列目: pixelデータ(輝度値:0~255)

- test.csv[提出用データ]  
  1列目: データid  
  2~785列目: pixelデータ(輝度値:0~255)  
  
- sample_submission[提出例]  
  1列目: データid  
  2列目: label(どの数字か[予測値])  

- Dig-MNIST.csv[検証用？]  
  1列目: label(どの数字か[実測値])  
  2~785列目: pixelデータ(輝度値:0~255)
  
## コンテストページ

[Kaggle | kannada MNIST](https://www.kaggle.com/c/Kannada-MNIST/overview)