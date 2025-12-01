# 食用教程

## 前言

> 这个模型训练的绝大多代码来自于AI（千问和Gemini 3 Pro），我只是收集了数据集，并且dog和cat的数据集来源于Kaggle。其他的数据集来源于百度、Bing。
>
> 注意:warning:：该训练使用的是`GPU`训练，因此你应该安装了你的显卡对应（或者支持）的`CUDA`和`cuDNN`。

# 环境

使用的`Python`版本是`3.10.0`，`Python`环境不能小于`3.10.0`。创建一个环境以免污染主环境。

创建好并激活环境执行`pip install -r requirements.txt`

# 各文件说明

> - `datasets`存放数据集的文件夹
> - `change_file_name.py` 修改`datasets\train_1\xxx`文件夹内的文件名
> - `check_img.py` 检查`datasets\train_1\xxx`文件夹内的图片是否可以用于训练。
> - `split_test_dataset.py` 用于划分`test`数据集的脚本，是使用`predict_test_model.py`的前提。
> - `predict_test_model.py` 当模型训练好以后，用于评估模型
> - `train_main.py` 训练的主程序