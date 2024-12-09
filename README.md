# DAGAN网络实现人脸生成

### 结果展示

> 训练100个epoch的效果，基本损失就到这了，不会再进一步减小。要想实现更好更清晰的图像，就需要对网络进行进一步的改进。

![1733731302710](image/README/1733731302710.png)

### 数据集

CelebA数据集

[下载链接](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### 预处理

由于数据集较大（20w张图片），所以直接在训练时再将数据导入Dataloader中会出现CPU瓶颈的现象。即CPU占用率特别高，GPU占用率却偏低。

所以建议下载好数据集之后，先使用 `img_resize_to_64x64.py`提前将数据集裁剪为64x64大小的图片保存到新文件夹中。这样会加快一定训练速度。

### 训练和测试

`dcgan_me.py`

将最下方的 `TRAIN `标志位置1进行训练，`TEST `标志位置1测试效果。

> 具体训练相关的超参数在 `train()`函数中调整
