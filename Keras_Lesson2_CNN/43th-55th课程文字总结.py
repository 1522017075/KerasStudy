"""
    43th: 训练一半保存模型
        可以使用回调函数keras.callbacks.ModelCheckpoint('路径', save_weights_only=True)等参数
        训练到一半保存下来当前参数或模型
    44th: 迁移学习
        keras.applications.VGG16(weights='imagenet', include_top = False)
        比如卷积, 可以使用别人预训练好的卷积基, 然后替换成自己的输出函数, 如 VGG 模型架构
    48th: 迁移学习优化
        使用已经调好的卷积基就不需要反向传播去训练他, 但是每次训练的时候, 图片都会重新经过卷积基提取特征造成重复运算
        可以手动写函数, 先使用卷积基提取特征, 然后将提取的特征作为全连接层的输入, 就会快很多
        手动提取特征就是先初始化一个结果矩阵/数组(知道他的输出格式), 然后经过运算, 填充到里面去, 最后将结果矩阵/数组返回
    49th: 迁移学习微调
        微调: 冻结模型库的底部的卷积层(一般是通用特征 比如纹理提取), 共同训练新添加的分类器层和顶部部分卷积层(一般与特定任务相关)
             只有分类器已经训练好了, 才能微调卷积基的顶部卷积层
        步骤: 1.在预训练卷积基上添加自定义层
             2. 冻结卷积基的所有层(卷积基.trainable=False)
             3. 训练添加的分类层
             4. 解冻卷积基的一部分层(卷积基.trainable=True, 然后遍历每一层, 解冻后三层(层.trainable=False)这样子)


    50th: 常见图像分类模型
        Xception VGG16 VGG19 ResNet50 InceptionV3 ...
        Xception : 在 ImageNet 上 验证集 top1 0.79 和 top5 0.945 的准确率
                    只支持 channels_last 的维度顺序(高, 宽, 通道)
        使用:
            keras.applications.xception.Xception(参数)
    51th: 输出层总结
        回归问题 - 输出一个连续的值 - 没有激活函数
        二分类问题 - 输出一个概率值 - sigmoid 激活函数
        多分类问题 - 输出 N 个值, 和为 1 - softmax 激活函数
        多标签问题 - 分解成很多个二分类/多分类问题(如性别, 家乡在哪个城市等)


    52th: 批标准化 / 归一化
        常见的数据标准化形式:
            标准化和归一化: 将数据减去其平均值使其中心为 0, 然后将数据除以其标准差使其标准差为 1
        批标准化:
            不仅在数据输入模型前对数据做标准化, 在网络每一次变换之后都应该考虑数据标准化
        Keras 实现批标准化:
            BatchNormalization 层通常在卷积层或密集连接层之后添加如下层
            layers.Batchnormalization()
    53th: 超参数选择原则
        首先开发一个过拟合的模型:
            添加更多层
            让每一层变得更大
            训练更多的轮次
        然后抑制过拟合:
            dropout
            正则化
            图像增强
        再次, 调节超参数
            学习速率
            隐藏层单元数
            训练轮次
        ...
        最后做交叉验证
    55th: 获取模型中间层的输出
        sub_model = keras.models.Model(inputs = conv_base.input,
                                        outputs = conv_base.get_layer('层的名字').output)
"""