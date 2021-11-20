from tensorflow.keras import layers

'''
卷积层简介：
    一般使用Conv2D卷积一张图片（长乘高）
    Conv2D参数介绍
      filters：训练多少个卷积核/生成图像的厚度是多少
      kernel_size: 卷积核大小（3*3， 5*5等）
      strides：卷积核移动的跨度（1，1） 为1，1的时候是卷积核挨着上下左右去卷，写2，2就是隔一个去卷
      padding: 边缘填充 valid(不填充周围，卷积后会略微减少原图大小) same(填充0， 卷积后原图大小不变)
      data_format: (batch, height, width, channels) (多少张图像, 长, 宽, RGB频道)
'''
layers.Conv2D()
'''
池化层简介：
    layers.MaxPooling2D() 最大池化：选取池化核， 选取核内值最大的像素做保留
    MaxPooling2D参数介绍：
        pool_size: (2, 2)一般是2*2
        strides：None 默认不跳像素
        padding：边缘填充 valid（。。。） same（。。。）
'''
layers.MaxPooling2D()

'''
卷积一个2000*2000图片的过程:
    A：先卷积 - 变厚
    B: 再池化 - 变小
    重复AB
    最后全连接激活
    
优化方式:
    增大卷积层与池化层, 会增强提取特征的能力
'''