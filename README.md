这个项目是利用tensorflow构建卷积神经网络用于人脸识别任务.

开发环境:

1. ubuntu18.04 TLS
2. python3.6
3. tensorflow-gpu 1.3.0
4. CUDA 9.0.176
5. cudnn 7.5
6. opencv3.4.3

1. ubuntu18.04TLS
2. 

数据集为班级同学人脸图像,尺寸128*128

数据读取方式为通过txt文件读取路径和标签而后生成tensorflow标准读取文件.tfrecord文件. 各个文件作用如下

1. create_negative_sets.py 创建负样本, 即不是人脸或者不属于要识别的人. 方式为从自己某个文件夹里读图, 利用opencv从图中切割部分, 以生成.
2. create_data.py 读取文件夹内文件, 生成训练集, 测试集, 验证集的txt文件.
3. create_tfrecords.py 利用之前生成的txt文件生成最终网络读取的数据集
4. file_transfer.py 读取数据集并进行预处理, 作为网络输入.
5. network.py 网络结构
6. train_val.py 训练和验证
7. test.py 在测试集上测试
8. test_by_camera.py 利用摄像头测试.  先用dlib从摄像头图片中提取人脸, 再进行与训练时一样的预处理, 最后进行测试.

本篇未对卷积神经网络内部实现原理及方式进行介绍, 读者感兴趣的话可以看我的另一个项目. [纯手写实现卷积神经网络](https://github.com/chst1/-c-) 

代码中可以有不少错误欢迎大家指正, 也欢迎大家与我讨论相关问题.

联系方式: QQ 2322253097



