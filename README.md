# LeNet-5神经网络

### 介绍
根据YANN LECUN的论文《Gradient-based Learning Applied To Document Recognition》设计的LeNet-5神经网络，C语言写成，不依赖任何第三方库。
MNIST手写字符集初代训练识别率92%，多代训练识别率98%。

### DEMO
main.c文件为MNIST数据集的识别DEMO，直接编译即可运行，训练集60000张，测试集10000张。

### 项目环境
该项目为VISUAL STUDIO 2015项目，用VISUAL STUDIO 2015 UPDATE1及以上直接打开即可编译。采用ANSI C编写，因此源码无须修改即可在其它平台上编译。
如果因缺少openmp无法编译，请将lenet.c中的#include&lt;omp.h&gt;和#pragma omp parallel for删除掉即可。

### API
#####批量训练
lenet:  LeNet5的权值的指针，LeNet5神经网络的核心

input： 要训练的多个图片对应unsigned char二维数组的数组,指向的二维数组的batchSize倍大小内存空间指针。在MNIST测试DEMO中二维数组为28x28，每个二维数组数值分别为对应位置图像像素灰度值

result：要训练的多个图片分别对应的标签。为batchSize大小的数组

batchSize:批量训练输入图像（二维数组）的数量

void TrainBatch(LeNet5 *lenet, image *input, uint8 *result, int batchSize);

#####单个训练
lenet:  LeNet5的权值的指针，LeNet5神经网络的核心

input： 要训练的图片对应二维数组

result: 要训练的图片对应的标签的数值

void Train(LeNet5 *lenet, image input, uint8 result);

#####预测
lenet:  LeNet5的权值的指针，LeNet5神经网络的核心

input:  输入的图像的数据

return  返回值为预测的结果

uint8 Predict(LeNet5 *lenet, image input);

#####初始化
lenet:  LeNet5的权值的指针，LeNet5神经网络的核心

rand:   double rand()函数，用于生成随机数的函数的指针，该函数要求生成-1~+1之间的双精度浮点随机数。

void Initial(LeNet5 *lenet, double(*rand)());