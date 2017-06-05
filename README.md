# 简单的手写数字识别

[TOC]

## 第一章 绪论

### 1.1 主要内容及流程

### 1.2 论文的组织结构

​	论文共分为五章，内容包含图像预处理、字符分割与字符识别等内容。

​	第一章为绪论，提出所做的主要工作和工作流程，并给出全文的组织结构。

​	第二章为图像预处理的研究，目标是图像实现图像二值化和去除噪声，对常用的三种图像二值化方法固定阈值、自适应阈值、Otsu 进行了比较；简述了形态学图像处理的腐蚀及膨胀操作，及形态学的高级形态开运算和闭运算。

​	第三章为字符分割的研究，其工作流程是，输入一个文本图像，先将文本分行，再将每行的单词分开，最后分割单个字符。

​	第四章为字符识别的研究，这里采用的是多层卷积神经网络，构建了 Softmax 回归模型。本代码使用了 TensorFlow 来对模型进行训练，在这里进行了详细的阐述。

## 第二章 图像预处理

### 2.1 二值化

​	摄像头拍摄的图像是彩色图像，包含的信息量巨大，对于图片的内容，可以简单的分为前景和背景，为了让计算机更快、更好地识别文字，需要对彩色图像进行处理，使图片只剩下前景信息与背景信息。可以简单的定义前景信息为白色，背景信息为黑色，这样就得到了二值化图。

​	首先使用 OpenCV 中的   `cvtColor` 函数将 RGB 图像转为灰度图像；然后，采用“图像阈值化”的方法将灰度图像的前景与背景分离。常用的阈值化方法有：(1). 固定阈值 (2). 自适应阈值 (3). Ostu

#### 2.1.1 固定阈值
​	固定阈值是一种最简单的图像二值化方法，即对于图像的所有像素，把大于某个临界灰度值的像素设为灰度最大值，而把小于这个值的像素灰度设为灰度最小值。而对于亮度不均匀的图像，无法确定一个适合于全局的阈值来将图片的前景和背景分离开，在这种情况下，应该把图片分割成小块，在每一块区域内确定一个合适的阈值，再进行分割，也就是自适应阈值。
#### 2.1.2 自适应阈值

​	自适应阈值是根据像素的邻域块的像素值分布来确定该像素位置上的二值化阈值。在灰度图像中，灰度值变化明显的区域往往是物体的轮廓，采用自适应阈值的好处在于每个像素位置处的二值化阈值不是固定不变的，而是由其周围邻域像素的分布来决定的。亮度较高的图像区域的二值化阈值通常会较高，而亮度较低的图像区域的二值化阈值则会相适应地变小。不同亮度、对比度、纹理的局部图像区域将会拥有相对应的局部二值化阈值。常用的局部自适应阈值有：(1). 局部邻域块的均值 (2). 局部邻域块的高斯加权和。

​                                    ![adaptive_threshold.png](https://ooo.0o0.ooo/2017/06/04/593392921baa5.png)

#### 2.1.3 Otsu

​	Otsu 方法是一种全局化的动态二值化方法，又叫大津法或最大类间方差算法。该算法的基本思想是，对于一幅图像，设当前景与背景的分割阈值为t时，前景点占图像比例为 $w_0$，均值为 $u_0$，背景点占图像比例为 $w_1$，均值为 $u_1$，则整个图像的均值为 $u = w_0u_0+w_1u_1$。建立目标函数 $g(t)=w_0*(u_0- u)^2+w_1(u_1-u)^2$，$g(t)$ 就是当分割阈值为t时的类间方差表达式。OTSU 算法使得g(t)取得全局最大值，当 $g(t)$ 为最大时所对应的t称为最佳阈值。大多数情况下，Otsu 算法都可以得到很好的结果。

### 1.2 噪声去除

​	现实中的数字图像在数字化和传输过程中常受到成像设备与外部环境噪声干扰等影响，称为含噪图像或噪声图像。噪声即是指妨碍计算机理解图像目标信息的内容，对于不同的文档，对噪声的定义会有所不同，根据噪声的特征进行去噪，就叫做噪声去除。图像去噪主要是在图像的频域上滤波或对图像进行形态学图像处理，常见的图像去噪方法有均值滤波器、自适应维纳滤波器、中值滤波器、形态学噪声滤除器和小波去噪。

​	本代码采用的去噪方法是形态学图像处理的闭运算，即对图像进行先膨胀后腐蚀的方法。膨胀就是图像中的高亮部分进行膨胀，即“领域扩张”，效果图拥有比原图更大的高亮区域。腐蚀就是原图中的高亮部分被腐蚀，即“领域被蚕食”，效果图拥有比原图更小的高亮区域。也就是说，膨胀和腐蚀是对图像的高亮部分而言的。

#### 1.2.1 膨胀与腐蚀

​	膨胀本质上是求局部最大值的操作，膨胀或腐蚀操作的原理是，将图像或图像的一部分区域 (A) 与核 (B) 进行卷积。核可以是任何的形状和大小，它拥有一个单独定义出来的参考点，我们称其为锚点（anchorpoint）。一般情况下，核是一个小的中间带有参考点和实心正方形或者圆盘。核 B 与图形卷积，即计算核 B 覆盖的区域的像素点的最大值，并把这个最大值赋值给参考点指定的像素。这样就会使图像中的高亮区域逐渐增长。膨胀的数学表达式为：

​                                                           $dst(x, y) = max_{(x',y')\neq0}src(x + x', y + y')$

​	效果图为：

​                                  ![dilation.png](https://ooo.0o0.ooo/2017/06/04/5933a477af906.png)

​	腐蚀和膨胀是一对相反的操作，也就是求局部最小值的操作。腐蚀的数学表达式为：

​				                         $dst(x, y) = max_{(x',y')\neq0}src(x + x', y + y')$

​	效果图为：

​                                   ![erosion.png](https://ooo.0o0.ooo/2017/06/04/5933a6b4b76b2.png)

​	膨胀与腐蚀的原理图为：![屏幕快照 2017-06-04 下午2.33.06.png](https://ooo.0o0.ooo/2017/06/04/5933a9c750106.png)



#### 1.2.2 开运算与闭运算

​	形态学的高级形态，如开运算、闭运算、形态学梯度、顶帽等是建立在腐蚀与膨胀的基础之上的。开运算（Opening Operation），其实就是先腐蚀后膨胀的过程。其数学表达式如下：

​				$dst = open(src, element)  = dilate(erode(src, element))$

​	开运算可以用来消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积。效果图为：

​                                                         ![opening.png](https://ooo.0o0.ooo/2017/06/04/5933acc018e19.png)

​	先膨胀后腐蚀的过程称为闭运算(Closing Operation)，其数学表达式如下：

​				 $dst = close(src, element) = erode(dilate(src, element))$

​	闭运算能够排除小型黑色区域。效果图如下所示：

​                                                         ![closing.png](https://ooo.0o0.ooo/2017/06/04/5933ad2b09144.png)



## 第三章 字符分割

### 3.1 分行



### 3.2 分单词



### 3.3 分字符

​	对单词中的每个字符进行分割时使用的是轮廓检测方法，OpenCV 封装了这个函数   `findContours` 。轮廓（Contours），指的是有相同颜色或者密度，连接所有连续点的一条曲线。检测轮廓的工作对形状分析和物体检测与识别都非常有用。为了提高轮廓检测的准确性，在轮廓检测之前，首先要对图片进行二值化或者 Canny 边缘检测。函数的原型为：

```python
cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])  
```

​	轮廓检测结果为：                                                        <img src="https://ooo.0o0.ooo/2017/06/04/5933c6d0bef26.png" alt="contours.png" title="contours.png" style="zoom:50%"/>

​	字符分割结果为：

![char_seg.png](https://ooo.0o0.ooo/2017/06/04/5933f965c6b1a.png)



## 第四章 字符识别

### 4.1 卷积神经网络

​	卷积神经网络（Convolutional Neural Network, CNN）是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现，是深度学习技术中极具代表的网络结构之一。

​	卷积神经网络由一个或多个卷积层和顶端的全连通层组成，同时也包括关联权重和池化层，这一结构使得卷积神经网络能够利用输入数据的二维结构。相比较其他深度、前馈神经网络，卷积神经网络需要估计的参数更少，在图像和语音识别方面能够给出更优的结果。

#### 4.1.1 局部连接与权值共享

​	卷积神经网络CNN的出现是为了解决多层神经网络中多层感知器全连接和梯度发散的问题。其引入三个核心思想：(1). 局部感知（Local Field）(2). 权值共享（Shared Weights）(3). 下采样（Subsampling），获得了某种程度的位移、尺度、形变不变性，极大地提升了计算速度，减少了连接数量。

1. 局部连接（**Sparse Connectivity**）

   ​	对于一个$1000 × 1000$ 的输入图像而言，如果下一个隐藏层的神经元数目为$10^6$个，采用全连接则有$1000 × 1000 × 10^6 = 10^{12}$个权值参数，如此数目巨大的参数几乎难以训练；而采用局部连接，隐藏层的每个神经元仅与图像中$10 × 10$的局部图像相连接，那么此时的权值参数数量为$10 × 10 × 10^6 = 10^8$，将直接减少4个数量级。

   ![PHb.jpg](https://ooo.0o0.ooo/2017/06/05/5934ef60b64a5.jpg)

   ​	（左边是全连接，右边是局部连接）

2. 权值共享（**Shared Weights**）

   ​	权值共享进一步减少了参数的数量。其原理是，不同的图像或者同一张图像共用一个卷积核，减少重复的卷积核。同一张图像当中可能会出现相同的特征，共享卷积核能够进一步减少权值参数。

   ​	在局部连接中隐藏层的每一个神经元连接的是一个$10 × 10$的局部图像，因此有$10 × 10$个权值参数，将这$10 × 10$个权值参数共享给剩下的神经元，也就是说隐藏层中$10^6$个神经元的权值参数相同，那么此时不管隐藏层神经元的数目是多少，需要训练的参数就是这$10 × 10$个权值参数，也就是卷积核的大小，如下图。

   ![59H.jpg](https://ooo.0o0.ooo/2017/06/05/5934ef60b417a.jpg)

   ​	滤波器就像一双眼睛，某人环游全世界，所看到的信息在变，但采集信息的双眼不变。然而不同人的双眼看同一个局部信息所感受到的不同，即一千个读者有一千个哈姆雷特，所以不同的滤波器就像不同的双眼，不同的人有着不同的反馈结果。

   ​	一个卷积核仅提取了图像的一种特征，如果要多提取出一些特征，可以增加多个卷积核。不同的卷积核能够得到图像的不同映射下的特征，称之为 Feature Map。如果有100个卷积核，最终的权值参数也仅为$100 × 100 = 10^4$个。另外，偏置参数也是共享的，同一种滤波器共享一个。

#### 4.1.2 结构

​	下图是一个经典的CNN结构，称为 LeNet-5网络。

![cnn.png](https://ooo.0o0.ooo/2017/06/04/5933f6073a98b.png)

​	可以看出，CNN中主要有两种类型的网络层，分别是卷积层和池化/采样层。卷积层的作用是提取图像的各种特征；池化层的作用是对原始特征信号进行抽象，从而大幅度减少训练参数，另外还可以减轻模型过拟合的程度。

1. 卷积层（**Convolutional Layer**）

   ​	卷积层是卷积核在上一级输入层上通过逐一滑动窗口计算而得，卷积核中的每一个参数都相当于传统神经网络中的权值参数，与对应的局部像素相连接，将卷积核的各个参数与对应的局部像素值相乘之和，通常还要再加上一个偏置参数，得到卷积层上的结果。

2. 线性整流层（**ReLU Layer**）

   ​	使用线性整流![{\displaystyle f(x)=\max(0,x)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5fa5d3598751091eed580bd9dca873f496a2d0ac)作为这一层神经的激活函数。它可以增强判定函数和整个神经网络的非线性特性，而本身并不会改变卷积层。

   ​	其他的一些函数也可以用于增强网络的非线性特性，如双曲正切函数![{\displaystyle f(x)=\tanh(x)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/1a319ec32dbb0c625fa4802baf9252d1f00854e2)或者 Sigmoid 函数![{\displaystyle f(x)=(1+e^{-x})^{-1}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/6f6e8c1bc5646e39b558bc46f997c5db23471af5)。相比其它函数来说，ReLU 函数的优势在于它可以将神经网络的训练速度提升数倍，而并不会对模型的泛化准确度造成产生显著影响。

3. 池化层（**Pooling Layer**）

   ​	实际上是一种形式的向下采样。有多种不同形式的非线性池化函数，最常见的是“最大池化”：将输入的图像划分为若干个矩形区域，对每个子区域输出最大值。其原理是，在发现一个特征之后，它的精确位置远不及它和其他特征的相对位置的关系重要。池化层会不断地减小数据的空间大小，因此参数的数量和计算量也会下降，这在一定程度上也控制了过拟合。

   ​	CNN 的卷积层之间通常都会周期性地插入池化层。池化层通常会分别作用于每个输入的特征并减小其大小。目前最常用形式的池化层是每隔2个元素从图像划分出![2\times 2](https://wikimedia.org/api/rest_v1/media/math/render/svg/f8a0e3400ffb97d67c00267ed50cddfe824cbe80)的区块，然后对每个区块中的4个数取最大值。这将会减少75%的数据量。

4. 损失函数层（**Loss Layer**）

   ​	通常是网络的最后一层，用于决定训练过程如何来“惩罚”网络的预测结果和真实结果之间的差异。各种不同的损失函数适用于不同类型的任务。例如，Softmax 交叉熵损失函数常常被用于在K个类别中选出一个，而 Sigmoid 交叉熵损失函数常常用于多个独立的二分类问题。

### 4.2 TensorFlow 的应用

​	TensorFlow是一个非常强大的用来做大规模数值计算的库。其所擅长的任务之一就是实现以及训练深度神经网络。

#### 4.2.1 TensorFlow 基本使用

1. 张量

   ​	TensorFlow 用张量（Tensor）这种数据结构来表示所有的数据。可以把张量想象成一个 n 维的数组，一个张量有一个静态类型和动态类型的维数。张量可以在计算图中的节点之间流通。

   ​	在 TensorFlow 系统中，张量的维数被描述为阶。但是张量的阶和矩阵的阶并不是同一个概念，张量的阶是张量维数的一个数量描述。除了维度，Tensors 有一个数据类型属性，可以为一个张量指定 tf.float32，tf.int64，tf.bool 等类型。

2. 计算图

   ​	TensorFlow 是一个编程系统，使用图来表示计算任务。图中的节点被称之为 *op* (operation 的缩写)。一个 op 获得 0 个或多个 Tensor，执行计算，产生 0 个或多个 Tensor 。

   ​	TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段，在构建阶段，op 的执行步骤 被描述成一个图。在执行阶段，使用会话执行执行图中的 op。

#### 4.2.2 构建 Softmax 回归模型

1. Softmax 回归

   ​	Softmax回归解决的是多分类问题，类标 ![\textstyle y](http://ufldl.stanford.edu/wiki/images/math/c/8/1/c81e76c28ed991b22b8c1bb8fa392701.png) 可以取 ![\textstyle k](http://ufldl.stanford.edu/wiki/images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) 个不同的值。因此，对于训练集 ![\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}](http://ufldl.stanford.edu/wiki/images/math/5/e/c/5ec89e9cf3712d45b80e93258352ea8f.png)，我们有 ![y^{(i)} \in \{1, 2, \ldots, k\}](http://ufldl.stanford.edu/wiki/images/math/7/d/c/7dc095cfb7e3e1fc6bdbc358bd3e2888.png)。例如，在数字识别任务中有 ![\textstyle k=10](http://ufldl.stanford.edu/wiki/images/math/1/b/8/1b84ec945b47439de6a73660b826df20.png)个不同的类别。

   假设函数（Hypothesis Function） ![\textstyle h_{\theta}(x)](http://ufldl.stanford.edu/wiki/images/math/8/8/7/887e72d0a7b7eb5083120e23a909a554.png) 形式如下：

   ![\begin{align}h_\theta(x^{(i)}) =\begin{bmatrix}p(y^{(i)} = 1 | x^{(i)}; \theta) \\p(y^{(i)} = 2 | x^{(i)}; \theta) \\\vdots \\p(y^{(i)} = k | x^{(i)}; \theta)\end{bmatrix}=\frac{1}{ \sum_{j=1}^{k}{e^{ \theta_j^T x^{(i)} }} }\begin{bmatrix}e^{ \theta_1^T x^{(i)} } \\e^{ \theta_2^T x^{(i)} } \\\vdots \\e^{ \theta_k^T x^{(i)} } \\\end{bmatrix}\end{align}](http://ufldl.stanford.edu/wiki/images/math/a/1/b/a1b0d7b40fe624cd8a24354792223a9d.png)

   ​	代价函数（Cost Function）为：

   ![\begin{align}J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{k}  1\left\{y^{(i)} = j\right\} \log \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)} }}\right]\end{align}](http://ufldl.stanford.edu/wiki/images/math/7/6/3/7634eb3b08dc003aa4591a95824d4fbd.png)

   ​	梯度公式（Gradient Descent）为：

   ![\begin{align}\nabla_{\theta_j} J(\theta) = - \frac{1}{m} \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = j\}  - p(y^{(i)} = j | x^{(i)}; \theta) \right) \right]  }\end{align}](http://ufldl.stanford.edu/wiki/images/math/5/9/e/59ef406cef112eb75e54808b560587c9.png)

2. 占位符

   ​	首先，通过为输入图像和目标输出类别创建节点，来构建计算图。

   ``` python
   x = tf.placeholder("float", shape=[None, 784])
   y_ = tf.placeholder("float", shape=[None, 10])
   ```

   ​	这里的 x 和 y_ 只是占位符，可以在TensorFlow运行某一计算时根据该占位符输入具体的值。

   ​	输入图片 `x` 是一个2维的浮点数张量。这里，分配给它的 shape 为`[None, 784]`，其中 `784  `是一张展平的图片的维度。`None` 表示其值大小不定，在这里作为第一个维度值，用以指代 batch 的大小，即 `x`的数量不定。输出类别值 `y_` 也是一个2维张量，其中每一行为一个10维的 one-hot 向量,用于代表对应某一MNIST图片的类别。

3. 变量

   ​	定义模型的权重 W 和偏移量 b。

   ```python
   W = tf.Variable(tf.zeros([784,10]))
   b = tf.Variable(tf.zeros([10]))
   ```

   ​	在调用 `tf.Variable` 时将 `W` 和 `b` 初始化为零向量，`W` 是一个784x10的矩阵（对应784个特征和10个输出值）。`b`是一个10维的向量（对应10个分类）。 

4. 类别预测与损失函数

   ​	把向量化后的图片 `x` 和权重矩阵 `W` 相乘，加上偏置 `b` ，然后计算每个分类的 softmax 概率值。

   ```python
   y = tf.nn.softmax(tf.matmul(x,W) + b)
   ```

   ​	损失函数是目标类别和预测类别之间的交叉熵。

   ```python
   cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
   ```

#### 4.2.3 构建一个多层卷积神经网络

1. 权重初始化

   ​	模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。由于这里使用的是 ReLU 神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。

   ```python
   def weight_variable(shape):
     initial = tf.truncated_normal(shape, stddev=0.1)
     return tf.Variable(initial)

   def bias_variable(shape):
     initial = tf.constant(0.1, shape=shape)
     return tf.Variable(initial)
   ```

2. 卷积和池化

   ​	卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。池化用2x2大小的模板做 max pooling。

   ```python
   def conv2d(x, W):
     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

   def max_pool_2x2(x):
     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
   ```

3. 第一层卷积

   ​	由一个卷积接一个 max pooling 完成，卷积在每个 5x5 的 patch 中算出 32 个特征。卷积的权重张量形状是 `[5, 5, 1, 32]`，前两个维度是 patch 的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。

   ```python
   W_conv1 = weight_variable([5, 5, 1, 32])
   b_conv1 = bias_variable([32])
   ```

   ​	为了用这一层，把 `x` 变成一个 4 维向量，其第 2、第 3 维分别对应图片的宽、高，第 4 维代表图片的颜色通道数(灰度图的通道数为1，RGB 彩色图为3)。

   ```python
   x_image = tf.reshape(x, [-1,28,28,1])
   ```

   ​	然后把 `x_image` 和权值向量进行卷积，加上偏置项，然后应用 ReLU 激活函数，最后进行 max pooling。

   ```python
   h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
   h_pool1 = max_pool_2x2(h_conv1)
   ```

4. 第二层卷积

   ​	为了构建一个更深的网络，可以把几个类似的层堆叠起来。第二层中，每个 5x5 的 patch 会得到 64 个特征。

   ```python
   W_conv2 = weight_variable([5, 5, 32, 64])
   b_conv2 = bias_variable([64])

   h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
   h_pool2 = max_pool_2x2(h_conv2)
   ```

5. 密集连接层

   ​	现在，图片尺寸减小到7x7，在这里加入一个有1024个神经元的全连接层，用于处理整个图片。然后把池化层输出的张量 reshape 成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。

   ```python
   W_fc1 = weight_variable([7 * 7 * 64, 1024])
   b_fc1 = bias_variable([1024])

   h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
   h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
   ```

   ​	为了减少过拟合，在输出层之前加入dropout。用一个 `placeholder` 来代表一个神经元的输出在 dropout 中保持不变的概率。这样我们可以在训练过程中启用 dropout ，在测试过程中关闭 dropout 。 TensorFlow的 `tf.nn.dropout` 操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。

   ```python
   keep_prob = tf.placeholder("float")
   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
   ```

6. 输出层

   ​	最后，添加一个softmax层，就像前面的单层 softmax regression 一样。

   ```python
   W_fc2 = weight_variable([1024, 10])
   b_fc2 = bias_variable([10])

   y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
   ```

#### 4.2.4 训练和评估模型

	1. 训练模型

​	用 ADAM 优化器来做最速下降法让交叉熵下降，步长为 $1e-4$。在 `feed_dict` 中加入额外的参数 `keep_prob` 来控制dropout比例。

```python
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```

​	返回的 `train_step` 操作对象，在运行时会使用梯度下降来更新参数。因此，整个模型的训练可以通过反复地运行 `train_step` 来完成。

```python
for i in range(20000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
```

​	每一步迭代，会加载50个训练样本，然后执行一次 `train_step` ，并通过 `feed_dict` 将 `x`  和  `y_` 张量占位符用训练训练数据替代。

2. 评估模型

   ​	首先找出预测正确的标签。`tf.argmax` 函数能给出某个 tensor 对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签， `tf.argmax(y_,1)` 代表正确的标签。然后，可以用 `tf.equal` 来检测预测值是否真实标签匹配。

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```



### 4.3 测试样例

## 第五章 总结与展望

