# TensorFlow-LinearRegression
TensorFlow实现线性回归

本文用TensorFlow实现一个线性回归预测模型，优化器为梯度下降。
划分了数据集、模型、损失函数以及优化器这几个作用域，方便在TensorBoard上观察。
首先导入`tensorflow`：
```
import tensorflow as tf
```

# 数据集
这里随机创建X集，然后通过X集计算出Y集，人为设定权重0.8，偏置1.0。
```
with tf.variable_scope("Data"):
    # 随机100条样本，每条样本一个特征值，值正态分布
    x = tf.random_normal(shape=[100, 1], mean=5.0, stddev=1.0, name="x_data")
    # 100条样子的目标值。人为设定权重0.8，偏置1.0
    y_true = tf.matmul(x, [[0.8]]) + 1.0
```

# 线性回归模型
```
with tf.variable_scope("Model"):
    # 权重初始值
    weight = tf.Variable(tf.random_normal(shape=[1, 1],
                                          mean=0.0,
                                          stddev=1.0,
                                          name="weight"))
    # 偏置初始值
    bais = tf.Variable(0.0, name="bais")
    # 预测值
    y_predict = tf.matmul(x, weight) + bais
```

# 损失函数
```
with tf.variable_scope("Loss"):
    # 均方误差
    loss = tf.reduce_mean(tf.square(y_true - y_predict))
```

# 优化器
```
with tf.variable_scope("Optimizer"):
    # 梯度下降优化器
    train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
```

# 张量收集
```
# 变量初始化器
var_init = tf.global_variables_initializer()

# 收集需要在board上显示的tensor
tf.summary.scalar(name="Loss", tensor=loss)
tf.summary.histogram(name="weight", values=weight)
summary_merged = tf.summary.merge_all()
```

# 运行
运行图，每10次训练打印一下权重值和偏置值，然后保存到文件：
```
with tf.Session() as sess:
    # 建立事件文件
    fw = tf.summary.FileWriter(logdir="summary", graph=sess.graph)

    sess.run(var_init)
    print("初始化 权重：{} 偏置：{}".format(weight.eval(), bais.eval()))

    for i in range(3000):
        # 训练
        sess.run(train)

        if i % 10 == 0:
            # 打印
            print("第{}次优化，权重：{} 偏置：{}".format(i, weight.eval(), bais.eval()))
            # 保存参数变化
            fw.add_summary(summary=sess.run(summary_merged), global_step=i)
```

打印输出：
```
初始化 权重：[[0.0822788]] 偏置：0.0
第0次优化，权重：[[1.4564301]] 偏置：0.27087461948394775
第10次优化，权重：[[0.95395017]] 偏置：0.19619160890579224
第20次优化，权重：[[0.95504194]] 偏置：0.2137095034122467
第30次优化，权重：[[0.95406306]] 偏置：0.23227156698703766
第40次优化，权重：[[0.9447488]] 偏置：0.24814006686210632
第50次优化，权重：[[0.9456911]] 偏置：0.26607203483581543

...

第2960次优化，权重：[[0.80022585]] 偏置：0.9988850355148315
第2970次优化，权重：[[0.80020916]] 偏置：0.9989074468612671
第2980次优化，权重：[[0.8002051]] 偏置：0.9989326596260071
第2990次优化，权重：[[0.8002005]] 偏置：0.9989558458328247
```

经过3000次梯度下降优化后，权重与偏置都非常接近我们预先设定的值0.8和1.0.

# TensorBoard

运行图：
![Graph](https://upload-images.jianshu.io/upload_images/2419179-018adb807a65e149.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)

损失函数值变化：
![Loss](https://upload-images.jianshu.io/upload_images/2419179-9560be61b7a54e44.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/800)

权重变化：
![Weight](https://upload-images.jianshu.io/upload_images/2419179-3ebe9b0c33ee0300.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/800)

本次运行，大概在第1500次优化后，得到比较稳定的参数值。