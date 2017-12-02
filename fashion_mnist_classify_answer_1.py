'''
@version: 0.0.1
@author: yuhuan
@Contact: redaihanyu@126.com
@site: 
@file: fashion_mnist_classify_answer_1.py
@time: 2017/11/30 下午5:31

'''



'''
【问题】
1. 基于 Tensorflow 实现图像的分类模型，
尝试在 fashion-mnist 数据集(https://github.com/zalandoresearch/fashion-mnist)上的到最好的结果。

-------------------------------------------------------------

【回答】
答案说明：
    1、为了输出的结果一致，产生随机数的时候设置随机种子
    2、2层卷积，最大值池化，全连接层连接Dropout层，最后连接Softmax层进行输出
    3、随着训练的进行，learning rate会慢慢变小
    4、L2正则化

-------------------------------------------------------------

【结果说明】
    (epochs = 4001， batch_size = 100， 学习率(初始1e-3，每batch次衰减4%), L2正则(lambda = 0.0001, softmax层lambda = 0))
    Fashion准确率为：0.91+      程序运行时间:  0:21:18.173482
    
    MNIST准确率为：0.99+(无正则化和衰减学习率)
    
    理论上可以通过数据增强(Data Augmenttation)增加训练数据，可以给单幅图片增加多个副本，
    提高图片的利用率，防止对某一张图片的学习过拟合。
    通过对图片添加噪声，提高模型的泛化能力。
'''



from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import datetime

begin_time = datetime.datetime.now()

# mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
mnist = input_data.read_data_sets('./data/fashion', one_hot=True)
sess = tf.InteractiveSession()

# 为了输出的结果一致，产生随机数的时候设置随机种子
# L2正则化
def weight_variable(shape, lambda_, seed):
    var = tf.Variable(tf.truncated_normal(shape, stddev=0.1, seed=seed))
    if lambda_ is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), lambda_, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
global_epochs = 4001
batch_size = 100
# 将照片的结构转化为28 * 28的形式  格式为[batch_size, in_height, in_width, in_channel]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一个卷积层
# 卷积核格式[filter_height, filter_width, in_channel, out_filters]
W_conv1 = weight_variable([5, 5, 1, 32], seed=1, lambda_=0.00001)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64], seed=2, lambda_=0.00001)
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024], seed=3, lambda_=0.00001)
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减轻过拟合，加入Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层，Softmax层
W_fc2 = weight_variable([1024, 10], seed=4, lambda_=0.0)
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
tf.add_to_collection('losses', cross_entropy)
loss = tf.add_n(tf.get_collection('losses'))
# 学习率
global_ = tf.Variable(tf.constant(0))
learning_rate = tf.train.exponential_decay(1e-3, global_, batch_size, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 效果输出
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()
for i in range(global_epochs):
    batch = mnist.train.next_batch(100)
    if i % batch_size == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g\tlearning rate = %g" %
              (i, train_accuracy, sess.run(learning_rate, feed_dict={global_: i, x: batch[0], y_: batch[1], keep_prob: 1.0})))
        # 提前停止训练
        # if train_accuracy >= 0.95:
        #     break
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("huan test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

end_time = datetime.datetime.now()

print("程序运行时间: ", end_time - begin_time)