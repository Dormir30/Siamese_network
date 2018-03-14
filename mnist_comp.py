# -*- coding: utf-8 -*-
from PIL import Image
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import traceback
from sklearn.utils import shuffle

X = {}
y = {}

train_img =[]
train_label=[]
test_img=[]
test_label=[]

# データの読み込み
for filename in glob.glob('./image/train/0/*.jpg'): 
    img=  Image.open(filename)
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.asarray(img)
    img = img.reshape(784)/255
    rep_0_img = img
    train_img.append(img)
    train_label.append([1.0,0.0])


for filename in glob.glob('./image/train/1/*.jpg'): 
    img=  Image.open(filename)
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.asarray(img)
    img = img.reshape(784)/255
    rep_1_img = img
    train_img.append(img)
    train_label.append([0.0,1.0])

for filename in glob.glob('./image/test/0/*.jpg'): 
    img=  Image.open(filename)
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.asarray(img)
    img = img.reshape(784)/255
    test_img.append(img)
    test_label.append([1.0,0.0])


for filename in glob.glob('./image/test/1/*.jpg'): 
    img=  Image.open(filename)
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.asarray(img)
    img = img.reshape(784)/255
    test_img.append(img)
    test_label.append([0.0,1.0])

  
print("Image_read_finished")    

X['train'] = train_img
y['train'] = train_label
X['test'] = test_img
y['test'] = test_label

# sessionの開始
sess = tf.InteractiveSession()

# 入力層の型
x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
# 出力層の型
y_ = tf.placeholder(dtype=tf.int32, shape=[None,2])
one_hot_y = tf.one_hot(y_,depth=2)


# 距離vectorパラメータを設定
diss = tf.placeholder(dtype=tf.float32, shape=[None])

# 畳み込みとプーリングの関数を設定しておく
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# 第一層の畳み込み
W_conv1 = weight_variable([4, 4, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二層の畳み込み
W_conv2 = weight_variable([5, 5, 16, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三層の畳み込み
W_conv3 = weight_variable([5, 5, 64, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


# 全結合層の設定
W_fc1 = weight_variable([512, 32])
b_fc1 = bias_variable([32])
h_pool1_flat = tf.reshape(h_pool3, [-1,512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Dropoutの設定
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax層
WB_fc1 = weight_variable([32, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, WB_fc1))+b_fc2

# 評価関数および最適化方法の設定
# ハイパーパラメタの定義
EPOCHS = 300
BATCH_SIZE = 1
Optrate = 0.001

sum_acc =tf.reduce_mean(tf.abs(tf.subtract(y_conv,tf.cast(tf.argmax(y_,1),"float32"))))
cross_entropy = -tf.reduce_sum(tf.log(tf.cast(y_conv,dtype=tf.float32)+(1e-7)))+sum_acc+tf.reduce_sum(tf.abs(WB_fc1))

train_step = tf.train.AdamOptimizer(Optrate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


start = time.time()

# バッチ実行
for i in range(EPOCHS):
    X_train, y_train = shuffle(X['train'], y['train'])

    eval_train_acc = sess.run(accuracy, feed_dict={x: X_train, y_: y_train,keep_prob:1})

    eval_test_acc = sess.run(accuracy, feed_dict={x: X['test'], y_: y['test'],keep_prob:1})
    print('TrainAccuracy: %f' % eval_train_acc)
    print('TestAccuracy: %f' % eval_test_acc)


    for OFFSET in range(0, len(X['train']), BATCH_SIZE):
        batch_x, batch_y = X_train[OFFSET:(OFFSET + BATCH_SIZE)], y_train[OFFSET:(OFFSET + BATCH_SIZE)]
        train_step.run(
            feed_dict={x: batch_x, y_: batch_y,keep_prob:0.5}

        )

saver.save(sess, "./model.ckpt")
print('Final Training Accuracy %f' % eval_train_acc)

Elapsed_time = time.time() - start

print('Final: TestAccuracy: %f' % eval_test_acc)
print('Time : %f' % Elapsed_time)
