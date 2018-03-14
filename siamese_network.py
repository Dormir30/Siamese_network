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
xL = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
# 出力層の型
y_ = tf.placeholder(dtype=tf.int32, shape=[None,2])
one_hot_y = tf.one_hot(y_,depth=2)

# 出力層の型
yL_ = tf.placeholder(dtype=tf.int32, shape=[None,2])
one_hot_yL = tf.one_hot(yL_,depth=2)

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

# もう１つの第一層の畳み込み
xL_image = tf.reshape(xL, [-1, 28, 28, 1])
hL_conv1 = tf.nn.relu(conv2d(xL_image, W_conv1) + b_conv1)
hL_pool1 = max_pool_2x2(hL_conv1)

# もう１つの第二層の畳み込み
hL_conv2 = tf.nn.relu(conv2d(hL_pool1, W_conv2) + b_conv2)
hL_pool2 = max_pool_2x2(hL_conv2)

# もう１つの第三層の畳み込み
hL_conv3 = tf.nn.relu(conv2d(hL_pool2, W_conv3) + b_conv3)
hL_pool3 = max_pool_2x2(hL_conv3)


# 全結合層の設定
WL_fc1 = W_fc1
bL_fc1 = b_fc1
hL_pool1_flat = tf.reshape(hL_pool3, [-1,512])
hL_fc1 = tf.nn.relu(tf.matmul(hL_pool1_flat, WL_fc1) + bL_fc1)
hL_fc1_drop = tf.nn.dropout(hL_fc1, keep_prob)

# Distance
diss_l =tf.abs(tf.subtract(hL_fc1_drop,h_fc1_drop))

# Softmax層
WB_fc1 = weight_variable([32, 1])
y_conv = tf.nn.sigmoid(tf.matmul(diss_l, WB_fc1))

# 評価関数および最適化方法の設定
# ハイパーパラメタの定義
EPOCHS = 300
BATCH_SIZE = 1
Optrate = 0.001

sum_acc =tf.reduce_mean(tf.abs(tf.subtract(y_conv,tf.cast(tf.argmax(y_,1),"float32"))))
cross_entropy = -tf.reduce_sum(diss * tf.log(tf.cast(y_conv,dtype=tf.float32)+(1e-7)))+sum_acc+tf.reduce_sum(tf.abs(WB_fc1))

train_step = tf.train.AdamOptimizer(Optrate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


start = time.time()

# minibatch実行
for i in range(EPOCHS):
    X_train, y_train = shuffle(X['train'], y['train'])
    XL_train, yL_train = shuffle(X['train'], y['train'])

    # 0かどうかの判定
    dummy_0 = np.asarray([1.0,0.0]*2)
    dummy_0 = dummy_0.reshape(2,2)
    img_0_dummy = np.asarray(list(rep_0_img)  * 2)
    img_0_dummy = img_0_dummy.reshape(2,784)
    distance_labels = [np.sum(x) for x in dummy_0]

    eval_acc = sess.run(y_conv, feed_dict={x: X_train, y_: y_train,keep_prob:1,xL: img_0_dummy, yL_: dummy_0, diss:distance_labels})

    # 1かどうかの判定
    dummy_1 = np.asarray([0.0,1.0]*len(X_train))
    dummy_1 = dummy_0.reshape(len(X_train),2)
    img_1_dummy = np.asarray(list(rep_1_img)  * 2)
    img_1_dummy = img_1_dummy.reshape(2,784)
    distance_labels = [(1-np.sum(x)) for x in dummy_1]

    eval_1acc = sess.run(y_conv, feed_dict={x: X_train, y_: y_train,keep_prob:1,xL: img_1_dummy, yL_: dummy_1, diss:distance_labels})

    matome = np.concatenate([eval_acc,eval_1acc], axis=1)
    predict_result = np.argmin(matome, axis=1)


    temp_y_right = np.asarray(y_train)
    temp_y_right = np.argmax(temp_y_right, axis=1)
    acc_result = np.equal(temp_y_right,predict_result)
    tmp = np.sum(acc_result)/len(acc_result)
    print('EPOCHS Step %d' % i)
    print('Training Accuracy: %f' % tmp)

    dummy_0 = np.asarray([1.0,0.0]*len(y['test']))
    dummy_0 = dummy_0.reshape(len(y['test']),2)
    img_0_dummy = np.asarray(list(rep_0_img)  * len(y['test']))
    img_0_dummy = img_0_dummy.reshape(len(y['test']),784)
    distance_labels = [np.sum(x) for x in dummy_0]

    eval_acc = sess.run(y_conv, feed_dict={x: X['test'], y_: y['test'],keep_prob:1,xL: img_0_dummy, yL_: dummy_0, diss:distance_labels})


    dummy_1 = np.asarray([0.0,1.0]*len(X['test']))
    dummy_1 = dummy_0.reshape(len(X['test']),2)
    img_1_dummy = np.asarray(list(rep_1_img)  * len(X['test']))
    img_1_dummy = img_1_dummy.reshape(len(X['test']),784)
    distance_labels = [(1-np.sum(x)) for x in dummy_1]

    eval_1acc = sess.run(y_conv, feed_dict={x: X['test'], y_: y['test'],keep_prob:1,xL: img_1_dummy, yL_: dummy_1, diss:distance_labels})


    matome = np.concatenate([eval_acc,eval_1acc], axis=1)
    predict_result = np.argmin(matome, axis=1)

    temp_y_right = np.asarray(y['test'])
    temp_y_right = np.argmax(temp_y_right, axis=1)
    acc_result = np.equal(temp_y_right,predict_result)
    tmp = np.sum(acc_result)/len(acc_result)
    tmp_train = tmp
    print('TestAccuracy: %f' % tmp)


    for OFFSET in range(0, len(X['train']), BATCH_SIZE):
        batch_x, batch_y = X_train[OFFSET:(OFFSET + BATCH_SIZE)], y_train[OFFSET:(OFFSET + BATCH_SIZE)]
        batch_xL, batch_yL = XL_train[OFFSET:(OFFSET + BATCH_SIZE)], yL_train[OFFSET:(OFFSET + BATCH_SIZE)]
        simi = np.asarray(batch_y) * np.asarray(batch_yL)
        distance_labels = [(1 - np.sum(x)) for x in simi]

        train_step.run(
            feed_dict={x: batch_x, y_: batch_y,keep_prob:0.5,xL: batch_xL, yL_:batch_yL,diss:distance_labels}

        )

saver.save(sess, "./model.ckpt")
print('Final Training Accuracy %f' % tmp_train)

Elapsed_time = time.time() - start


# テストデータで今回のモデルを最終評価　指標は精度
dummy_0 = np.asarray([1.0,0.0]*len(y['test']))
dummy_0 = dummy_0.reshape(len(y['test']),2)
img_0_dummy = np.asarray(list(rep_0_img)  * len(y['test']))
img_0_dummy = img_0_dummy.reshape(len(y['test']),784)
distance_labels = [np.sum(x) for x in dummy_0]
eval_acc = sess.run(y_conv, feed_dict={x: X['test'], y_: y['test'],keep_prob:1,xL: img_0_dummy, yL_: dummy_0, diss:distance_labels})


dummy_1 = np.asarray([0.0,1.0]*len(X['test']))
dummy_1 = dummy_0.reshape(len(X['test']),2)
img_1_dummy = np.asarray(list(rep_1_img)  * len(X['test']))
img_1_dummy = img_1_dummy.reshape(len(X['test']),784)
distance_labels = [(1-np.sum(x)) for x in dummy_1]
eval_1acc = sess.run(y_conv, feed_dict={x: X['test'], y_: y['test'],keep_prob:1,xL: img_1_dummy, yL_: dummy_1, diss:distance_labels})

matome = np.concatenate([eval_acc,eval_1acc], axis=1)
predict_result = np.argmin(matome, axis=1)

temp_y_right = np.asarray(y['test'])
temp_y_right = np.argmax(temp_y_right, axis=1)
acc_result = np.equal(temp_y_right,predict_result)
tmp = np.sum(acc_result)/len(acc_result)
print('Final: TestAccuracy: %f' % tmp)
print('Time : %f' % Elapsed_time)
