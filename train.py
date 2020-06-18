import tensorflow as tf
import numpy as np
import os
import time
import datetime
from utils.dataHelper import InputHelper
from config import siamese_config as config
from model.siamese_model import SiameseLSTM
from sklearn import metrics

#定义训练函数
def train_step(left_x_batch, right_x_batch, labels_train_batch):
    """
    A single training step #单个训练步骤
    """
    #cnn feed数据
    feed_dict = {
      model.input_x1: left_x_batch,
      model.input_x2: right_x_batch,
      model.input_y: labels_train_batch,
      model.dropout_keep_prob: config["dropout_keep_prob"],
    }
    # y_label = np.argmax(labels_train_batch, 1)
    time_str = datetime.datetime.now().isoformat()#取当前时间，Python的函数
    # if num < 5:
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, model.loss, model.accuracy],
        feed_dict)
    # print(metrics.classification_report(y_label, prediction))
    print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))

    train_summary_writer.add_summary(summaries, step)

#定义测试函数
def dev_cnn_step(x_dev, labels_dev, writer=None):
    """
    Evaluates model on a dev set    #用测试集评估模型
    """

    feed_dict = {
      model.input_x: x_dev,
      model.input_y: labels_dev,
      model.dropout_keep_prob: 1.0,#神经元全部保留
      model.is_training: False
    }
    y_label = np.argmax(labels_dev, 1)

    _, step, summaries, loss, prediction = sess.run(
        [train_op, global_step, train_summary_op, model.loss, model.predictions],
        feed_dict)

    time_str = datetime.datetime.now().isoformat()
    print(metrics.classification_report(y_label, prediction))
    print("{}: step {}, loss {:g}".format(time_str, step, loss))
    if writer:
        writer.add_summary(summaries, step)
data_helper = InputHelper()
#选取最佳的len
maxlen = data_helper.select_best_length(config)
#构造词表，准备数据
datas, word_dict = data_helper.build_data(config, is_training=True)
vocab_size = len(word_dict)
#数据准备，加载数据
print("Loading data...")
left_x_train, right_x_train, y_train = data_helper.modify_data(datas, word_dict, maxlen, is_training=True)
print(left_x_train)
print(y_train)
# Training
#训练开始
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=config["allow_soft_placement"],
      log_device_placement=config["log_device_placement"])
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #siamese LSTM 导入
        model = SiameseLSTM(
            sequence_length=maxlen,
            vocab_size=vocab_size,
            embedding_size=config["embedding_size"],
            hidden_units=config["hidden_units"],
            batch_size=config["batch_size"],#一共有几个filter
            l2_reg_lambda=config["l2_reg_lambda"])#L2正则化项

        # Define Training procedure
        #定义训练程序

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config["learning_rate"])#定义优化器
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        #跟踪梯度值和稀疏即tensorboard
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        #模型和summaries的输出目录
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        #损失函数和准确率的参数保存
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        #训练数据保存
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        #测试数据保存
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=config["num_checkpoints"])
        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
        # vocab_processor_fast.save(os.path.join(out_dir, "vocab.pickle"))
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  #tf.metrics.accuracy会产生两个局部变量
        batches = data_helper.batch_iter(
            list(zip(left_x_train, right_x_train, y_train)), config["batch_size"], config["num_epochs"])
        for batch in batches:
            left_x_batch, right_x_batch, labels_train_batch = zip(*batch)#按batch把数据拿进来
            train_step(left_x_batch, right_x_batch, labels_train_batch)

            current_step = tf.train.global_step(sess, global_step)#将Session和global_step值传进来
            # if current_step % config["evaluate_every"] == 0: # config.evaluate_every次每100执行一次测试
            #     print("\nEvaluation:")
            #     dev_cnn_step(x_dev, labels_dev, writer=dev_summary_writer)

            if current_step % config["checkpoint_every"] == 0:# 每checkpoint_every次执行一次保存模型
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)# 定义模型保存路径
                print("Saved model checkpoint to {}\n".format(path))



