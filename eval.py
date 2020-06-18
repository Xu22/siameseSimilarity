import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
from sklearn import metrics
from utils.dataHelper import InputHelper
from config import siamese_config as config
from model.siamese_model import SiameseLSTM
import json
# CHANGE THIS: Load data. Load your own data here
data_helper = InputHelper()
if config["test_file"]:
    datas, _ = data_helper.build_data(config, is_training=False)
    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        word_dict = eval(f.read())
    #数据准备，加载数据
    print("Loading data...")

    left_x_train, right_x_train, y_train = data_helper.modify_data(datas, word_dict, config["max_length"], is_training=False)

else:
    print("please set a test_file path to test model")
    left_x_train, right_x_train, y_train = "", "", ""

# Map data into vocabulary
# print(config["checkpoint_dir"])
# vocab_path = os.path.join(config["checkpoint_dir"], "vocab.pickle")
#
# vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#
# x_test = np.array(list(vocab_processor.transform(x_text)))
# print(x_test)

print("\nEvaluating...\n")

# reset graph
tf.reset_default_graph()
graph_1 = tf.Graph()
with graph_1.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=config["allow_soft_placement"],
      log_device_placement=config["log_device_placement"])
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}/checkpoints/model-5000.meta".format(config["checkpoint_dir"]))

        saver.restore(sess, tf.train.latest_checkpoint("{}/checkpoints".format(config["checkpoint_dir"])))

        # Get the placeholders from the graph by name
        input_x1 = graph_1.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph_1.get_operation_by_name("input_x2").outputs[0]
        input_y = graph_1.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph_1.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        accuracy = graph_1.get_operation_by_name("accuracy/accuracy").outputs[0]

        # Generate batches for one epoch
        batches = data_helper.batch_iter(list(zip(left_x_train, right_x_train, y_train)), config["batch_size"], 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        scores_sum = []
        for batch in batches:
            left_x_train, right_x_train, y_train = zip(*batch)  # 按batch把数据拿进来
            accuracy = \
                sess.run(accuracy, {input_x1: left_x_train, input_x2: right_x_train, input_y: y_train, dropout_keep_prob: 1.0})
            # all_predictions = np.concatenate([all_predictions, accuracy])

        print(accuracy)
        # print(metrics.classification_report(y_test, all_predictions, digits=4))


