import numpy as np
from keras.preprocessing.sequence import pad_sequences
import os
from collections import Counter
import json
from config import siamese_config as config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class InputHelper(object):
    def __inin__(self):
        pass
    '''根据样本长度,选择最佳的样本max-length'''
    def select_best_length(self, config):
        len_list = []
        max_length = 0
        cover_rate = 0.0
        for line in open(config["train_path"], encoding="utf-8"):
            line = line.strip().split('	')
            if not line:
                continue
            sent = line[0]
            sent_len = len(sent)
            len_list.append(sent_len)
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()  #输出出现的(元素,次数)对,且类型是list
        for i in len_dict:
            sum_length += i[1]*i[0]
        average_length = sum_length/all_sent
        for i in len_dict:
            rate = i[1]/all_sent
            cover_rate += rate
            if cover_rate >= config["LIMIT_RATE"]:
                max_length = i[0]
                break
        print('average_length:', average_length)
        print('max_length:', max_length)

        return max_length

    '''构造数据集'''
    def build_data(self, config, is_training=True):
        sample_x = []
        sample_y = []
        sample_x_left = []
        sample_x_right = []
        vocabs = {'UNK'}
        if is_training:
            path = config["train_path"]
        else:
            path = config["test_file"]
        for line in open(path, encoding="utf-8"):
            line = line.rstrip().split('\t')
            if not line:
                continue
            sent_left = line[0]
            sent_right = line[1]
            label = line[2]
            sample_x_left.append([char for char in sent_left if char])
            sample_x_right.append([char for char in sent_right if char])
            sample_y.append(label)
            for char in [char for char in sent_left + sent_right if char]:
                vocabs.add(char)
        print(len(sample_x_left), len(sample_x_right))
        sample_x = [sample_x_left, sample_x_right]

        datas = [sample_x, sample_y]
        if is_training:
            word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
            self.write_file(word_dict, config["vocab_path"])
        else:
            word_dict = ""
        return datas, word_dict

    '''将数据转换成tensorflow所需的格式'''
    def modify_data(self, datas, word_dict, maxlen, is_training=True):
        sample_x = datas[0]
        sample_y = datas[1]
        sample_x_left = sample_x[0]
        sample_x_right = sample_x[1]
        if is_training:
            left_x_train = [[word_dict[char] for char in data] for data in sample_x_left]
            right_x_train = [[word_dict[char] for char in data] for data in sample_x_right]
        else:
            left_x_train = [[word_dict.get(char, word_dict["UNK"]) for char in data] for data in sample_x_left]
            right_x_train = [[word_dict.get(char, word_dict["UNK"]) for char in data] for data in sample_x_right]
        y_train = [int(i) for i in sample_y]
        left_x_train = pad_sequences(left_x_train, maxlen=maxlen)
        right_x_train = pad_sequences(right_x_train, maxlen=maxlen)
        # y_train = np.expand_dims(y_train, 2)
        return left_x_train, right_x_train, y_train

    '''保存字典文件'''
    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+', encoding="utf-8") as f:
            f.write(str(wordlist))

    '''产生batch数据'''

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        # 每次只输出shuffled_data[start_index:end_index]这么多
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1  # 每一个epoch有多少个batch_size
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            # 每一代都清理数据
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size  # 当前batch的索引开始
                end_index = min((batch_num + 1) * batch_size, data_size)  # 判断下一个batch是不是超过最后一个数据了
                yield shuffled_data[start_index:end_index]
