# -*- coding: utf-8 -*-

siamese_config = {
    'train_path': 'data/train.txt',   #Percentage of the training data to use for validation
    'test_file': 'data/test.txt',
    'vocab_path': 'data/vocab.txt',  # 数据路径  格式：标签\t文本
    'LIMIT_RATE': 0.8,  # 数据路径  格式：标签\t文本
    'embedding_size': 300,
    'hidden_units': 50,
    'batch_size': 100,
    'num_epochs': 5,
    'l2_reg_lambda': 0.01,
    'learning_rate': 1e-3,
    'dropout_keep_prob': 0.5,
    'evaluate_every': 10,  # 每隔多少步打印一次验证集结果
    'checkpoint_every': 1000,  # 每隔多少步保存一次模型
    'num_checkpoints': 5,  # 最多保存模型的个数
    'allow_soft_placement': True,  # 是否允许程序自动选择备用device
    'log_device_placement': False,  # 是否允许在终端打印日志文件
    'max_length': 16,
    'checkpoint_dir': 'runs/1592472528'

}
