import os
import shutil

import tensorflow as tf

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

#import the data set data/hatespeech.csv
data_path = os.path.join(os.getcwd(), 'data/hatespeech.csv')
"""Loads the data set."""
# Load the data set
dataset = tf.data.experimental.CsvDataset(
    data_path, [tf.float32, tf.string], header=True)
# Split into train/eval
train_size = int(0.8 * dataset.cardinality())
train_dataset = dataset.skip(train_size)
eval_dataset = dataset.skip(train_size)
# Shuffle, repeat, and batch the examples.
train_dataset = train_dataset.shuffle(1000).repeat().batch(32)
eval_dataset = eval_dataset.shuffle(1000).repeat().batch(32)
print(train_dataset, eval_dataset)
