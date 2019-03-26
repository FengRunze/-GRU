from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from gruModels.FirstDerivativeVDRNNCell import FirstVDRNNCell
from gruModels.FirstDerivativeVDRNNCell import FirstVDRNNCell
"""PARAMETERS"""
input_size = 2
time_step = 12
state_size = 128
output_size = 2

learning_rate = 0.001
batch_size = 128
train_epochs = 300

print(type(time_step))


"""READ DATA"""
train_data = pd.read_csv("task_one_2/train_data.csv", header=None)
train_label = pd.read_csv("task_one_2/train_label.csv", header=None)
validation_data = pd.read_csv("task_one_2/validation_data.csv", header=None)
validation_label = pd.read_csv("task_one_2/validation_label.csv", header=None)
test_data = pd.read_csv("task_one_2/test_data.csv", header=None)
test_label = pd.read_csv("task_one_2/test_label.csv", header=None)


train_data =train_data[0:150000]
train_label=train_label[0:150000]
validation_data = validation_data[0:80000]
validation_label =validation_label[0:80000]
test_data = test_data[0:80000]
test_label = test_label[0:80000]

print(train_data.shape)
print(train_label.shape)
print(validation_data.shape)
print(validation_label.shape)
print(test_data.shape)
print(test_label.shape)
"""RESHAPE DATA"""
train_data = train_data.values.reshape([-1, time_step, input_size])
train_label = train_label.values
validation_data = validation_data.values.reshape([-1, time_step, input_size])
validation_label = validation_label.values
test_data = test_data.values.reshape([-1, time_step, input_size])
test_label = test_label.values

"""MODEL"""
x = tf.placeholder("float32", [None, time_step, input_size])
y = tf.placeholder("float32", [None, output_size])

Wsy = tf.get_variable(name='Wsy', shape=[state_size, output_size],
                      initializer=tf.random_normal_initializer(mean=0, stddev=0.1))

By = tf.get_variable(name='By', shape=[output_size], initializer=tf.constant_initializer(0.1))

cell = FirstVDRNNCell(state_size)

#
# lstm_fw_cell = FirstVDRNNCell(state_size)
# # Backward direction cell
# lstm_bw_cell = FirstVDRNNCell(state_size)
# outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
#
#                                               dtype=tf.float32)
#
# cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=5, state_is_tuple=True)
state, tuple_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

final_state = tf.reduce_max(state,axis = 1)

# transposed_state = tf.transpose(state, [1, 0, 2])
# final_state = transposed_state[-1]
pred = tf.matmul(final_state, Wsy) + By

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

train_accuracy = []
validation_accuracy = []
test_accuracy = []

train_loss = []

for epoch in range(train_epochs):
    total_train_loss_val = 0
    for step in range(len(train_data) // batch_size + 1):
        min_index = step * batch_size
        max_index = min((step + 1) * batch_size, len(train_data))
        train_data_batch = train_data[min_index:max_index]
        train_label_batch = train_label[min_index:max_index]
        train_loss_val, _ = sess.run([loss, train_op], {x: train_data_batch, y: train_label_batch})
        total_train_loss_val += train_loss_val

    train_loss.append(total_train_loss_val)

    train_accuracy_val = sess.run(accuracy, {x: train_data, y: train_label})
    train_accuracy.append(train_accuracy_val)

    validation_accuracy_val = sess.run(accuracy, {x: validation_data, y: validation_label})
    validation_accuracy.append(validation_accuracy_val)

    test_accuracy_val = sess.run(accuracy, {x: test_data, y: test_label})
    test_accuracy.append(test_accuracy_val)

    print("Epoch: %d; TrainLoss: %.4f; ValidationAccuracy: %.4f; TestAccuracy: %.4f; " % (
        epoch, total_train_loss_val, validation_accuracy_val, test_accuracy_val))

plt.figure(figsize=(6, 6))

plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.plot(train_accuracy, color="blue", label="Train")
plt.plot(validation_accuracy, color="red", label="Validation")
plt.plot(test_accuracy, color="green", label="Test")
plt.legend(loc="lower right")
plt.show()


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss, color="black", label="Train")
plt.legend(loc="upper right")
plt.show()


print(train_loss)
print(validation_accuracy)
print(test_accuracy)
