import tensorflow as tf

benchmark_name = 'XorNetwork'
input_count = 2
hidden_count_1 = 10
output_count = 1

tf.reset_default_graph()

Relu = tf.nn.relu
Tanh = tf.nn.tanh
BatchNormalization = tf.layers.batch_normalization
Dropout = tf.layers.dropout
Dense = tf.layers.dense

x = tf.placeholder(tf.float64, [None, input_count], name= 'input_node')
target = tf.placeholder(tf.float64, [None, output_count], name = "target_node")

hidden_1 = Dropout(Dense(x,        hidden_count_1, tf.nn.relu, use_bias = True, kernel_initializer = tf.glorot_normal_initializer(), name = "Hidden_1"))
output = tf.layers.dense(hidden_1, output_count, tf.nn.tanh, use_bias = True, kernel_initializer = tf.zeros_initializer, bias_initializer = tf.zeros_initializer, name ="output_node")

prediction = tf.identity(output, name = "prediction_node")

total_loss = tf.keras.losses.mean_squared_error(y_true = target, y_pred = output)
train_op = tf.train.AdamOptimizer(learning_rate = 0.01, name = "Adam").minimize(total_loss, name = 'optimize_node')

init = tf.global_variables_initializer()

sess = tf.Session()

train_writer = tf.summary.FileWriter('myGraph', sess.graph)
train_writer.close()

sess.run(init)


# n_samples = 0
# avg_cost = 0.
# for epoch in range(1000):
#     n_samples = n_samples + 1
#
#     batch_inputs  = [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]
#     batch_outputs = [[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]]
#
#     A_cost, B_cost, total_cost, _ = sess.run((A_loss, B_loss, total_loss, train_op), feed_dict= {x: batch_inputs, target: batch_outputs})
#
#     predictedVector = sess.run(prediction, feed_dict= {x: batch_inputs, target: batch_outputs})
#
#     np.set_printoptions(suppress=True)
#     print(predictedVector)
#
#     print("Epoch:", '%04d' % (epoch + 1), "total_cost = ", "{:.9f} ".format(sum(total_cost)), "A_cost = ", "{:.9f} ".format(sum(A_cost)), "B_cost = ", "{:.9f} ".format(sum(B_cost)))

#
#
#
# saver_def = tf.trainPolicy.Saver().as_saver_def()
#
# print('Operation to initialize variables:       ', init.name)
# print('Tensor to feed as input data:            ', x.name)
# print('Tensor to feed as training targets:      ', y_.name)
#
#
# print('Tensor to fetch as prediction:           ', y.name)
# print('Operation to trainPolicy one step:             ', train_op.name)
# print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
# print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
# print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
# print('Tensor to read value of W                ', W.value().name)
# print('Tensor to read value of b                ', b.value().name)

with open('../../../resources/tfModel/graph_' + benchmark_name + '.pb', 'wb') as f:
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())



# builder = tf.saved_model.builder.SavedModelBuilder("C:/Users/Snurka/init_model")
# builder.add_meta_graph_and_variables(
#   sess,
#   [tf.saved_model.tag_constants.SERVING]
# )
# builder.save()

