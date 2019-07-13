import tensorflow as tf

benchmark_name = 'XorNetwork'
input_count = 2
hidden_count_1 = 10
output_count = 1

tf.compat.v1.reset_default_graph()

Relu = tf.compat.v1.nn.relu
Tanh = tf.compat.v1.nn.tanh
BatchNormalization = tf.compat.v1.layers.batch_normalization
Dropout = tf.compat.v1.layers.dropout
Dense = tf.compat.v1.layers.dense

x = tf.compat.v1.placeholder(tf.float64, [None, input_count], name= 'input_node')
target = tf.compat.v1.placeholder(tf.float64, [None, output_count], name = "target_node")
hidden_1 = Dropout(Dense(x, hidden_count_1, tf.nn.relu, use_bias = True, kernel_initializer = tf.compat.v1.glorot_normal_initializer, name = "Hidden_1"))

output = tf.layers.dense(hidden_1, output_count, tf.nn.tanh, use_bias = True, kernel_initializer = tf.compat.v1.zeros_initializer, bias_initializer = tf.compat.v1.zeros_initializer, name ="output_node")

prediction = tf.compat.v1.identity(output, name = "prediction_node")

total_loss = tf.compat.v1.losses.mean_squared_error(labels = target, predictions = output)
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.01, name = "Adam").minimize(total_loss, name = 'optimize_node')

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()

train_writer = tf.compat.v1.summary.FileWriter('myGraph', sess.graph)
train_writer.close()

# sess.run(init)

with open('../../../resources/tfModel/graph_' + benchmark_name + '.pb', 'wb') as f:
    f.write(tf.compat.v1.get_default_graph().as_graph_def().SerializeToString())



# builder = tf.saved_model.builder.SavedModelBuilder("C:/Users/Snurka/init_model")
# builder.add_meta_graph_and_variables(
#   sess,
#   [tf.saved_model.tag_constants.SERVING]
# )
# builder.save()

