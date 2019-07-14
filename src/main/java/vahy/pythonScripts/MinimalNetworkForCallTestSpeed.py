import tensorflow as tf

benchmark_name = 'MinimalNetworkForCallTestSpeed'
input_count = 1
Q_output_count = 1
risk_output_count = 1
action_output_count = 1

output_count = Q_output_count + risk_output_count + action_output_count

tf.reset_default_graph()

seed = 0
tf.random.set_random_seed(seed)

x = tf.placeholder(tf.float64, [None, input_count], name= 'input_node')
target = tf.placeholder(tf.float64, [None, output_count], name = "target_node")

Q_target = tf.slice(target, [0, 0], [-1, Q_output_count], name = "Q_slice_node")
risk_target = tf.slice(target, [0, Q_output_count], [-1, risk_output_count], name = "risk_slice_node")
action_target = tf.slice(target, [0, Q_output_count + risk_output_count], [-1, action_output_count], name = "action_slice_node")

Q_output      = tf.layers.dense(x, Q_output_count,                   use_bias = True, kernel_initializer = tf.zeros_initializer, bias_initializer = tf.zeros_initializer, name = 'Q_output_node')
risk_output   = tf.layers.dense(x, risk_output_count, tf.nn.sigmoid, use_bias = True, kernel_initializer = tf.zeros_initializer, bias_initializer = tf.zeros_initializer, name = "risk_output_node")
action_output = tf.layers.dense(x, action_output_count, tf.nn.softmax, use_bias = True, kernel_initializer = tf.zeros_initializer, bias_initializer = tf.zeros_initializer, name = "action_output_node")

prediction = tf.concat([Q_output, risk_output, action_output], 1, name = "concat_node")
prediction_identity = tf.identity(prediction, name = "prediction_node")

Q_loss = tf.keras.losses.mean_squared_error(y_true = Q_target, y_pred = Q_output)
risk_loss = tf.keras.losses.binary_crossentropy(y_true = risk_target, y_pred = risk_output)
policy_loss = tf.keras.losses.categorical_crossentropy(y_true = action_target, y_pred = action_output)

total_loss = Q_loss + risk_loss + policy_loss
train_op = tf.train.AdamOptimizer(learning_rate = 0.01, name = "Optimizer").minimize(total_loss, name = 'optimize_node')

init = tf.global_variables_initializer()

sess = tf.Session()

train_writer = tf.summary.FileWriter('myGraph', sess.graph)
train_writer.close()

sess.run(init)

with open('../../../resources/tfModel/graph_' + benchmark_name + '.pb', 'wb') as f:
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())


# builder = tf.saved_model.builder.SavedModelBuilder("C:/Users/Snurka/init_model")
# builder.add_meta_graph_and_variables(
#   sess,
#   [tf.saved_model.tag_constants.SERVING]
# )
# builder.save()

