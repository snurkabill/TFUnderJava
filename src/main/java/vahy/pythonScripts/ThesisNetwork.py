import tensorflow as tf

benchmark_name = 'ThesisNet'
input_count = 7
hidden_count_1 = 50
hidden_count_2 = 50
hidden_count_3 = 100
Q_output_count = 1
risk_output_count = 1
action_output_count = 3

output_count = Q_output_count + risk_output_count + action_output_count

tf.reset_default_graph()

Relu = tf.nn.relu
Tanh = tf.nn.tanh
BatchNormalization = tf.layers.batch_normalization
Dense = tf.layers.dense
Dropout = tf.nn.dropout


x = tf.placeholder(tf.float64, [None, input_count], name= 'input_node')
target = tf.placeholder(tf.float64, [None, output_count], name = "target_node")
keep_prob = tf.placeholder(tf.float64, [], name = "keep_prob_node")
learning_rate = tf.placeholder(tf.float64, [], name = "learning_rate_node")

Q_target = tf.slice(target, [0, 0], [-1, Q_output_count], name = "Q_slice_node")
risk_target = tf.slice(target, [0, Q_output_count], [-1, risk_output_count], name = "risk_slice_node")
action_target = tf.slice(target, [0, Q_output_count + risk_output_count], [-1, action_output_count], name = "action_slice_node")

hidden_1 = Dropout(Dense(x,        hidden_count_1, tf.nn.relu, use_bias = True, kernel_initializer = tf.glorot_normal_initializer(), name = "hidden_1"), keep_prob = keep_prob) #, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.0))
hidden_2 = Dropout(Dense(hidden_1, hidden_count_2, tf.nn.relu, use_bias = True, kernel_initializer = tf.glorot_normal_initializer(), name = "hidden_2"), keep_prob = keep_prob) #, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.0))
# hidden_3 = Dense(hidden_2, hidden_count_3, tf.nn.relu, use_bias = True, kernel_initializer = tf.glorot_normal_initializer(), name = "hidden_3") #, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.0))

Q_output      = tf.layers.dense(hidden_2, Q_output_count,                   use_bias = True, kernel_initializer = tf.zeros_initializer, bias_initializer = tf.zeros_initializer, name = 'Q_output_node')
risk_output   = tf.layers.dense(hidden_2, risk_output_count, tf.nn.sigmoid, use_bias = True, kernel_initializer = tf.zeros_initializer, bias_initializer = tf.zeros_initializer, name = "risk_output_node")
action_output = tf.layers.dense(hidden_2, action_output_count, tf.nn.softmax, use_bias = True, kernel_initializer = tf.zeros_initializer, bias_initializer = tf.zeros_initializer, name = "action_output_node")

prediction = tf.concat([Q_output, risk_output, action_output], 1, name = "concat_node")
prediction_identity = tf.identity(prediction, name = "prediction_node")

Q_loss = tf.keras.losses.mean_squared_error(y_true = Q_target, y_pred = Q_output)
risk_loss = tf.keras.losses.binary_crossentropy(y_true = risk_target, y_pred = risk_output)
policy_loss = tf.keras.losses.categorical_crossentropy(y_true = action_target, y_pred = action_output)

total_loss = Q_loss + risk_loss + policy_loss
train_op = tf.train.AdamOptimizer(learning_rate = learning_rate, name = "Optimizer").minimize(total_loss, name = 'optimize_node')

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

