import tensorflow as tf

benchmark_name = 'AddingTwoNumbers4'
input_count = 2
output_count = 1

tf.reset_default_graph()

seed = 0
tf.random.set_random_seed(seed)

a = tf.placeholder(tf.float64, [None, 1], name= 'A_node')
b = tf.placeholder(tf.float64, [None, 1], name= 'B_node')
c = tf.add(a, b)
prediction = tf.identity(c, name = "prediction_node")

init = tf.global_variables_initializer()
sess = tf.Session()

finalize = sess.graph.finalize()

sess.run(init)



train_writer = tf.summary.FileWriter('myGraph', sess.graph)
train_writer.close()


with open('../../../resources/tfModel_linux/graph_' + benchmark_name + '.pb', 'wb') as f:
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())


# builder = tf.saved_model.builder.SavedModelBuilder("C:/Users/Snurka/init_model")
# builder.add_meta_graph_and_variables(
#   sess,
#   [tf.saved_model.tag_constants.SERVING]
# )
# builder.save()

