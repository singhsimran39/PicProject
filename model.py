import tensorflow as tf
import glob
import numpy as np
from scipy import misc
from Dataset import Dataset

def conv_layer(input_, out_maps, filter_shape=[5, 5], filter_strides=[2, 2], pool_shape=[2, 2], pool_strides=[2, 2], name='conv2d'):
    with tf.variable_scope(name):
        weights = tf.get_variable(name=name+'_W', shape=[filter_shape[0], filter_shape[1], input_.shape[3], out_maps],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable(name=name+'_b', shape=[out_maps], initializer=tf.truncated_normal_initializer(stddev=0.02))

        conv = tf.nn.conv2d(input=input_, filter=weights, strides=[1, filter_strides[0], filter_strides[1], 1], padding='SAME')
        conv += biases
        conv = tf.nn.relu(conv, name=name+'_conv')

        pool = tf.nn.max_pool(value=conv, ksize=[1, pool_shape[0], pool_shape[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1],
                              padding='SAME', name=name+'_pool')

        return pool

def get_batches(batch_size, num_examples):
    if num_examples % batch_size == 0:
        return num_examples / batch_size
    else:
        return (num_examples / batch_size) + 1

class PicProject(object):

    def __init__(self, sess, batch_size, input_height, input_width, input_channels, fc1_dim, fc2_dim, out_dim, num_maps, epochs, learning_rate,
                 train_path, test_path, model_name):

        self.sess = sess
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.out_dim = out_dim
        self.num_maps = num_maps

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.train_path = train_path
        self.test_path = test_path
        self.model_name = model_name

        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(name='in_pic', shape=[None, self.input_height, self.input_width, self.input_channels], dtype=tf.float32)
        self.y = tf.placeholder(name='real_person', shape=[None, self.out_dim], dtype=tf.float32)

        self.preds, self.logits = self.cnn_network(self.x, self.y)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()


    def cnn_network(self, image, y):

        layer1 = conv_layer(input_=image, out_maps=self.num_maps, name='h0')
        layer2 = conv_layer(input_=layer1, out_maps=self.num_maps*2, name='h1')

        layer2_shape = layer2.get_shape().as_list()
        flattened = tf.reshape(layer2, [-1, layer2_shape[1]*layer2_shape[2]*layer2_shape[3]], name='l2_flat')

        w_fc1 = tf.get_variable(name='w_fc1', shape=[layer2_shape[1]*layer2_shape[2]*layer2_shape[3], self.fc1_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable(name='b_fc1', shape=[self.fc1_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))

        dense_layer1 = tf.matmul(flattened, w_fc1) + b_fc1
        dense_layer1 = tf.nn.relu(dense_layer1, name='fc1')

        w_fc2 = tf.get_variable(name='w_fc2', shape=[self.fc1_dim, 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable(name='b_fc2', shape=[2], initializer=tf.truncated_normal_initializer(stddev=0.02))

        dense_layer2 = tf.matmul(dense_layer1, w_fc2) + b_fc2
        preds = tf.nn.softmax(dense_layer2, name='fc2')

        return preds, dense_layer2

    def train(self):

        def get_images(path):
            all_pics = glob.glob(path+'*.jpg')
            all_pics.sort()
            y = np.zeros([len(all_pics), self.out_dim])
            i = 0

            for pic in all_pics:
                sw = pic.split(path)[1]
                if sw.startswith('S'):
                    y[i, 1] = 1
                else:
                    y[i, 0] = 1
                i += 1

            x = np.array([misc.imread(pic) for pic in all_pics])
            x = np.array(x).astype(np.float32)
            y = np.array(y).astype(np.float32)

            return x, y

        optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy, var_list=self.t_vars)
        correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.preds, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        tf.global_variables_initializer().run()

        train_x, train_y = get_images(self.train_path)
        test_x, test_y = get_images(self.test_path)

        train_x -= np.mean(train_x)
        test_x -= np.mean(test_x)
        print(np.mean(test_x))
        fulldata = Dataset(train_x, train_y)

        num_batches = get_batches(self.batch_size, train_x.shape[0])

        for epoch in range(self.epochs):
            avg_cost = 0

            for i in range(int(num_batches)):
                batch_x, batch_y = fulldata.next_batch(self.batch_size)
                _, c = self.sess.run([optim, self.cross_entropy], feed_dict={self.x:batch_x, self.y:batch_y})
                avg_cost += c / num_batches

            _, train_acc = self.sess.run([optim, accuracy], feed_dict={self.x:train_x, self.y:train_y})
            _, test_loss, test_acc = self.sess.run([optim, self.cross_entropy, accuracy], feed_dict={self.x:test_x, self.y:test_y})
            print('Epoch:', (epoch+1), 'TrainAccuracy:', '{:.4f}'.format(train_acc), 'TrainLoss:', '{:.6f}'.format(avg_cost),
                  'TestAccuracy:', '{:.4f}'.format(test_acc), 'TestLoss:', '{:.6f}'.format(test_loss))

        self.save()

    def save(self):
        self.saver.save(self.sess, self.model_name)
