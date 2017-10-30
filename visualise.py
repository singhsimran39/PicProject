import tensorflow as tf
import os
import glob
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

path = '/Stuff/Courses/Projects/CNN/PicProject/new_pic/'
all_pics = glob.glob(path+'s.jpg')
all_pics.sort()

pics = np.array([misc.imread(pic) for pic in all_pics]).astype(np.float64)
# pics -= 160.739
pics.shape = [1, 200, 200, 3]

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('picProject.meta')
	saver.restore(sess, 'picProject')

	# for op in tf.get_default_graph().get_operations():
	# 	print(str(op.name))
	#final_pred = sess.run('fc2:0', feed_dict={'in_pic:0':pics})
	#print(final_pred)
	first_conv_layer, first_pool_layer = sess.run(['h0/h0_conv:0', 'h0/h0_pool:0'], feed_dict={'in_pic:0':pics})

for i in range(0, 16):
	plt.subplot(4, 4, (i+1))
	plt.imshow(misc.imrotate(first_conv_layer[0,:,:,i], -90))
plt.suptitle('First Conv Layer')
plt.show()

for i in range(0, 16):
	plt.subplot(4, 4, (i+1))
	plt.imshow(misc.imrotate(first_pool_layer[0,:,:,i], -90))
plt.suptitle('First Pool Layer')
plt.show()
