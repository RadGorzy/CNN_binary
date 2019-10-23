#Clasify binary images

import tensorflow as tf
import math
import numpy
import Create_Dataset as Data
tf.app.flags.DEFINE_integer('number_of_classes', 5,
                            'Number of classes (among which we will classify images) in taining and test dataset')
RESTORE=False
FLAGS = tf.app.flags.FLAGS
print("Learning {} classes".format(str(FLAGS.number_of_classes)))

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)
batch_size_train=100
train_dataset,size = Data.read_data_set("./data/TFRecords/train")# "/media/radek/SanDiskSSD/SemanticKITTI/TFRecords/train"
#train_dataset=train_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))# batch - number of images multiplied at once, repeat() - it will continue infinitely (there will never be OutOfRangeError)
train_dataset=train_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
train_dataset = train_dataset.map(Data.edit,num_parallel_calls=4)
train_dataset=train_dataset.batch(batch_size_train)
#train_dataset=train_dataset.prefetch(batch_size_train)

test_dataset,size_test=Data.read_data_set("./data/TFRecords/test")# "/media/radek/SanDiskSSD/SemanticKITTI/TFRecords/validation"
test_dataset=test_dataset.shuffle(buffer_size=10000)
test_dataset=test_dataset.map(Data.edit,num_parallel_calls=4)
test_dataset=test_dataset.batch(100)
#test_dataset=test_dataset.prefetch(100)

print("Train dataset size = "+str(size))
print("Test dataset size = "+str(size_test))

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                              train_dataset.output_shapes)

next_element = iterator.get_next()


# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
#mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
#print('train size: '+str(mnist.train.images.shape[0]))

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X  [batch, 299, 299, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1=>6 stride 1        W1 [6, 6, 1, 6]        B1 [6]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 299, 299, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6=>12 stride 2       W2 [5, 5, 6, 12]       B2 [12]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 150, 150, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12=>24 stride 2      W3 [4, 4, 12, 24]      B3 [24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 75, 75, 24]
#      @ @ @ @ @        -- conv. layer 3x3x24=>48 stride 2      W4 [3, 3, 24, 48]      B3 [48]
#      ∶∶∶∶∶∶∶∶∶                                                Y4 [batch, 38, 38, 48]
#       @ @ @ @         -- conv. layer 3x3x48=>96 stride 2      W5 [3, 3, 48, 96]      B3 [96]
#       ∶∶∶∶∶∶∶                                                 Y5 [batch, 19, 19, 96] => reshaped to YY [batch, 19*19*96]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout) W6 [19*19*96, 200]     B4 [200]
#       · · · ·                                                 Y6 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W7 [200, 5]            B5 [5]
#        · · ·                                                  Y  [batch, 5]
sess = tf.InteractiveSession()
#if RESTORE==True:
#    saver = tf.train.import_meta_graph('./checkpoints/-7000.meta')#./checkpoints/PATH_TO_LATEST_CHECKPOINT.meta

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
#X = tf.placeholder(tf.float32, [None, 299, 299, 1])
# correct answers will go here
#Y_ = tf.placeholder(tf.float32, [None, 4])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 48
O = 96
P = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1), name='W1')  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name='B1')
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1), name='W2')
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]), name='B2')
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1), name='W3')
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]), name='B3')
W4 = tf.Variable(tf.truncated_normal([3, 3, M, N], stddev=0.1), name='W4')
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]), name='B4')
W5 = tf.Variable(tf.truncated_normal([3, 3, N, O], stddev=0.1), name='W5')
B5 = tf.Variable(tf.constant(0.1, tf.float32, [O]), name='B5')

W6 = tf.Variable(tf.truncated_normal([19 * 19 * O, P], stddev=0.1), name='W6')
B6 = tf.Variable(tf.constant(0.1, tf.float32, [P]), name='B6')
W7 = tf.Variable(tf.truncated_normal([P, FLAGS.number_of_classes], stddev=0.1), name='W7')
B7 = tf.Variable(tf.constant(0.1, tf.float32, [FLAGS.number_of_classes]), name='B7')

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(next_element[0], W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
stride = 2
Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='SAME') + B4)
stride = 2
Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='SAME') + B5)

# reshape the output from the third convolution for the fully connected layer

YY = tf.reshape(Y5, shape=[-1, 19 * 19 * O])

Y6 = tf.nn.relu(tf.matmul(YY, W6) + B6)
YY6 = tf.nn.dropout(Y6, pkeep)
Ylogits = tf.matmul(YY6, W7) + B7
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=next_element[4])
#Also add this line to bring the test and training cross-entropy to the same scale for display - only for visualization -for exapmple if we read 100 images in train batch and 10000 images in test then this factor should be 100 (100*100) :
cross_entropy = tf.reduce_mean(cross_entropy)*100
# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(next_element[4], 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#for test dataset we calculate accuracy based on below function, because this way we can calculate this accuracy splitting test dataset into batches instead loading it at once (what caused OutOfMemory Error)
total_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
total_accuracy=0
# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1]), tf.reshape(W6, [-1]),tf.reshape(W6, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1]), tf.reshape(B6, [-1]),tf.reshape(B6, [-1])], 0)


init_op = tf.global_variables_initializer()


#https://www.tensorflow.org/api_docs/python/tf/train/import_meta_graph
#Later we can continue training from this saved meta_graph without building the model from scratch.
# Create a saver, for saving checkpoints (https://stackoverflow.com/questions/46917588/restoring-a-tensorflow-model-that-uses-iterators) and at the end whole graph:
# Build the iterator SaveableObject.
saveable_obj = tf.contrib.data.make_saveable_from_iterator(iterator)
# Add the SaveableObject to the SAVEABLE_OBJECTS collection so
# it can be automatically saved using Saver.
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)
saver = tf.train.Saver(max_to_keep=2)
# Remember the training_op we want to run by adding it to a collection.
tf.add_to_collection('train_step', train_step)
tf.add_to_collection('all_weights',allweights)
tf.add_to_collection('all_biasses',allbiases)


#for visualization in tensorboard, merge all the summaries
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

model_version=tf.as_string([tf.cast(W1.shape,tf.float32),tf.cast(W2.shape,tf.float32),tf.cast(W3.shape,tf.float32),tf.cast(W4.shape,tf.float32),tf.cast(W5.shape,tf.float32),tf.concat([tf.cast(W6.shape,tf.float32),[0,0]],0),tf.concat([tf.cast(W7.shape,tf.float32),[0,0]],0)])
model_version = tf.summary.text("Model version", model_version)


train_writer = tf.summary.FileWriter('tensorboard_train',sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_test')

#at the beginning add info about model structure
test_writer.add_summary(model_version.eval())
train_writer.add_summary(model_version.eval())

sess.run(init_op)  #initilize all variables


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data,save_checkpoint):

    # training on batches of 100 images with 100 labels
    #batch_X, pre_height, pre_width, pre_filename, batch_Y = sess.run(next_element)

    # learning rate decay
    max_learning_rate = 0.003  #0.003
    min_learning_rate = 0.0001 #0.0001
    decay_speed = 2000.0 #2000
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)



    # compute test values for visualisation # liczymy tutaj dokladnosc wykorzystujac funkcje tf.reduce_sum (total_correct), dzieki czemu mozemy zbior testowy podzielic na batche mieszczace sie w pamiecie (brak OOM error)
    if update_test_data:
        sess.run(test_init_op)
        correct_sum = 0
        while True:
            try:
                batch_correct_count = sess.run(total_correct,feed_dict={pkeep: 1.0})
                correct_sum += batch_correct_count
            except tf.errors.OutOfRangeError: #after passing through whole test or validation dataset:
                total_accuracy = correct_sum / size_test
                break
        total_accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag="Test accuracy", simple_value=total_accuracy),])
        test_writer.add_summary(total_accuracy_summary,i)# add point for test visualization in Tenosrboard
        #summary, a, c = sess.run([merged, accuracy, cross_entropy], feed_dict={pkeep: 1.0})
        print(str(i) + ": ********* epoch " + str(i * batch_size_train // size + 1) + " ********* test accuracy:" + str(total_accuracy))# + " test loss: " + str(c))
        sess.run(training_init_op)  #WARNING tf.data.Iterator.from_structure together with make_initilizer works that way: each time we initiliaze it with test_init_op or
                                    # training_init_op, the data is loaded from the beginning -> it means that if we dint finish one epoch before calling another initilizer,
                                    #not all data will be processed
                                    #So it looks like if we want to pocess whole training dataset we can load test data only once per epoch (after whole training dataset was processed) !

    # compute training values for visualisation
    if update_train_data:
        #sess.run(training_init_op)
        summary, a, c, w, b = sess.run([merged, accuracy, cross_entropy,allweights, allbiases], feed_dict={pkeep: 1.0})
        train_writer.add_summary(summary,i)  # add point for train visualization in Tenosrboard
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    # the backpropagation training step
    sess.run(train_step, feed_dict={lr: learning_rate, pkeep: 0.75}) #0.75

    if save_checkpoint:
        saver.save(sess, './checkpoints/model.ckpt', global_step=i)


epochs=5
rounded_size=size - size % -batch_size_train #czyli rounded_size to jest zokraglony rozmiar zbioru trenujacego do nastepnej mozliwej do osiagniecia w petli wartosci, (np. size=96347, batch= 100, rounded_size=96400) ->po to aby spelnic warunek w parametrze funkcji training step -> (i*batch_size_train) % rounded_size == 0)
print("rounded size= "+str(rounded_size))
iterations=int((rounded_size/batch_size_train)*epochs)
print("Doing "+str(iterations)+" iterations, with batches of "+str(batch_size_train)+" which results in "+str(epochs)+" epochs.")
update_checkpoint_each_n_iterations=1000 #if you want to update checkpoint based on time you can use  saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=2)

startintgIteration=0
if RESTORE==True:
    print("RESTORING variables in order to continue training")
    meta_graph_path='./checkpoints/CNN_range_SemanticKITTI/model.ckpt-354000.meta'
    saver = tf.train.import_meta_graph(meta_graph_path)  # ./checkpoints/model.ckpt-7000.meta# ./checkpoints/PATH_TO_LATEST_CHECKPOINT.meta
    saver.restore(sess, save_path='./checkpoints/CNN_range_SemanticKITTI/model.ckpt-354000')#./checkpoints/model.ckpt-7000 #./checkpoints/PATH_TO_LATEST_CHECKPOINT  without .data-..... for ex. :"model.ckpt.data-00000-of-00001" then you should only use "model.ckpt" # you can also use: save_path=tf.train.latest_checkpoint(checkpoint_dir)
    startintgIteration=int(meta_graph_path.split(".")[-2].split("-")[-1])+1#ckpt number + 1
sess.run(training_init_op) #it only initializes dataset, so it gets only new images, but it doesnt reset other values as for ex. train step, model parameters
for i in range(startintgIteration,iterations+1):
    try:
        training_step(i, (i*batch_size_train) % rounded_size == 0, i % 20 == 0, i%update_checkpoint_each_n_iterations==0)# for i in range(9641): training_step(i, (i*batch_size_train) % rounded_size == 0, i % 20 == 0) #for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)
    except Exception as e:
        print("EXCEPTION:{} {}".format(i,e))
        break
        #image, height, width, filename, label_wec = sess.run(next_element)
        #print("Error while: {} {};{};{};{}={};{}".format(i, image.shape, height.shape, width.shape, filename.shape,
                                                         #filename, label_wec.shape))
#last graph and parameters save
save_path=saver.save(sess, './saved/my-model')
saver.export_meta_graph('./saved/my-model.meta')
print("Model saved in file: %s" % save_path)


sess.close()
