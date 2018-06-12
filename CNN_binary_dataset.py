#Clasify binary images

import tensorflow as tf
import math
import numpy
import Create_Dataset as Data


print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)
batch_size_train=100
train_dataset,size = Data.read_data_set("./data/TFRecords/train")                         #padded_batch(100,padded_shapes=([None],[None],[None],[None],[None])).
train_dataset = train_dataset.map(Data.edit)                                              #tu jeszcze jest argument num_parallel_calls=None, ktory oznacza iosc elementow przetwarzanych na raz
train_dataset=train_dataset.shuffle(buffer_size=1000).batch(batch_size_train).repeat()    # padded_batch -musze zastosowac taki rodzaj (a nie samo batch),bo tensory w dataset nie maja takich samych wymiarow - ilosc obrazow w jednej partii (wymnazancyh za jednym razem), repeat() - czyli bedzie powtarzac w nieskonsczonosc (nigdy nie bedzie OutOfRangeError)

test_dataset,size_test=Data.read_data_set("./data/TFRecords/test")
test_dataset=test_dataset.map(Data.edit)
test_dataset=test_dataset.shuffle(buffer_size=1000).batch(100) #tutaj batch powinno byc rowne ilosci wszystkich elementow w zbiorze testowym: 40142 #nie ma na koncu repeat(), zeby po przejsciu
                                                                #calego zbioru wyrzucil tf.errors.OutOfRangeError (co pozniej wykorzystuje)
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
#       \x/x\x/         -- fully connected layer (softmax)      W7 [200, 4]            B5 [4]
#        · · ·                                                  Y  [batch, 4]
sess = tf.InteractiveSession()


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
W7 = tf.Variable(tf.truncated_normal([P, 4], stddev=0.1), name='W7')
B7 = tf.Variable(tf.constant(0.1, tf.float32, [4]), name='B7')

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
cross_entropy = tf.reduce_mean(cross_entropy)*100
# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(next_element[4], 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Dla zbioru testowego liczymy dokladnosc na podstawie ponizszej funkcji, bo w ten sposob mozna liczyc dokladnosc dzielac zbior testowy na batche, a nie wczytujac caly (co powodowalo OutOfMemory Error)
total_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
total_accuracy=0
# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1]), tf.reshape(W6, [-1]),tf.reshape(W6, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1]), tf.reshape(B6, [-1]),tf.reshape(B6, [-1])], 0)


init_op = tf.global_variables_initializer()

#zapisywanie parametrow (domyslnie wszystko sie zapisuje, jak sie chce konkretne parametry to okresla sie to wyzej - np. tylko b: saver = tf.train.Saver([b])
#https://www.tensorflow.org/api_docs/python/tf/train/import_meta_graph
#Later we can continue training from this saved meta_graph without building the model from scratch.
# Create a saver.
saver = tf.train.Saver()
# Remember the training_op we want to run by adding it to a collection.
tf.add_to_collection('train_step', train_step) #train_step - aby przywrocic trenowanie od jakiegos punktu (sess.run(train_step,....)
tf.add_to_collection('all_weights',allweights)
tf.add_to_collection('all_biasses',allbiases)


#Do wizualizacji w tensorboard// Merge all the summaries and write them out to /tmp/mnist_logs (by default)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

model_version=tf.as_string([tf.cast(W1.shape,tf.float32),tf.cast(W2.shape,tf.float32),tf.cast(W3.shape,tf.float32),tf.cast(W4.shape,tf.float32),tf.cast(W5.shape,tf.float32),tf.concat([tf.cast(W6.shape,tf.float32),[0,0]],0),tf.concat([tf.cast(W7.shape,tf.float32),[0,0]],0)])
model_version = tf.summary.text("Model version", model_version)


train_writer = tf.summary.FileWriter('tensorboard_train',sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_test')

#dodanie na poczatku informacji o strukturze modelu
test_writer.add_summary(model_version.eval())
train_writer.add_summary(model_version.eval())

sess.run(init_op)  #tu inicjalizuje wszystkie wielkosci


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    #batch_X, pre_height, pre_width, pre_filename, batch_Y = sess.run(next_element)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)



    # compute test values for visualisation # liczymy tutaj dokladnosc wykorzystujac funkcje tf.reduce_sum (total_correct), dzieki czemu mozemy zbior testowy podzielic na batche mieszczace sie w pamiecie (brak OOM error)
    if update_test_data:
        sess.run(test_init_op)
        correct_sum = 0
        while True:
            try:
                batch_correct_count = sess.run(total_correct,feed_dict={pkeep: 1.0})
                correct_sum += batch_correct_count
            except tf.errors.OutOfRangeError: #jak przejdziesz przez caly zbior tesotwy to:
                total_accuracy = correct_sum / size_test
                break
        total_accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag="Test accuracy", simple_value=total_accuracy),])
        test_writer.add_summary(total_accuracy_summary,i)
        #summary, a, c = sess.run([merged, accuracy, cross_entropy], feed_dict={pkeep: 1.0})
        #test_writer.add_summary(summary,i)  # do zapisywania parametrow okreslonych wmerged w celu wizualizacji w tensorboard
        print(str(i) + ": ********* epoch " + str(i * batch_size_train // size + 1) + " ********* test accuracy:" + str(total_accuracy))# + " test loss: " + str(c))
        sess.run(training_init_op)  #UWAGA tf.data.Iterator.from_structure razem z make_initilizer dzialaja tak, ze za kazdym razem gdy to jes uruchomione na nowo wczytywana jest
                                    #kolejna epoka. Dlatego, dane testowe z wykorzystaniem tego pipelinu mozna aktualizowac tylko co epoke (bo po inicjalizacji datasetu (next_element)
                                    #danymi testowymi ponownie trzeba inicjalizowac danymi treningowymi, by dalej uczyc siec. Ale to oznacza, ze batche z danych treningowych sa pobierane
                                    #znowu od poczatku. Czyli jezeli zainicjalizuje dane treningowe (sess.run(training_init_op)) wczesniej niz skonczy sie epoka, to nie wszystkie dane treningowe
                                    #zostana uzyte (wykorzystam tylko tyle batchy ile bylo do momentu ponownej inicjalizacji). Z danymi testowymi nie ma porblemu, bo wczytuje wszystkie w jednym batchu

    # compute training values for visualisation
    if update_train_data:
        #sess.run(training_init_op)
        summary, a, c, w, b = sess.run([merged, accuracy, cross_entropy,allweights, allbiases], feed_dict={pkeep: 1.0})
        train_writer.add_summary(summary,i)  # do zapisywania parametrow okreslonych wmerged w celu wizualizacji w tensorboard
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    # the backpropagation training step
    sess.run(train_step, feed_dict={lr: learning_rate, pkeep: 0.75}) #0.75


epochs=15
rounded_size=size - size % -batch_size_train #czyli rounded_size to jest zokraglony rozmiar zbioru trenujacego do nastepnej mozliwej do osiagniecia w petli wartosci, (np. size=96347, batch= 100, rounded_size=96400) ->po to aby spelnic warunek w parametrze funkcji training step -> (i*batch_size_train) % rounded_size == 0)
print("rounded size= "+str(rounded_size))
#sess.run(training_init_op) # to ininjalizuje tylko dataset, czyli pobiera jakby kolejne zdjecia, nie zeruje natomiast innych wielkosci np. train step
for i in range(9641): training_step(i, (i*batch_size_train) % rounded_size == 0, i % 20 == 0) #for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

#do zapisyania parametro (tez calego grafu)
save_path=saver.save(sess, './zapisane/my-model')
saver.export_meta_graph('./zapisane/my-model.meta')
print("Model saved in file: %s" % save_path)


sess.close()
