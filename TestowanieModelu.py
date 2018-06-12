import tensorflow as tf
import scipy
import numpy

#Wczytanie zdjecia, ktore chcemy sklasyfikowac
test = scipy.ndimage.imread("testing/0000000003.bin.pcd_d_c_4.pcd_p4.jpg",flatten=1)  # z tych przykladowych to zle klasyfikuje 92.png, pozostale ok - wyglada na to, model jest dobrze wytrenowany, ale nie wiem nadal dlaczego tak
# print("zdjecie"+str(test[6][15]))
test = tf.multiply(tf.cast(test, tf.float32), 1.0 / 255.0)
test=tf.expand_dims(tf.expand_dims(test,axis=-1),axis=0)


with tf.Session() as sess:
    test = test.eval()
    print("test.shape: " + str(test.shape))

    #zaimportowanie wczesniej wyeksportowanego grafu
    saver = tf.train.import_meta_graph('./zapisane/CNN_5warstwKonw/my-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./zapisane/CNN_5warstwKonw'))

    #Pobranie operacji, tenosrow (elementy grafu, czyli tez modelu sieci) do ktorego ladujemy
    #interesujace nas operacje najepiej szukac na reprezetnacji graficznej zaimportowanego grafu w Tensorboard (np. w terminalu: tensorboard --logdir=/home/radek/DeepLearning/CNN_binaryensorboard_train/CNN_5warstwKonw   ,a nastepnie w przegladarce: http://localhost:6006  )
    next_element_image = tf.get_default_graph().get_operation_by_name('IteratorGetNext').outputs[0] # outputs[4] - daje next_element[4] - tensor do ktorego laduje sie labele - dane zapisane w next_element okreslone sa w CreateData.py
    pkeep = tf.get_default_graph().get_operation_by_name('Placeholder_1').outputs[0] #dla testowanie pkeep=1 (parametr do dropoutu)
    Y = tf.get_default_graph().get_operation_by_name('Softmax').outputs[0]  #Y warstwa wyjsciowa (czyli po softmax - wektor wartosci [0-1] okreslajacych przekonanie, ze dane naleza do klasy odpowiadajacej indexowi tego wektora
    print(next_element_image)

    Y = sess.run(Y, feed_dict={next_element_image: test,pkeep: 1})
    print(Y)
    print("Wynik klasyfikacji: " + str(numpy.argmax(Y)))

