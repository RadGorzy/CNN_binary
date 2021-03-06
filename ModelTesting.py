import tensorflow as tf
import imageio
import numpy
#tested with tensorflow-gpu==1.9
print("IM IN PYTHON ................")
#solves problem: https://github.com/google/oauth2client/issues/642
#import sys
#if not hasattr(sys, 'argv'):
    #sys.argv  = ['']



tf.app.flags.DEFINE_integer('number_of_classes', 5,
                            'Number of classes (among which we will classify images) in taining and test dataset')
tf.app.flags.DEFINE_string('model_directory', './saved/CNN_range_3D_BIWI',    #'./data/train',                  #CNN_binary_3D_Map_person_1 #CNN_range_3D_BIWI #CNN_mapillary_SV
                           'Training data directory')

FLAGS = tf.app.flags.FLAGS

def classify(test):

    with tf.Session() as sess:

        test = tf.multiply(tf.cast(test, tf.float32), 1.0 / 255.0)
        test = tf.expand_dims(tf.expand_dims(test, axis=-1), axis=0)
        test = test.eval()
        print("test.shape: " + str(test.shape))

        # import previously exported graph
        saver = tf.train.import_meta_graph(FLAGS.model_directory+'/my-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_directory))

        # Getting operations, tensors (elements of the graph, so also of the nn model)
        # Its best to look for required operations on graphical representation of imported graph in Tensoroboard(for ex. in terminal: tensorboard --logdir=/home/radek/DeepLearning/CNN_binaryensorboard_train/CNN_5warstwKonw   ,and then in web browser: http://localhost:6006  )
        next_element_image = tf.get_default_graph().get_operation_by_name('IteratorGetNext').outputs[0]  # outputs[4] - gives next_element[4] - tensor to which we load labels - data saved in next_element is described in CreateData.py
        pkeep = tf.get_default_graph().get_operation_by_name('Placeholder_1').outputs[0]  # for testing pkeep=1 (dropout parameter used only in training)
        Y = tf.get_default_graph().get_operation_by_name('Softmax').outputs[0]  # Y output layer (after softmax - vector of values [0-1] defining a belief, that data belong to class corresponding to index of this vector
        print(next_element_image)

        Y_ = sess.run(Y, feed_dict={next_element_image: test,pkeep: 1})
        print(Y_)
        print("Classification result: " + str(numpy.argmax(Y_)))
        return numpy.argmax(Y_)

def classify_multiple_projections(test,modelDirectory,number_of_classes): #clasify single object based on its multiple projections
    print("USING {} MODEL ".format(modelDirectory))

    with tf.Session() as sess:
        # import previously exported graph
        saver = tf.train.import_meta_graph(modelDirectory + '/my-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(modelDirectory))

        # Getting operations, tensors (elements of the graph, so also of the nn model)
        # Its best to look for required operations on graphical representation of imported graph in Tensoroboard(for ex. in terminal: tensorboard --logdir=/home/radek/DeepLearning/CNN_binaryensorboard_train/CNN_5warstwKonw   ,and then in web browser: http://localhost:6006  )
        next_element_image = tf.get_default_graph().get_operation_by_name('IteratorGetNext').outputs[0]  # outputs[4] - gives next_element[4] - tensor to which we load labels - data saved in next_element is described in CreateData.py
        pkeep = tf.get_default_graph().get_operation_by_name('Placeholder_1').outputs[0]  # for testing pkeep=1 (dropout parameter used only in training)
        Y = tf.get_default_graph().get_operation_by_name('Softmax').outputs[0]  # Y output layer (after softmax - vector of values [0-1] defining a belief, that data belong to class corresponding to index of this vector
        # print(next_element_image)

        max_sum = numpy.zeros((1,number_of_classes))
        k=0
        #print("test_shape "+str(numpy.shape(test)))
        for image in test:

            image = tf.multiply(tf.cast(image, tf.float32), 1.0 / 255.0)
            image = tf.expand_dims(tf.expand_dims(image, axis=-1), axis=0)
            image = image.eval()
            #print("image.shape: " + str(image.shape))

            Y_ = sess.run(Y, feed_dict={next_element_image: image,pkeep: 1})
            """
            if(Y_[0,numpy.argmax(Y_)]<0.9):  #minimal neural net response for single projection, if less then threshold than class is unknown
                print("CNN response for projection number "+str(k)+ " of current object: " + str(Y_))
                print("No class: "+str(Y_[0,numpy.argmax(Y_)]))
                return -1
            """
            max_sum=max_sum+Y_
            print("CNN response for projection number " + str(k) + " of current object: " + str(Y_))
            k=k+1

        print("max_sum = "+str(max_sum))
        print("Classification result: " + str(numpy.argmax(max_sum)))

    tf.reset_default_graph()
    return numpy.argmax(max_sum)
def classify_multiple_projections_and_get_response_vector(test,modelDirectory,number_of_classes): #clasify single object based on its multiple projections
    print("USING {} MODEL ".format(modelDirectory))

    with tf.Session() as sess:
        # import previously exported graph
        saver = tf.train.import_meta_graph(modelDirectory + '/my-model.meta')
        """
        counter = 0
        for op in tf.get_default_graph().get_operations():
            print("{} {}".format(counter, str(op.name)))
            counter = counter + 1
        """
        saver.restore(sess, tf.train.latest_checkpoint(modelDirectory))

        # Getting operations, tensors (elements of the graph, so also of the nn model)
        # Its best to look for required operations on graphical representation of imported graph in Tensoroboard(for ex. in terminal: tensorboard --logdir=/home/radek/DeepLearning/CNN_binaryensorboard_train/CNN_5warstwKonw   ,and then in web browser: http://localhost:6006  )
        next_element_image = tf.get_default_graph().get_operation_by_name('IteratorGetNext').outputs[0]  # outputs[4] - gives next_element[4] - tensor to which we load labels - data saved in next_element is described in CreateData.py
        pkeep = tf.get_default_graph().get_operation_by_name('Placeholder_1').outputs[0]  # for testing pkeep=1 (dropout parameter used only in training)
        Y = tf.get_default_graph().get_operation_by_name('Softmax').outputs[0]  # Y output layer (after softmax - vector of values [0-1] defining a belief, that data belong to class corresponding to index of this vector
        # print(next_element_image)

        max_sum = numpy.zeros((1,number_of_classes))
        k=0
        #print("test_shape "+str(numpy.shape(test)))
        for image in test:

            image = tf.multiply(tf.cast(image, tf.float32), 1.0 / 255.0)
            image = tf.expand_dims(tf.expand_dims(image, axis=-1), axis=0)
            image = image.eval()
            #print("image.shape: " + str(image.shape))

            Y_ = sess.run(Y, feed_dict={next_element_image: image,pkeep: 1})

            max_sum=max_sum+Y_
            print("CNN response for projection number " + str(k) + " of current object: " + str(Y_))
            k=k+1

        print("max_sum = "+str(max_sum))
        print("Classification result: " + str(numpy.argmax(max_sum)))

    tf.reset_default_graph()

    result=max_sum[0].tolist()
    result.insert(0,numpy.argmax(max_sum))

    return tuple(result)
def testing(test):
    print("image.shape=")
    print(test)

def main(unused_argv):
    # Wczytanie zdjecia, ktore chcemy sklasyfikowac

    vec=[]
    test = numpy.asarray(imageio.imread("./testing/[b'building_CasV_1_h00_p039.jpg']_[[0. 1. 0. 0. 0.]]test.jpg",as_gray=1))
    #test1 = scipy.ndimage.imread("./testing/Building_Finnmarken91_0_h00_p081.jpg", flatten=1)
    #test2 = scipy.ndimage.imread("./testing/person_0042_h00_p012.jpg", flatten=1)
    vec.append(test)
    #vec.append(test1)
    #vec.append(test2)
    #testing(test1)

    #print(classify_multiple_projections(vec))
    print(classify_multiple_projections_and_get_response_vector(vec,"/home/radek/DeepLearning/CNN_binary/zapisane/CNN_range_SemanticKITTI",5)) #/home/radek/DeepLearning/CNN_binary/zapisane/CNN_range_SemanticKITTI #/home/radek/DeepLearning/CNN_binary/zapisane/CNN_range_3D #/home/radek/DeepLearning/CNN_binary/zapisane/Porownanie/CNN_binary_3D

    #for image in vec:
     #   result=classify(image)
      #  print(result)


    """
    test = scipy.ndimage.imread("./testing/0000002_0005817_0000004_0001082_4_bb1_e.jpg", flatten=1)
    classify(test)
    """
if __name__ == '__main__':
    tf.app.run()
