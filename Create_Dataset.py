#zawiera funkcje do wczytania i edycji danych z plikow TFRecords
#wersja wykorzystujaca strukture tf.data , a nie przestarzaly sposob z wykorzystaniem QueueRunner (np. tf.train.batch())
import tensorflow as tf
import numpy as np
import list_TFRecordFiles as TFRfile
import cv2 as cv #do celow testowych

flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_integer("image_number_train", 9, "Number of images in your tfrecord, default is 300.")
#flags.DEFINE_integer("image_number_test", 1, "Number of images in your tfrecord, default is 300.")
flags.DEFINE_integer("class_number", 5, "Number of class in your dataset/label.txt, default is 3.") #to lepiej przekazywac jako parametr (po dodaniu klas mozna zapomniec tu zmienic)
flags.DEFINE_integer("image_height", 299, "Height of the output image after crop and resize. Default is 299.")
flags.DEFINE_integer("image_width", 299, "Width of the output image after crop and resize. Default is 299.")


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_features(example_proto):
    features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64), }
    parsed_features = tf.parse_single_example(example_proto, features)
    # print("Features" + str(parsed_features))
    return parsed_features["image/encoded"], parsed_features["image/height"], parsed_features["image/width"], \
           parsed_features["image/filename"], parsed_features[
               "image/class/label"]  # samo parsed_features bedzie zwracalo te parametry w losowej kolejnosci


# """
def edit(image, height, width, filename, label,resize=0):
    image = tf.image.decode_jpeg(image, channels=1)  # image_raw
    if resize == 1:
        image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.image_height,
                                                       FLAGS.image_width)  # cropped image with size 299x299
    #    current_image_object.image = tf.cast(image_crop, tf.float32) * (1./255) - 0.5
    # z uint8 do float32 i jednoczesnie z 0:255 do 0:1:
    image = tf.multiply(tf.cast(image, tf.float32), 1.0 / 255.0)
    #image = tf.expand_dims(image,0)  #taki wymiar jest wymagany w warstwie sieci, wstawia dodatkowy wymiar w indexie 0
    # print("image:  "+str(image))
    # przeksztalcenie etykiety z postaci cyfra 0 = 1 (bo na razie 0 oznacza klase niezdefiniowana) do postaci wektora: cyfra 0 = [1 0 0 0 0 0 0 0 0 0]
    # !!!Uwaga, tutaj dlugosc wektora label zalezy od ilosci neuronow w warstwie wysjciowej w sieci - czyli od ilosci klas na ktore klasyfikujemy obiekty
    label = tf.cast(label, tf.uint8)  # label of the raw image
    label_wec = np.zeros((FLAGS.class_number,), dtype=np.uint8)
    if (label >= 1) is not None:  # TypeError: Using a tf.Tensor as a Python bool is not allowed. Use if t is not None: instead of if t: to test if a tensor is defined, and use the logical TensorFlow ops to test the value of a tensor.
        indices = label - 1
        depth = FLAGS.class_number
        label_wec = tf.one_hot(indices, depth, on_value=1.0, off_value=0.0, dtype=tf.float32)
        #label_wec = tf.expand_dims(label_wec,0)
    else:
        print("Wartosc label musi byc >=1")
    return image, height, width, filename, label_wec


# """
def read_data_set(path):
    tfrecord_list = TFRfile.tfrecord_auto_traversal(
        path)  # tfrecord_list=TFRfile.tfrecord_auto_traversal("./data/TFRecords")
    size = 0  # ilosc obiektow w plikach TFRecord:
    for fn in tfrecord_list:
        for record in tf.python_io.tf_record_iterator(fn):
            size += 1
    dataset = tf.data.TFRecordDataset(tfrecord_list)
    dataset = dataset.map(get_features, num_parallel_calls=8)
    return dataset, size  # zwraca dataset i jego rozmiar (ilosc obiektow)

def next_batch(dataset, batch_size):
    dataset = dataset.shuffle(buffer_size=1000)
    batched_dataset = dataset.batch(
        batch_size)  # tutaj definuje batchSize - ilosc obrazow w jednej partii (wymnazancyh za jednym razem)
    batched_dataset = batched_dataset.repeat()  # czyli bedzie powtarzac w nieskonsczonosc (nigdy nie bedzie OutOfRangeError)

    iterator = batched_dataset.make_one_shot_iterator()
    #iterator=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
    next_element = iterator.get_next()
    #init_op = iterator.make_initializer(dataset)

    return next_element   #, init_op # ewentualnie zwracaj tutaj same zdjecia i label czyli return next_element[0], next_element[4]


# def read_to_array(path)
#main function just for testing :
def main(unused_argv):
    batch_size_train = 1
    train_dataset, size = read_data_set("./data/TFRecords/train")  # padded_batch(100,padded_shapes=([None],[None],[None],[None],[None])).
    train_dataset = train_dataset.map(edit,num_parallel_calls=8)
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size_train).repeat()  # padded_batch -musze zastosowac taki rodzaj (a nie samo batch),bo tensory w dataset nie maja takich samych wymiarow - ilosc obrazow w jednej partii (wymnazancyh za jednym razem), repeat() - czyli bedzie powtarzac w nieskonsczonosc (nigdy nie bedzie OutOfRangeError)

    test_dataset, size_test = read_data_set("./data/TFRecords/test")
    test_dataset = test_dataset.map(edit,num_parallel_calls=8)
    test_dataset = test_dataset.shuffle(buffer_size=1000).batch(1)  # tutaj batch powinno byc rowne ilosci wszystkich elementow w zbiorze testowym: 40142 #nie ma na koncu repeat(), zeby po przejsciu
    # calego zbioru wyrzucil tf.errors.OutOfRangeError (co pozniej wykorzystuje)
    print("Train dataset size = " + str(size))
    print("Test dataset size = " + str(size_test))

    # create general iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    next_element = iterator.get_next()

    # make datasets that we can initialize separately, but using the same structure via the common iterator
    training_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Test images:   ")
        for i in range(1):  # for i in range(FLAGS.image_number_test):  # Uwaga, teraz odwiedzam 8 obrazkow, a nie 9. - generalnie ilosc wykonan tej petli zalezy od liczby obrazow w jednej batch.  -> il.wyk=il.obrazow/batchSize Ale tak musze zr
            #zainicjalizuj zbiorem trenujacym
            sess.run(training_init_op)
            pre_image, pre_height, pre_width, pre_filename, pre_label = sess.run(next_element)
            print("Train_Image_shape" + str(pre_image.shape))
            print("Train_Label_shape\n" + str(pre_label.shape))
            print("Filename" + str(pre_filename) + "label: " + str(pre_label) + "----------")
            print(np.unique(pre_image))
            pre_image = np.multiply(pre_image, 255)
            cv.imwrite("./testing/"+str(pre_filename)+"_"+str(pre_label)+"train.jpg",pre_image[0,:,:,0])
            #zainicjalizuj zbiorem testujacym
            sess.run(test_init_op)
            pre_image, pre_height, pre_width, pre_filename, pre_label = sess.run(next_element)
            print("Test_Image_shape" + str(pre_image.shape))
            print("Test_Label_shape\n" + str(pre_label.shape))
            print("Filename" + str(pre_filename) + "label: " + str(pre_label) + "----------")
            pre_image = np.multiply(pre_image, 255)
            cv.imwrite("./testing/" + str(pre_filename) + "_" + str(pre_label) + "test.jpg",pre_image[0,:,:,0])


if __name__ == '__main__':
    tf.app.run()

