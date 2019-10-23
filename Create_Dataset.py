#contains functions used for loading and editing data in TFRecords files
#uses tf.data structure instead of deprecated QueueRunner (np. tf.train.batch())
import tensorflow as tf
import numpy as np
import list_TFRecordFiles as TFRfile

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
               "image/class/label"]  #parsed_features will return this parameters in random order


# """
def edit(image, height, width, filename, label,resize=0):
    image = tf.image.decode_jpeg(image, channels=1)  # image_raw
    if resize == 1:
        image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.image_height,
                                                       FLAGS.image_width)  # cropped image with size 299x299
    #    current_image_object.image = tf.cast(image_crop, tf.float32) * (1./255) - 0.5
    # z uint8 do float32 i jednoczesnie z 0:255 do 0:1:
    image = tf.multiply(tf.cast(image, tf.float32), 1.0 / 255.0)
    # print("image:  "+str(image))
    # transform labels to vector form:for ex. class 0 = [1 0 0 0 0 0 0 0 0 0]
    # WARNING, label vector length depends here on  number of neurons in output layer of neural net - which  is equivalent to number of classes we classify objects on
    label = tf.cast(label, tf.uint8)  # label of the raw image
    label_wec = np.zeros((FLAGS.class_number,), dtype=np.uint8)
    if (label >= 1) is not None:  # TypeError: Using a tf.Tensor as a Python bool is not allowed. Use if t is not None: instead of if t: to test if a tensor is defined, and use the logical TensorFlow ops to test the value of a tensor.
        indices = label - 1
        depth = FLAGS.class_number
        label_wec = tf.one_hot(indices, depth, on_value=1.0, off_value=0.0, dtype=tf.float32)
        #label_wec = tf.expand_dims(label_wec,0)
    else:
        print("Label value must be >=1")
    return image, height, width, filename, label_wec


# """
def read_data_set(path):
    tfrecord_list = TFRfile.tfrecord_auto_traversal(
        path)  # tfrecord_list=TFRfile.tfrecord_auto_traversal("./data/TFRecords")
    size = 0  # number of objects in TFRecord files:
    for fn in tfrecord_list:
        for record in tf.python_io.tf_record_iterator(fn):
            size += 1
    dataset = tf.data.TFRecordDataset(tfrecord_list)
    dataset = dataset.map(get_features, num_parallel_calls=8)
    return dataset, size  # returns dataset and its size (number of objects)

def next_batch(dataset, batch_size):
    dataset = dataset.shuffle(buffer_size=1000)
    batched_dataset = dataset.batch(
        batch_size)  # batchSize -number of images multiplied at once
    batched_dataset = batched_dataset.repeat()  # so it will be repeating indefinitely (OutOfRangeError will never be risen)

    iterator = batched_dataset.make_one_shot_iterator()
    #iterator=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
    next_element = iterator.get_next()
    #init_op = iterator.make_initializer(dataset)

    return next_element   #, init_op # optionaly return only images and label (for better performance): return next_element[0], next_element[4]


# def read_to_array(path)
#main function just for testing :
def main(unused_argv):
    print("TEST CASE")


if __name__ == '__main__':
    tf.app.run()

