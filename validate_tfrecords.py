import tensorflow as tf
import list_TFRecordFiles as TFRfile
import os
import datetime
import argparse


def validate_dataset(filenames, reader_opts=None):
    """
    Attempt to iterate over every record in the supplied iterable of TFRecord filenames
    :param filenames: iterable of filenames to read
    :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
    """
    i = 0
    toBeRepaired=[]
    for fname in filenames:
        print('validating initially', fname)

        record_iterator = tf.python_io.tf_record_iterator(path=fname, options=reader_opts)
        try:
            for record in record_iterator:
                i += 1
        except Exception as e:
            print('error in {} at record {}'.format(fname, i))
            print(e)
            #print(str(record))
            print("FIle {} needs repairing".format(fname))
            toBeRepaired.append(fname)
    return toBeRepaired
#############################
# The code below here uses the crcmod package to implement an alternative method which is able to print out
# if it finds a bad record and attempt to keep going. If the corruption in your file is just flipped bits this may be helpful.
# If the corruption is added or deleted bytes this will probably crash and burn.
import struct
from crcmod.predefined import mkPredefinedCrcFun

_crc_fn = mkPredefinedCrcFun('crc-32c')


def calc_masked_crc(data):
    crc = _crc_fn(data)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xFFFFFFFF

def writeRecord(dstTFRecordPath,length,len_crc,data,data_crc):
    with open(dstTFRecordPath, 'ab') as f:
        f.write(length)
        f.write(len_crc)
        f.write(data)
        f.write(data_crc)

def validate_dataset_slower(filenames,repair=False):
    print("\nINITIAL VALIDATION\n")
    toBeRepaired=validate_dataset(filenames)
    print("\nFOLLOWING FILES NEED REPAIRING: {}\n".format(toBeRepaired))
    total_records = 0
    total_bad_len_crc = 0
    total_bad_data_crc = 0
    for f_name in toBeRepaired:
        i = 0
        print('validating ', f_name)
        name = os.path.basename(f_name)
        dirname = os.path.dirname(f_name)
        repairedName = dirname + "/" + os.path.splitext(name)[0] + "_repaired.tfrecord"  # .../xxxx_repaired.tfrecord
        if repair:
            with open(dirname + "/corruptedData.txt", 'a') as f:
                f.write("{}\n".format(datetime.datetime.now()))

        with open(f_name, 'rb') as f:
            len_bytes = f.read(8)
            while len(len_bytes) > 0:
                # tfrecord format is a wrapper around protobuf data
                length, = struct.unpack('<Q', len_bytes) # u64: length of the protobuf data (excluding the header)
                len_crc, = struct.unpack('<I', f.read(4)) # u32: masked crc32c of the length bytes
                data = f.read(length) # protobuf data
                data_crc, = struct.unpack('<I', f.read(4)) # u32: masked crc32c of the protobuf data

                #if everything is OK, than write it to new TFRecord
                if repair and (len_crc == calc_masked_crc(len_bytes)) and (data_crc == calc_masked_crc(data)):
                    writeRecord(repairedName,
                            struct.pack('<Q',length),
                            struct.pack('<I', len_crc),
                            data,
                            struct.pack('<I',data_crc))

                if len_crc != calc_masked_crc(len_bytes):
                    print('bad crc on len at record', i)
                    total_bad_len_crc += 1
                    print(data)
                    print(tf.train.Example.FromString(data))
                    with open(dirname+"/corruptedData.txt", 'a') as infoFile:
                        infoFile.write("bad crc on len in {} at record {}\n".format(name+".tfrecord",i))
                        infoFile.write("{}\n".format(tf.train.Example.FromString(data)))
                        infoFile.write("***************\n")

                if data_crc != calc_masked_crc(data):
                    print('bad crc on data at record', i)
                    total_bad_data_crc += 1
                    print(data)
                    print(tf.train.Example.FromString(data))
                    with open(dirname + "/corruptedData.txt", 'a') as infoFile:
                        infoFile.write("bad crc on data in {} at record {}\n".format(name+".tfrecord",i))
                        infoFile.write("{}\n".format(tf.train.Example.FromString(data)))
                        infoFile.write("***************\n")
                i += 1
                len_bytes = f.read(8)

        print('checked', i, 'records')
        total_records += i
    print('checked', total_records, 'total records')
    print('total with bad length crc: ', total_bad_len_crc)
    print('total with bad data crc: ', total_bad_data_crc)
import Create_Dataset as Data

def dryRun():
    tf.set_random_seed(0)
    batch_size_train = 100
    train_dataset, size = Data.read_data_set(
        "/media/radek/SanDiskSSD/SemanticKITTI/TFRecords/testing/train")  # padded_batch(100,padded_shapes=([None],[None],[None],[None],[None])).
    train_dataset = train_dataset.map(
        Data.edit)  # tu jeszcze jest argument num_parallel_calls=None, ktory oznacza iosc elementow przetwarzanych na raz
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(
        batch_size_train).repeat()  # padded_batch -musze zastosowac taki rodzaj (a nie samo batch),bo tensory w dataset nie maja takich samych wymiarow - ilosc obrazow w jednej partii (wymnazancyh za jednym razem), repeat() - czyli bedzie powtarzac w nieskonsczonosc (nigdy nie bedzie OutOfRangeError)

    test_dataset, size_test = Data.read_data_set("/media/radek/SanDiskSSD/SemanticKITTI/TFRecords/testing/validation")
    test_dataset = test_dataset.map(Data.edit)
    test_dataset = test_dataset.shuffle(buffer_size=1000).batch(
        100)  # tutaj batch powinno byc rowne ilosci wszystkich elementow w zbiorze testowym: 40142 #nie ma na koncu repeat(), zeby po przejsciu
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

    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    epochs = 5  # dla CNN_mapillary_streetview
    rounded_size = size - size % -batch_size_train  # czyli rounded_size to jest zokraglony rozmiar zbioru trenujacego do nastepnej mozliwej do osiagniecia w petli wartosci, (np. size=96347, batch= 100, rounded_size=96400) ->po to aby spelnic warunek w parametrze funkcji training step -> (i*batch_size_train) % rounded_size == 0)
    print("rounded size= " + str(rounded_size))
    iterations = int((rounded_size / batch_size_train) * epochs)
    print("Doing " + str(iterations) + " iterations, with batches of " + str(
        batch_size_train) + " which results in " + str(epochs) + " epochs.")
    sess.run(training_init_op) # to ininjalizuje tylko dataset, czyli pobiera jakby kolejne zdjecia, nie zeruje natomiast innych wielkosci np. train step
    for i in range(iterations + 1):
        try:
            #print("EVALUATING ...")
            #print("{} {}".format(i,sess.run(next_element)))
            image, height, width, filename, label_wec=sess.run(next_element) #image, height, width, filename, label_wec
            print("{} {};{};{};{}={};{}".format(i,image.shape,height.shape,width.shape,filename.shape,filename,label_wec.shape))
        except Exception as e:
            print("Error while: {} {};{};{};{}={};{}".format(i, image.shape, height.shape, width.shape, filename.shape, filename,label_wec.shape))

def main(unused_argv):
    """
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dir", "--directory_path", required=True,
                    help="path to directory with TFRecords files we want to check")
    args = vars(ap.parse_args())

    tfrList=TFRfile.tfrecord_auto_traversal(args["directory_path"])
    validate_dataset_slower(tfrList,True)
    """
    dryRun()
if __name__ == '__main__':
    tf.app.run()