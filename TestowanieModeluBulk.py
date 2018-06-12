import tensorflow as tf
import scipy
import scipy.misc
import numpy
import os

def list_classes(path): #path - sciezka do folderu data (zawirajacego podfoldery zawierajace zdjecia kolejnych klas) ->zwraca sciezki do folderow podzednych lub do folderu src, jezeli nie ma folderow podrzednych
    data_folder_classes_list= [path+"/"+d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(data_folder_classes_list)>0:
        print("%s classes (folders) were found under current folder." % len(data_folder_classes_list))
        return data_folder_classes_list
    else:
        print("No, calses was found in src folder. Using %s folder." % path)   #os.path.basename('C:/folder1/folder2') = 'folder2'
        data_folder_classes_list.append(path)
        return data_folder_classes_list


# Traverse current directory
def jpg_auto_traversal(path):
    jpg_list = []
    name_list = []
    current_folder_filename_list = os.listdir(path)  # Change this PATH to traverse other directories if you want.
    #print("currnet folder filename list:" + str(current_folder_filename_list))
    if current_folder_filename_list != None:
        for i in range(len(current_folder_filename_list)):
            # current_file_abs_path = os.path.abspath(file_list[i])
            current_file_abs_path = os.path.abspath(os.path.join(path, current_folder_filename_list[i]))
            # print("path:" + str(current_file_abs_path) + "-----------------")
            if current_file_abs_path.endswith(".jpg"):
                jpg_list.append(current_file_abs_path)
                name_list.append(current_folder_filename_list[i])
                # print("Found %s successfully!" % file_list[i])
            else:
                pass
        if len(jpg_list) == 0:
            print("Cannot find any jpg files, please check the path.")
    return jpg_list, name_list

src="/home/radek/Documents/zrodlaChmur/kitti/JPG/2011_09_26_drive_0048"
dst="/home/radek/Documents/zrodlaChmur/kitti/klasy"
folder_list = list_classes(src)
print("Folder list : "+str(folder_list))
file_list = []


with tf.Session() as sess:
    # zaimportowanie wczesniej wyeksportowanego grafu
    saver = tf.train.import_meta_graph('./zapisane/CNN_5warstwKonw/my-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./zapisane/CNN_5warstwKonw'))

    # Pobranie operacji, tenosrow (elementy grafu, czyli tez modelu sieci) do ktorego ladujemy
    # interesujace nas operacje najepiej szukac na reprezetnacji graficznej zaimportowanego grafu w Tensorboard (np. w terminalu: tensorboard --logdir=/home/radek/DeepLearning/CNN_binaryensorboard_train/CNN_5warstwKonw   ,a nastepnie w przegladarce: http://localhost:6006  )
    next_element_image = tf.get_default_graph().get_operation_by_name('IteratorGetNext').outputs[0]  # outputs[4] - daje next_element[4] - tensor do ktorego laduje sie labele - dane zapisane w next_element okreslone sa w CreateData.py
    pkeep = tf.get_default_graph().get_operation_by_name('Placeholder_1').outputs[0]  # dla testowanie pkeep=1 (parametr do dropoutu)
    Y = tf.get_default_graph().get_operation_by_name('Softmax').outputs[0]  # Y warstwa wyjsciowa (czyli po softmax - wektor wartosci [0-1] okreslajacych przekonanie, ze dane naleza do klasy odpowiadajacej indexowi tego wektora
    print(next_element_image)

    for folder in folder_list:
        print("folder: "+str(folder))
        file_list, name = jpg_auto_traversal(folder)
        for i in range(numpy.size(file_list)):
            image = scipy.ndimage.imread(file_list[i],flatten=1)

            test = tf.multiply(tf.cast(image, tf.float32), 1.0 / 255.0)
            test = tf.expand_dims(tf.expand_dims(test, axis=-1), axis=0)
            test = test.eval() #inaczej: TypeError: The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars, strings, lists, numpy ndarrays, or TensorHandles

            Y_ = sess.run(Y, feed_dict={next_element_image: test, pkeep: 1})
            print(Y_)
            print("Wynik klasyfikacji: " + str(numpy.argmax(Y_)))
            if Y_[0,numpy.argmax(Y_)]>=0.99:
                print(dst+"/"+str(numpy.argmax(Y))+"/"+name[i]+".jpg")
                scipy.misc.imsave(dst+"/"+str(numpy.argmax(Y_))+"/"+name[i],image)
            else:
                #scipy.misc.imsave(dst + "/none/"+name[i], image)
                print("None class:"+name[i])
    print("------------------------------------Finished folder :" + folder)
