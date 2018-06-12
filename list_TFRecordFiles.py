import os  # handle system path and filenames
import tensorflow as tf  # import tensorflow as usual


# define a function to list tfrecord files.
def list_tfrecord_file(file_list,path):
    tfrecord_list = []
    for i in range(len(file_list)):
        #current_file_abs_path = os.path.abspath(file_list[i])
        current_file_abs_path = os.path.abspath(os.path.join(path, file_list[i]))
        print("path:" + str(current_file_abs_path) + "-----------------")
        if current_file_abs_path.endswith(".tfrecord"):
            tfrecord_list.append(current_file_abs_path)
            print("Found %s successfully!" % file_list[i])
        else:
            pass
    return tfrecord_list


# Traverse current directory
def tfrecord_auto_traversal(path):
    current_folder_filename_list = os.listdir(path)  # Change this PATH to traverse other directories if you want.
    #print("currnet folder filename list:" + str(current_folder_filename_list))
    if current_folder_filename_list != None:
        print("%s files were found under current folder. " % len(current_folder_filename_list))
        print("Please be noted that only files end with '*.tfrecord' will be load!")
        tfrecord_list = list_tfrecord_file(current_folder_filename_list,path)
        if len(tfrecord_list) != 0:
            for list_index in range(len(tfrecord_list)):
                print(tfrecord_list[list_index])
        else:
            print("Cannot find any tfrecord files, please check the path.")
    return tfrecord_list


def main():
    tfrecord_list = tfrecord_auto_traversal("./data/TFRecords/train")


if __name__ == "__main__":
    main()