# CNN_binary
Clasify binary images with convolutional neural network

First put your train and test images (299x299) in "train" and "test" subfolders of "data" folder.

Next run "build_TFRecordData.py" to create .tfrecord files from your pictures.

Finally run the "CNN_binary_dataset.py" to train your network.

After your network is trained you can check your model for specific image by means of "ModelTesting.py"
