from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn
import tenforflow as tf
#I know that the image processor and adapter to the cnn is with KerasTF, im not blind, but i didn't thinked in any better option for passing the image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#-------------------------------------------------------------
#This is a image generator that adapt the images and make them readable for the generator
datagen = ImageDataGenerator(rescale=1./255)
#This is where he will get the images, with the directory
img_dir = "/ruta/a/directorio/de/imagenes"
img_width, img_height = 150, 150
#We create the image "transformer"
train_generator = datagen.flow_from_directory(
        img_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
#-------------------------------------------------------------
#This is the CNN, we pass it from relu to softmax, then we declare it a regression using ADAM optimizer, ignore de learning rate LOL
convnet = input_data(shape=[None, 28, 28, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet)
#-------------------------------------------------------------
#We save all on our Coockies jar
history = model.fit(train_generator, n_epoch=1000, show_metric=True)
model.save("CoockiesJar.tflearn")
