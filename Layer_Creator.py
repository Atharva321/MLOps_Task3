import random,os
acc = 0.00
n = 4

filt,filt_cnt = 64,1
f,k = 0,4096
def Dense_relu_Dropout(k):
    arch = open("NN.py", "a+")
    arch.write("""\nmodel.add(Dense(%d, input_shape=(224*224*3,)))\nmodel.add(Activation(‘relu’))\n# Add Dropout
\nmodel.add(Dropout(0.4))""" % k)
    arch.close()
    
def Dense_softmax():
    arch = open("NN.py", "a+")
    arch.write("""\nmodel.add(Dense(17)) \nmodel.add(Activation(‘softmax’))""")
    arch.close()
def MaxPool():
    arch = open("NN.py", "a+")
    arch.write("\nmodel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))")
    arch.close()
def Conv2D(s):
    arch = open("NN.py", "a+")
    g = random.randint(2,12)
    arch.write("""\nmodel.add(Conv2D(filters= {0}, kernel_size=({1},{2}), strides=(1,1), padding=’valid’))""".format(s,g,g))
    arch.close()
#while(acc < 95):
arch = open("NN.py","w+")
    #for n in range(0,4):
arch.write("""\nimport tensorflow as tf \nx_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()\n
\nx_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
\nx_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
\ninput_shape = (28, 28, 1)
\n# Making sure that the values are float so that we can get decimal points after division
\nx_train = x_train.astype('float32')
\nx_test = x_test.astype('float32')
\n# Normalizing the RGB codes by dividing it to the max RGB value.
\nx_train /= 255
\nx_test /= 255
\nimport keras \nfrom keras.models import Sequential \nfrom keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D\n
from keras.layers.normalization import BatchNormalization\nimport numpy as np\nmodel = Sequential()""")
arch.close()
Conv2D(filt)
MaxPool()
for m in range(1,6):
    Conv2D(filt)
    if(m%2==0):
        MaxPool()
    filt_cnt += 1
    filt += filt
    
for i in range(0,3):
    Dense_relu_Dropout(k)
    k -= int(k/2)
Dense_softmax()
abc = open("NN.py", "a+")
abc.write("""\nmodel.compile(loss=keras.losses.categorical_crossentropy, optimizer=’adam’, metrics=[“accuracy”]) \nmodel.evaluate(x_test, y_test)\n""")
abc.close()
#os.system("python NN.py")
