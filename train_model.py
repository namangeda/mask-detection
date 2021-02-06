#importing all necessary libraries
import tensorflow as tf
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import sklearn.metrics 
import matplotlib.pyplot as plt
import os

#directory where dataset is svaed
dir = "dataset"
LABELS = ["with_mask", "without_mask"]

#creating two empty lists to store all our preprocessed images and labels
images = []
labels = []


#defining a fuction to preprocess every image
def preprocess_image(image_path):
    #loading image in 224*224 because mobileNetv2 will be used 
    #and it was originnaly trained on this size images 
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    
    #converting image into image array
    image = tf.keras.preprocessing.image.img_to_array(image)

    #preprocesing image so that it can be fit to be used in mobilenetV2
    #that is basically normalizing its values between -1 to 1
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image


#preprocessing all the dataset and than saving that preprocessed datset into two lists mentinoed above
for label in LABELS:
    path = os.path.join(dir, label)
    for img in os.listdir(path):
        imagePath = os.path.join(path, img)
        image = preprocess_image(imagePath)
        images.append(image)
        labels.append(label)



#converting list images into numpy array for further calculations
images = np.array(images, dtype="float32")


#binarizing all labels
lb = preprocessing.LabelBinarizer()
labels = lb.fit_transform(labels)

#getting matrix representation of all the labels
labels = tf.keras.utils.to_categorical(labels)

#converting labels into numpy arrays
labels = np.array(labels)


#creating many images of diffrent physicality from one image
virtualImage = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=None,
            shear_range=0.15,
            zoom_range=0.15,
            fill_mode='nearest',
            horizontal_flip=True)


#spliting the entire dataset into train, test split  
print("spliting dataset into (train, test) split ......")
(train_x, test_x, train_y, test_y) = train_test_split(images, labels, 
    test_size=0.25, stratify=labels, random_state=42)                  


#getting mobilenetV2 
#that is loading moblienetV2 network
# final network consist mobilenetV2 as Base and a Fully Connected Layers network as Head on Base
print("loding mobilenetV2......")
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, 
            input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))


#making the Head Model
#also that is adding one pooling layer, one dropout layer, etc as listed below
print("getting head_model done......")
head_model = base_model.output
head_model = tf.keras.layers.AveragePooling2D(pool_size=(7,7))(head_model)
head_model = tf.keras.layers.Flatten(name="flatten")(head_model)
head_model = tf.keras.layers.Dense(128, activation="relu")(head_model)
head_model = tf.keras.layers.Dropout(0.5)(head_model)
head_model = tf.keras.layers.Dense(2, activation="softmax")(head_model)


#getting final network as combination of both the above models
print("final model is ready.....")
model = tf.keras.models.Model(inputs=base_model.input, outputs=head_model)


#because i am using mobilenetV2 with pretrained weights of imagenet
#so layers of mobileneV2 are not supposed to be trained again
#that is why setting these layers as untrainable
print("seting mobileNetV2 layers as untrainable.... ")
for layer in base_model.layers:
    layer.trainable = False



#compiling model
#using adam optimizer and binary_crossenropy loss
#using accuracy metrics to get accuracies on every epoch 
print("compiling model........")
learningRate = 0.0001
Epochs = 20
batchSize = 32
Optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate, 
            decay = learningRate / Epochs)
model.compile(optimizer=Optimizer, loss="binary_crossentropy", 
              metrics=['accuracy'])


#training the model
#flowing the data trough virtualImage generator into the model
print("training the model....")
history = model.fit(
          virtualImage.flow(train_x, train_y, batch_size=batchSize),
          steps_per_epoch=len(train_x)//batchSize,
          validation_data=(test_x, test_y),
          validation_steps=len(test_x)//batchSize,
          epochs=Epochs)


#getting evaluation of model   
print("evaluating the network.........")
probablities = model.predict(test_x, batch_size = batchSize)

#getting the label with highest probablity
probablities = np.argmax(probablities, axis=1)


#saving the model into the directory
model.save("mask_detector.model", save_format="h5")
print("***trained model has been saved into the directory***")


#getting classsification report of the model
print("********* Classification Report **********")
print(sklearn.metrics.classification_report(test_y.argmax(axis=1), 
      probablities, 
      target_names=lb.classes_))


#ploting the whole process of training against time
#that is taking a visual look on how the model performed
#creating a list xs that will be measuring units on x-axis
xs = []
for i in range (0, Epochs):
    xs.append(i)

xs = np.array(xs)  

#plotting graph for training process
plt.figure()
plt.plot(xs, history.history["loss"], label="trainig_loss")
plt.plot(xs, history.history["accuracy"], label="tarining_accuracy")
plt.title("training over epochs")
plt.xlabel("epochs -->")
plt.ylabel("loss/accuracy -->")
plt.legend(loc="upper right")
plt.savefig("training_plot.png")

#plotting graph for validation process
plt.figure()
plt.plot(xs, history.history["val_accuracy"], label="validation_accuracy")
plt.plot(xs, history.history["val_loss"], label="validation_loss")
plt.title("validtion over epochs")
plt.xlabel("epochs -->")
plt.ylabel("loss/accuracy -->")
plt.legend(loc="upper right")
plt.savefig("validation_plot.png")