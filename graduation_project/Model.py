import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras import models,layers,datasets
import h5py
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
print("img classification")
print("img preprocssing")
train_img_generator=ImageDataGenerator(rescale=1.0/255)#rescale all img
test_img_generator=ImageDataGenerator(rescale=1.0/255)#rescale all img

training_images=train_img_generator.flow_from_directory('covid19dataset/train',
                                                        target_size=(64,64),batch_size=8,class_mode='binary')
testing_images=train_img_generator.flow_from_directory('covid19dataset/test',
                                                        target_size=(64,64),batch_size=8,class_mode='binary')
def plotimg(images):
    fig,axes=plt.subplots(1,5,figsize=(20,20))
    axes=axes.flatten()
    for img, ax in zip(images,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
sample_training_images, _ =next(training_images)
plotimg(sample_training_images[:5])
print("create cnn model")
model=Sequential()
model=models.Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D((2,2)))


model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

#FURTHER ADDING THE NEURALNET TO CONVNET MODEL
model.add(Flatten())
model.add(Dense(128,activation='relu'))#128 neruons

model.add(Dense(128,activation='relu'))#add onr more dense 128 neruons
model.add(Dense(1,activation='sigmoid'))#final output result with 1 neruons 10 result
print("step 3 train the model")
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fit_generator insteat of fit if we use image data generator for our image data set Creation
history=model.fit_generator(training_images,epochs=5,validation_data=testing_images)
print("step 4 Visualizing accuracy and loss of the model")
acc=history.history["accuracy"]
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs_range=range(5)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='Training Accuracy')
plt.plot(epochs_range,val_acc,label='test Accuracy')
plt.legend(loc='lower right')
plt.title('Acurracy')
plt.show()
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss,label='test Loss')
plt.legend(loc='upper left')
plt.title('LOSS')
plt.show()

model.save("model.h5")
print("model saved")