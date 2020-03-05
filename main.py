import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle



import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


#Defining labels 

def get_label(argument):
	labels = {0:'Happy', 1:'Sad'}
	return(labels.get(argument, "Invalid emotion"))


def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				training_data.append([img_array, class_num])
			except Exception as e:
				pass
				

file_list = []
class_list = []

DATADIR = "BDD"

# All the categories you want your neural network to detect
CATEGORIES = ["Happy", "Sad"]


# The size of the images that your neural network will use
IMG_SIZE = 100



training_data = []


create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


##############################################################################

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))


X_train=X[:15]
y_train=y[:15]

X_test=X[15:]
y_test=y[15:]


model = Sequential()


model.add(Conv2D(32, (3, 3), padding='same' , activation='relu', input_shape=(100, 100, 1)))
#Adding more layers
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same' , activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Flattening
model.add(Flatten())
# Adding fully connected layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile( loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



batch_size=256
epochs =25

history = model.fit( X_train,y_train, epochs=epochs, batch_size = batch_size , verbose = 1 )

model.evaluate(X_test, y_test)

#resize plots
plt.figure(figsize=[12,8])
# Display random image from training data
plt.subplot(3,3,1)
plt.imshow(np.squeeze(X_train[2]),cmap='gray')
plt.title("entrainement : {}".format(get_label(int(y_train[2])) ) )


# Display random image from predected data
prediction = model.predict(X_test)

plt.subplot(3,3,3)

plt.imshow(np.squeeze(X_test[0]),cmap='gray')
if(prediction[0][0]>0.5 ) :
	plt.title("test :Happy")
else:
	plt.title("test :Sad")
print(prediction[0])
plt.subplot(3,3,4)
plt.imshow(np.squeeze(X_test[1]),cmap='gray')
if(prediction[1][0]>0.5 ) :
	plt.title("test :Happy")
else:
	plt.title("test :Sad")

print(prediction[1])
plt.subplot(3,3,6)
plt.imshow(np.squeeze(X_test[2]),cmap='gray')
if(prediction[2][0]>0.5 ) :
	plt.title("test :Happy")
else:
	plt.title("test :Sad")
print(prediction[2])
plt.subplot(3,3,7)
plt.imshow(np.squeeze(X_test[3]),cmap='gray')
if(prediction[3][0]>0.5 ) :
	plt.title("test :Happy")
else:
	plt.title("test :Sad")
print(prediction[3])
plt.subplot(3,3,9)
plt.imshow(np.squeeze(X_test[4]),cmap='gray')
if(prediction[4][0]>0.5 ) :
	plt.title("test :Happy")
else:
	plt.title("test :Sad")
print(prediction[4])




plt.show()



