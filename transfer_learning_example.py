import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.models import Model
#%matplotlib inline

train_path = 'cats-and-dogs/train'
valid_path = 'cats-and-dogs/valid'
test_path = 'cats-and-dogs/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10) #cat a dog jsou tady názvy složek v train directory
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)

# just technical thing
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

imgs, labels = next(train_batches)
# print(len(imgs))
plots(imgs, titles=labels)
plt.show()
# model = Sequential([
        # Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)),
        # Flatten(),
        # Dense(2, activation='softmax'),
    # ])
    
# model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit_generator(train_batches, steps_per_epoch=4, 
                    # validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)
                    

# test_imgs, test_labels = next(test_batches)
# plots(test_imgs, titles=test_labels)

# test_labels = test_labels[:,0] #převod z labelů 0,1 na nul 1,0 na jedna
# print(test_labels)
# predictions = model.predict_generator(test_batches, steps=1, verbose=0)
# print(predictions)


####IMPORT UŽ NATRÉNOVANÉHO MODELU - VGG16
#tento model vyhrál soutěž na imagenet
vgg16_model = keras.applications.vgg16.VGG16()
#vgg16_model.summary()

#print(type(vgg16_model)) #tento model je type Model nikoliv Sequential ->musíme převést
#tvoříme si model,  který je sequential a vkládáme do něho layers z VGG16

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
#model.summary()

#odstranění poslední layer - bohužel už to nějak nefunguje, řeším v loopu nad tímto řádkem, prostě nepřidám poslední layer
# model.layers.pop()
# model.summary()

#zmražení všech vrstev modelu pro učení, weights zůstanou jak jsou
for layer in model.layers:
    layer.trainable = False

#přidání poslední a jediné trénovací vrstvy
model.add(Dense(2, activation='softmax')) #2 pro cat dog
#model.summary()   

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit_generator(train_batches, steps_per_epoch=4, 
                    validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

#konec trénovaání VGG16 modelu
#################                    
                    

test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)


test_labels = test_labels[:,0]
predictions = model.predict_generator(test_batches, steps=1, verbose=0)


#mimochodem kdybchom se chtěli podívat na mapování psů, koček co má nulu nebo
#jedničku, tak stačí zavolat test_batches.class_indices - vrací pro každou
#class (dog, cat) INDEX kde se nachází jedničko ve one hot vector pro danou class
# např. kočka má třeba vektor [0,1] -> vrátí jedničku

