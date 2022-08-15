from keras.preprocessing.image import load_img, ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np


img = load_img(r"havuzDazzle.jpg")

print(type(img))
print(img.format)
print(img.mode)
print(img.size)

data = img
samples = np.expand_dims(data, 0)
datagen = ImageDataGenerator(width_shift_range=[-200, 200], height_shift_range=[-200, 200])
it = datagen.flow(samples,batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

datagen = ImageDataGenerator(horizontal_flip=True)
it = datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

datagen = ImageDataGenerator(rotation_range=90)
it = datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
it = datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
it = datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show() 