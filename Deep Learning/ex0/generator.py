import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False,
                 shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.image_set = []
        self.label_set = []

        with open(self.label_path) as f:
            self.data = json.load(f)

        for key, value in self.data.items():
            self.image_set.append(key + ".npy")
            self.label_set.append(value)

        self.data_size = len(self.data)
        self.indices = np.arange(self.data_size)
        self.index = 0
        self.epoch = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        images = []
        labels = []
        # images before resizing and augmenting
        img = []
        # images after resizing
        resized_image = []
        # images after augmenting
        augmented_image = []

        counter = self.index + self.batch_size

        if counter > (self.data_size - 1):
            counter -= self.data_size
            for i in range(self.index, self.data_size):
                img.append(np.load(self.file_path + "/" + self.image_set[i]))
                labels.append(self.label_set[i])

            if counter:
                self.epoch += 1
                if self.shuffle:
                    shuffled_images = []
                    shuffled_labels = []
                    np.random.shuffle(self.indices)
                    for i in range(self.data_size):
                        shuffled_images.append(self.image_set[self.indices[i]])
                        shuffled_labels.append(self.label_set[self.indices[i]])
                    self.image_set = shuffled_images
                    self.label_set = shuffled_labels

            for i in range(0, counter):
                img.append(np.load(self.file_path + "/" + self.image_set[i]))
                labels.append(self.label_set[i])

            self.index = counter

        else:
            if self.index == 0:
                self.epoch += 1
                if self.shuffle:
                    shuffled_images = []
                    shuffled_labels = []
                    np.random.shuffle(self.indices)
                    for i in range(self.data_size):
                        shuffled_images.append(self.image_set[self.indices[i]])
                        shuffled_labels.append(self.label_set[self.indices[i]])
                    self.image_set = shuffled_images
                    self.label_set = shuffled_labels

            for i in range(self.index, counter):
                img.append(np.load(self.file_path + "/" + self.image_set[i]))
                labels.append(self.label_set[i])

            self.index = counter

        # resizing images
        for i in img:
            image = resize(i, (self.image_size[0], self.image_size[1]), anti_aliasing=True)
            resized_image.append(image)

        if self.rotation or self.mirroring:
            for i in resized_image:
                image = self.augment(i)
                augmented_image.append(image)
            images = np.array(augmented_image)
        else:
            images = np.array(resized_image)

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        m = [0, 1]
        r = [1, 2, 3]

        if self.mirroring:
            img = np.flip(img, np.random.choice(m))

        if self.rotation:
            img = np.rot90(img, k=np.random.choice(r))

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch - 1

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict.get(x)

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        images, labels = self.next()
        plt.figure(figsize=(20, 20))
        print(len(images))
        for i in range(len(images)):
            plt.subplot(10, 5, i+1)
            plt.title(self.class_name(labels[i]))
            plt.imshow(images[i])
            plt.axis("off")

        plt.show()

if __name__ == '__main__':
    label_path = './Labels.json'
    file_path = './exercise_data/'
    gen = ImageGenerator(file_path, label_path, 50, [32, 32, 3], rotation=False, mirroring=False,
                         shuffle=True)
    gen.next()
    gen.next()
    gen.show()

        