# (Obsidian Notes) cnn-horse-or-human-binary-classification
____
> Here we will build a model on the [Horse or Human](https://www.tensorflow.org/datasets/catalog/horses_or_humans) dataset. We will use the [[ImageDataGenerator]] class to prepare this dataset and feed it to the [[Convolutional Neural Networks]]. 

#### Getting Dataset Ready
___
First we will download the Horse or Human dataset.

```python
!wget https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip
```

Then we will unzip the file using the `zipfile` module.

```python
import unzipfile

# Unzip the dataset
local_zip = './horser-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(local_zip)
zip_ref.close()
```

The contents of the .zip file are extracted to the base directory `./horse-or-human`, which in turn contains `horses` and `humans` subdirectory. 

We do not explicitly label the images as a horse or humans. We will use the [[ImageDataGenerator]] API instead - this code will automatically label the images according to the directory names and structure. 

We can define each of these directories:

```python
import os

# Directory with our training horse images
train_horse_dir = os.path.join('./horse-or-human/horses')

# Directory with our training humans images
train_human_dir = os.path.join('./horse-or-human/humans')
```

We can now see what filenames look like in the `horses` and `humans` training directories:

```python
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])
```

We can also find out the total number of images for each directory:

```python
print(len(os.listdir(train_horse_dir)))
```

#### Displaying Dataset Images
___
Now we can take a look at a few images in the dataset using `matplotlib`

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters of our graph; we'll output images in a 4x4 configuration
nrows = 4,
ncols = 4, 

# Index for iterating over images
pic_index = 4
```

``` python
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8

next_horse_pix = [os.path.join(train_horse_dir, fname)
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                for fname in train_human_names[pic_index-8:pic_index]]
for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
```


#### Building a Small Model from Scratch
___
Now we can define the model architecture that we want to train. We will add convolutional layers as in the previous example, and flatten the final result to feed into the densely connected layers.
Because this is a *binary classification problem*, we will end our network with a [[Sigmoid activation]]. This makes the network a single scaler between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0)

```python
model = tf.keras.models.Sequential([
	# The input shape is the desire size of the image 300x300 with 3 byte colours.
	# This is the first convolution
	tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3))
	tf.keras.layers.MaxPooling2D(2,2),
	# The second convolution
	tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2)
	# The third convolution
	tf.keras.layers.Conv2D(64, (3,3), activation='relu')
	tf.keras.layers.MaxPooling(2,2)
	# the fourth convolution
	tf.keras.layers.Conv2D(64, (3,3), activation='relu')
	tf.keras.layers.MaxPooling(2,2)
	# the fith convolution
	tf.keras.layers.Conv2D(64, (3,3), activation='relu')
	tf.keras.layers.MaxPooling(2,2)
	# Flatten the result to feed into a DNN
	tf.keras.layers.Flatten()
	# 512 neuron hidden layers
	tf.keras.layers.Dense(518, activation='relu')
	# Only 1 output neuron.
	tf.keras.layers.Dense(1, activation='sigmoid')
])
```

> In the code above, the model architecture consists of multiple convolutional layers, each with a different number of filters. This is a common approach in convolutional neural networks (CNNs) to gradually increase the complexity and abstraction of the learned features.

We can review the network architecture and the output shapes with `model.summary()`

```python
model.summary()
```

Next, we will configure the specification for the model training. We will train the model with `binary_crossentropy` loss because it's a binary classification problem, and the final activation is [[Sigmoid activation]]. 

Next, you'll configure the specifications for model training. You will train the model with the [`binary_crossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) loss because it's a binary classification problem, and the final activation is a sigmoid. (For a refresher on loss metrics, see this [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture).) You will use the `rmsprop` optimizer with a learning rate of `0.001`. During training, you will want to monitor classification accuracy.

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001),
			 metrics=['accuracy'])
```

#### Data Processing
___
The next step is to set up the data generator that will read pictures in our source folder, convert them to `float32` [[tensors]], and feed them (with their labels) to the model. You will have one generator for the training images and one for the validation images. These generators will yield batches of images of size 300x300 and their labels (binary).

The image will also be normalised  in someway to make it more amenable to processing by the network (it is uncommon to feed raw pixels to [[Convolutional Neural Networks]]). In this case we will preprocess the image by normalising the pixel values to be in the `[0,1]` range.  

In [[Keras]] this can be done via the `keras.preprocessing.image.ImageDataGenerator` class using the `rescale` parameter. This `ImageDataGenerator` class allows you to instantiate the generator of augmented image batches (and their labels). `.flow(data,labels)` or `.flow_from_directory(directory)`

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
		'./horse-or-human',
		target_size=(300,300),
		batch_size=128,
		class_mode = 'binary'
)
```


#### Training
____
We can train for 15 epochs - this might take a few minutes to run.
The `loss` and `accuracy` are great indicators of progress in training. `loss` measures the current model prediction against the known labels, calculating the result. `accuracy`, on the other hand, is the portion of correct guesses.

```python
history = model.fit(
	train_generator,
	steps_per_epoch=8,
	epochs=15,
	verbose=1,
)
```

#### Make Prediction
___
```python
import numpy as np
from google.colab import files
from tensorflow.keras.utils import load_img, img_to_array

uploaded = file.upload()

for fn in uploaded.keys():
	# predict images
	path = '/content/' + fn
	img = load_img(path, target_size=(300,300))
	x = img_to_array(img)
	x = x / 255
	x = np.expand_dims(x, axis=0)

	images = np.vstack([x])
	classes = model.predict(images, batch_size=10)
	print(classes[0])

	if classes[0]>0.5:
		print(fn + "is a human")
	else:
		print(fn + "is a horse")
```


#### Adding a Validation Set
___
First we will need to import the validation set and store it in the validation directory:

```python
# Download the validation set
!wget https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip
```

We can then unzip this file as done previously

```python
import zipfile

local_zip = './validation-horse-or-human.zip'
zip_ref = zip.file(local_zip, 'r')
zip_ref.extractall('validation-horse-or-human')
zip_ref.close()
```

Now we will setup a data generator for the validation set

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = validation_datagen.flow_from_directory(
		'./validation-horse-or-human',
		target_size=(300,300),
		batch_size=32,
		class_mode='binary'
)
```

Then we can add the validation generator to the training using the `validation_data` and `validation_steps` parameters.

```python
history = model.fit(
	train_generator,
	steps_per_epoch=8,
	epochs=15,
	verbose=1,
	validation_data = validation_generator,
	validation_steps=8
)
```
