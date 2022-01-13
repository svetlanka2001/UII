

from fastapi import FastAPI
from tensorflow.keras.models import load_model
import numpy as np
from numpy import asarray
from PIL import Image
from tensorflow.keras import utils # Используем для to_categorical
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from io import BytesIO

model = load_model('model_fmr_all') # Загрузка модели
test_path = '/content/gdrive/MyDrive/mnist_6.jpg' # Путь до тестового файла

def process(image_file):

  image_file=Image.open(test_path).convert('L')
  img = image_file.resize((28, 28))
  img = np.array(img)
  INPUT_SHAPE = (28, 28, 1)
  resized_image = img.reshape(INPUT_SHAPE)
  resized_image= np.expand_dims(resized_image, axis=0)
  cls_image = model.predict(resized_image)

  return str(np.argmax(cls_image))

