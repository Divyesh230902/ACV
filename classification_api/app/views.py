from django.shortcuts import render
from django.shortcuts import render
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from django.core.files.base import ContentFile
import os
import matplotlib
matplotlib.use("agg")
from .models import image_storage
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializer import image_serializer
import tensorflow as tf

# Create your views here.

def fruit_classify(img):
    DATA_DIR = "/Users/divyeshpatel/Desktop/Coding/7th_sem/ACV/Fruit/fruits-360_dataset/fruits-360/Training/"
    IMG_SIZE = 100

    CATEGORIES = [i for i in os.listdir(DATA_DIR) if i != ".DS_Store"]

    model = tf.keras.models.load_model("/Users/divyeshpatel/Desktop/Coding/7th_sem/ACV/fruit_model.model/")
    img = cv.imdecode(np.fromstring(img.read(), np.uint8), cv.IMREAD_UNCHANGED)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    prediction = model.predict([img])
    name_fruit = CATEGORIES[np.argmax(prediction[0])]
    img = img.reshape(IMG_SIZE, IMG_SIZE, 3)
    print(name_fruit)
    return name_fruit,img
    


def get_form(request):
    return render(request,'addimage.html')

@api_view(['POST'])
def get_data(request):
    if request.method == "POST":
        print("hello post")
        image = request.FILES["image"]
        print(image)
        fruit,final_img = fruit_classify(image)
        print(fruit)
        f_img = cv.imencode('.jpg', final_img)
        final_img = ContentFile(f_img[1].tobytes())
        db_object = image_storage.objects.create(fruit_output=fruit)
        db_object.img.save(image.name,final_img)
        db_object.save()
        ob = image_serializer(db_object)
        return Response(ob.data)