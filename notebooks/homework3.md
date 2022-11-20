# Домашнее задание по курсу МППР
## Задание
Необходимо создать web-приложение для классификации изображений с использованием предобученной модели на основе датасета cifar100. 

## Установка IDE PyCharm Professional
Для выполнения лабораторной работы потребуется [PyCharm Professional](https://www.jetbrains.com/pycharm/download/#section=windows//).

Студенческую лицензию можно получить, указав бауманскую почту.

## Часть 1. Сохранение модели в Google Colab
### Загрузка модели и ее экспорт в формат ONNX
#### Шаг 1
Откройте [Google colab](https://colab.research.google.com/) или создайте локальный ноутбук.
#### Шаг 2
Импортируем фреймворк.
```python
import torch
``` 
#### Шаг 3
Создаем девайс.
```python
device = torch.device('cuda'  if torch.cuda.is_available()  else  'cpu')
``` 
#### Шаг 4
Загружаем модель в соответствии с вариантом:

 - Четный номер в списке группы - cifar100_mobile;
 - Нечетный номер в списке группы - cifar100_resnet.

```python
model = torch.hub.load("chenyaofo/pytorch-cifar-models",
	"cifar100_mobilenetv2_x0_5",
	# 'cifar100_resnet20',
	pretrained=True)
```
#### Шаг 5
Загружаем модель на девайс.
```python
model.to(device)
```
#### Шаг 6
Экспортируем модель и сохраняем ее в формате ONNX.
```python
x = torch.randn(1,  3,  32,  32, requires_grad=True).to(device)

torch.onnx.export(model,  # модель
					x,  # входной тензор (или кортеж нескольких тензоров)
					"cifar100_CNN_RESNET20.onnx",  # куда сохранить (либо путь к файлу либо fileObject)
					export_params=True,  # сохраняет веса обученных параметров внутри файла модели
					opset_version=9,  # версия ONNX
					do_constant_folding=True,  # следует ли выполнять укорачивание констант для оптимизации
					input_names = ['input'],  # имя входного слоя
					output_names = ['output'],  # имя выходного слоя
					dynamic_axes={'input'  :  {0  :  'batch_size'},  # динамичные оси, в данном случае только размер пакета
					'output'  :  {0  :  'batch_size'}})
```

## Часть 2. Web-приложение классификации изображений
### Создание web-приложения для классификации изображений полученного набора данных

#### Шаг 1
Пример создания проекта Django в IDE Pycharm можно просмотреть по данной [ссылке](https://github.com/iu5team/iu5web-fall-2021/blob/main/tutorials/lab4/lab4_tutorial.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5-%D1%83%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D1%8F-%D0%BF%D0%BE-%D0%B2%D1%8B%D0%BF%D0%BE%D0%BB%D0%BD%D0%B5%D0%BD%D0%B8%D1%8E-%D0%BB%D0%B0%D0%B1%D0%BE%D1%80%D0%B0%D1%82%D0%BE%D1%80%D0%BD%D0%BE%D0%B9-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B-4//).

#### Шаг 2
После создания проекта требуется создать в корне проекта папку media для последующего сохранения изображений и файлов формата ONNX. Внутри папки media необходимо создать папки "images" и "models".

#### Шаг 3
В файл setting.py требуется добавить пути к ранее созданной папке media. В самом конце необходимо добавить следующий блок кода:
```python
import os

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media')
```

#### Шаг 4
Добавить Python файл views.py в ту же папку, где был файл setting.py.

```python
from django.shortcuts import render  
from django.core.files.storage import FileSystemStorage  
import onnxruntime  
import numpy as np  
from PIL import Image  
from io import BytesIO  
import base64  
from torchvision import transforms   
  
imageClassList = {0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'cra', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'}  #Сюда указать классы  
  
  
def scoreImagePage(request):  
    return render(request, 'scorepage.html')  
  
def predictImage(request):  
    fileObj = request.FILES['filePath']  
    fs = FileSystemStorage()  
    filePathName = fs.save('images/'+fileObj.name,fileObj)  
    filePathName = fs.url(filePathName)  
    modelName = request.POST.get('modelName')  
    scorePrediction, img_uri = predictImageData(modelName, '.'+filePathName)  
    context = {'scorePrediction': scorePrediction, 'filePathName': filePathName, 'img_uri': img_uri}  
    return render(request, 'scorepage.html', context)  
  
def predictImageData(modelName, filePath):  
    img = Image.open(filePath).convert("RGB")  
    resized_img = img.resize((32, 32), Image.ANTIALIAS)  
    img_uri = to_data_uri(resized_img)  
    input_image = Image.open(filePath)  
    preprocess = transforms.Compose([  
        transforms.Resize(32),  
  transforms.CenterCrop(32),  
  transforms.ToTensor(),  
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
  ])  
    input_tensor = preprocess(input_image)  
    input_batch = input_tensor.unsqueeze(0)  
     
 sess = onnxruntime.InferenceSession(r'C:\PRIS_DZ1\PRIS_DZ1\media\models\cifar100.onnx') #<-Здесь требуется указать свой путь к модели  
  outputOFModel = np.argmax(sess.run(None, {'input': to_numpy(input_batch)}))  
    score = imageClassList[outputOFModel]  
  
    return score, img_uri  
  
def to_numpy(tensor):  
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()  
  
def to_image(numpy_img):  
    img = Image.fromarray(numpy_img, 'RG')  
    return img  
  
def to_data_uri(pil_img):  
    data = BytesIO()  
    pil_img.save(data, "JPEG")  # pick your format  
  data64 = base64.b64encode(data.getvalue())  
    return u'data:img/jpeg;base64,' + data64.decode('utf-8')
```

##### Шаг 4.1
Необходимо поставить ограничение на 3 класса по варианту  

 - Класс 1 - номер в группе + номер группы   
 - Класс 2 - номер в группе + номер группы + 30
 - Класс 3 - номер в группе + номер группы + 60
**Где номер группы - 1, 2, 3, 4 и т.д.

#### Шаг 4.2
 Использовать модель по варианту: 
 - Четный номер в списке группы - cifar100_mobile 
 - Нечетный номер в списке группы - cifar100_resnet
```python
sess = onnxruntime.InferenceSession(r'C:\PRIS_DZ1\PRIS_DZ1\media\models\cifar100.onnx') #<-Здесь требуется указать свой путь к модели
```
#### Шаг 5
В файле urls.py требуется заменить содержимое на следующий блок кода:

```python
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.scoreImagePage, name='scoreImagePage'),
    path('predictImage', views.predictImage, name='predictImage'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

#### Шаг 6
Установить следующие библиотеки: onnx, onnxruntime, numpy, pillow.
Установить библиотеки можно в терминала PyCharm с помощью pip install [название библиотеки].

#### Шаг 7
В папку templates добавить файл scorepage.html.

```html
<!DOCTYPE html>  
<html lang="ru">  
<head>  
<meta charset="windows-1251">  
<title>DZ1</title>  
<style>  
  /* Add some padding on document's body to prevent the content  
 to go underneath the header and footer */  body{  
        padding-top: 60px;  
  padding-bottom: 40px;  
  }  
    .fixed-header, .fixed-footer{  
        width: 100%;  
  position: fixed;  
  background: #333;  
  padding: 10px 0;  
  color: #fff;  
  }  
    .fixed-header{  
        top: 0;  
  }  
    .fixed-footer{  
        bottom: 0;  
  }  
    .container{  
        width: 80%;  
  margin: 0 auto; /* Center the DIV horizontally */  
  }  
    nav a{  
        color: #fff;  
  text-decoration: none;  
  padding: 7px 25px;  
  display: inline-block;  
  }  
</style>  
</head>  
<body>  
 <div class="fixed-header">  
 <div class="container">  
  
 </div> </div> <div class="container">  
 <form action="predictImage" method="post" enctype="multipart/form-data">  
  {% csrf_token %}  
  
            <div class="col-md-4 col-sm-4">  
 <label for="FilePath">Select:</label>  
 </div> <input name="filePath" type="file"><br><br>  
 <input type="submit" value="Submit" >  
 </form> </div> <div> <br>  {% if scorePrediction %}  
        <h3>The classification is : {{scorePrediction}}</h3>  
  {% endif %}  
  
    </div>  
  
 <div>  {% if scorePrediction %}  
        <img src="{{ img_uri }}">  
  {% endif %}  
    </div>  
 <div class="fixed-footer">  
 <div class="container"></div>  
 </div></body>  
</html>
```
#### Шаг 8
Запустить проект, выполнив в терминале PyCharm следующую команду: "python3 manage.py runserver".

#### Шаг 9
Загрузить изображение и нажать на кнопку `submit`.
