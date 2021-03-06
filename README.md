# Распознавание эмоций по выражению лица

<p align="center"><img src="/img/multifaces.jpg" width="500" alt="Valence-Arousal"></p>

## Оглавление
  * [Введение в проблематику задачи](#введение-в-проблематику-задачи)
  * [Подходы к решению задачи](#подходы-к-решению-задачи)
    + [Задача классификации](#задача-классификации)
    + [Задача регрессии](#задача-регрессии)
  * [Реализованные в проекте решения](#реализованные-в-проекте-решения)
    + [Классификация эмоций](#классификация-эмоций)
    + [Регрессия эмоций](#регрессия-эмоций)
  * [Прикладное применение обученных моделей](#прикладное-применение-обученных-моделей)
    + [Основные компоненты практической реализации](#основные-компоненты-практической-реализации)
      - [Детектирование лиц](#детектирование-лиц)
      - [Модель классификации эмоций](#модель-классификации-эмоций)
      - [Обработка предсказаний и визуализация результата](#обработка-предсказаний-и-визуализация-результата)
    + [Распознавание эмоций на фотографии](#распознавание-эмоций-на-фотографии)
    + [Распознавание эмоций в видео-файле](#распознавание-эмоций-в-видео-файле)
    + [Распознавание эмоций в realtime-видео с веб-камеры](#распознавание-эмоций-в-realtime-видео-с-веб-камеры)
  * [Зависимости](#зависимости)
  * [Особенности публикации на GitHub](#особенности-публикации-на-github)
  * [Направления для дальнейшей работы](#направления-для-дальнейшей-работы)


## Введение в проблематику задачи
Распознавание эмоций по выражению лица одного человека или группы лиц - рабочий способ автоматизированного определения эмоционального состояния респондентов. Решение может применяться как в случае необходимости оценки и классификации эмоциональной реакции исследуемого на изучаемое явление (рекламный видеоролик или трейлер фильма), так и в клиентском сервисе для оценки уровня удовлетворенности клиентов (разница между средними настроениями входящих в кафе и выходящих из кафе посетителей может служить для оценки удовлетворенности гостей обслуживанием и кухней).

## Подходы к решению задачи
### Задача классификации
Основным способом определения эмоции по выражению лица является решение задачи классификации изображения с использованием сверточных нейронных сетей, с обучением сверточной модели на датасете с разметкой изображение-эмоция. Также обучающий датасет может быть разделен на каталоги фото с эмоциями, где имена каталогов совпадают с названиями эмоции, фото которой содержатся в папке. Такой структуру файлов достаточно для формирования обучающего датасета.

### Задача регрессии
Одной из множества формальных моделей эмоций является двумерная Valence-Arousal модель Джеймса Рассела, сформулированная в 1980 году. Согласно этой модели, любая эмоция может быть разделена на две составляющие - Valence (настроение, положительное-отрицательное) и Arousal (возбуждение, сильное-слабое). Разложение эмоций на составляющие в этом пространстве будет выглядеть следующим образом:
<p align="center"><img src="/img/valence-arousal.jpg" width="500" alt="Valence-Arousal"></p>
При таком подходе задача определения эмоции сводится к задаче регрессии с предсказанием двух численных значений - координат точки в указанном на изображении пространстве.
Сложностью решения такой задачи является необходимость разметки обучающего датасета с указанием численных значений Valence и Arousal. Модель AffectNet, содержащая данную информацию, недоступна для массового использования, что затрудняет применение данного решения. При этом, в случае наличия требуемой разметки, подход позволяет получить более точную количественную оценку любого произвольного эмоционального состояния изучаемых лиц, не ограничиваясь только качественным анализом и отнесением эмоции к одной из заранее определенных категорий.
Я использовал численные значения эмоций из работы "Moving Faces, Looking Places: Validation of the Amsterdam Dynamic Facial Expression Set" под авторством Job van der Schalk, Skyler T. Hawk, Agneta H. Fischer, and Bertjan Doosje. В ней используется шкала значений от 1 до 7 по каждой шкале (4 - нейтральное значение). Я выбрал эту модель, поскольку в ней мне удалось найти наибольшее количество эмоций из имеющегося обучающего датасета:
<p align="center"><img src="/img/valence-arousal_values.jpg" width="300" alt="Valence-Arousal"></p>


## Реализованные в проекте решения
### Классификация эмоций
Для задачи классификации эмоций я использовал преобученную модель VGGFace на базе ResNET-50. Я выбрал эту модель, поскольку она изначально основана для работы с лицами и требует минимального времени на дообучение под имеющийся датасет. К модели с `include_top = False` я добавил Flatten-слой, затем - один скрытый полносвязный слой на 512 нейронов, слой регуляризации Dropout со значением 0.25, а затем выходной слой на 9 нейронов по количеству эмоций в обучающем датасете, возвращающий предсказания в виде логитов. Я разморозил и дообучил последние 13 слоев исходной модели, веса остальных слоев остались в исходном состоянии. Подготовка и обучение модели производились с использованием ноутбука `1. VGGFace_finetuning.ipynb`.
Время инференса такой модели составило 26мс. Этого достаточно, чтобы можно было классифицировать эмоцию на реал-тайм видео с веб-камеры на каждом кадре (время инференса меньше, чем 1/25сек).

### Регрессия эмоций
Для отработки на практике использования Valence-Arousal модели я присвоил каждой фотографии случайное значение в окрестности содержащейся на фотографии эмоции. Модель также основана на VGGFace (разморожены последние 2 слоя) + Flatten + Dense(512) + Dropout(0.25), но после них я добавил полносвязный слой с двумя нейронами и линейной активацией. Feature-инжиниринг значений Valence и Arousal, а также обучение модели регрессии выполнены в ноутбуке `3. Valence-Arousal model tuning.ipynb`.

## Прикладное применение обученных моделей
### Основные компоненты практической реализации
#### Детектирование лиц
Для локализации лиц на изображении (фотографии или кадре из видео) я использовал OpenCV. Реализация детектирования лиц в данной библиотеке не использует нейросети и основана на использовании каскадов Хаара. OpenCV предлагает несколько .xml-файлов для детектирования лиц, все они хранятся в папке `\haarcascades` моего проекта. В случае, если в конкретном пользовательском окружении используемая конфигурация показывает ложные сработки или детектирует лица недостаточно точно, можно изменить используемый файл конфигурации на любой другой из доступных.

#### Модель классификации эмоций
Для использования предобученной модели в практических задачах я описал ее в классе VGGFerModel, который импортируется в \*.py скриптах через `from _models.vgg_fer import VGGFerModel`. Модель содержит два метода:
- __init__ - в нем структура модели загружается из json-файла, после чего в нее загружаются сохраненные веса. Конструктор класса может вызываться как `VGGFerModel()` для предсказания только названия эмоции, либо как `VGGFerModel(val_ar=True)` для предсказания эмоции и ее значений Valence-Arousal. Во втором случае внутри объекта создается 2 модели - классификации и регрессии.
- predict - отвечает за предсказание эмоции. Метод получает на вход изображение любого размера и преобразует ее в 224х224, на которых обучены модели предсказания. Если модель была инициализирована без val_ar= True, в качестве предсказания возвращается текстовое значение предсказанной эмоции. Если был указан параметр val_ar= True, метод возвращает список из \[наименование эмоции, значение Valence, значение Arousal\].

#### Обработка предсказаний и визуализация результата
За визуализацию bounding box вокруг детектированного лица и подпись распознанной эмоции отвечает функция draw_bbox_with_emotion из `from image_proc_func.functions import draw_bbox_with_emotion'. Функция получает на вход исходное изображение, модель для распознавания эмоции, а также результаты OpenCV по детектированию лица - top-left-x, top-left-y, width и heigth.
Если указанная модель возвращает str с названием эмоции - рисуется bbox стандартного зеленого цвета, наименование эмоции подписывается над его левым верхним углом. Если указанная модель возвращает список - то цвет рамки определяется настроением эмоции:
- \[1, 4\) - красный
- 4 - серый
- (4, 7\]  - зеленый.
В caption рамки при этом подставляются рассчитанные значения valence и arousal.

### Распознавание эмоций на фотографии
Для распознавания эмоции по фотографии необходимо использовать скрипт `foto_fer.py`.
Обязательным параметром запуска является путь к файлу, на котором мы ищем эмоции. Пример запуска - `python foto_fer.py ./test_photo/anger.jpg` В этом случае определяется, какая из 9 эмоций представлена на изображении. Распознавание эмоции выполняется  быстрее, потому что инициализируется только модель классификации.
<p align="center"><img src="/img/detected_anger.jpg" width="300" alt="Detected anger"></p>
Дополнительным параметром является флаг `va`. Пример запуска - `python foto_fer.py ./test_photo/anger.jpg va` 
<p align="center"><img src="/img/detected_anger_va.jpg" width="300" alt="Detected anger with Valence-Arousal values"></p>
В этом случае модель классификации инициализируется из двух моделей, на изображении отображается название эмоции и ее численные значения.Распознавание эмоции выполняется  быстрее, потому что инференс осуществляется последовательно на двух моделях.

### Распознавание эмоций в видео-файле
Для распознавания эмоции в видеофайле необходимо использовать скрипт `video_fer.py`.
Обязательным параметром запуска является путь к файлу, на котором мы ищем эмоции. Пример запуска - `python foto_fer.py ./test_photo/anger.jpg` В этом случае определяется, какая из 9 эмоций представлена в кадре.
<p align="center"><img src="/img/emotion_from_video.jpg" width="300" alt="Emotion from video"></p>
Дополнительным параметром является флаг `va`. Пример запуска - `python foto_fer.py ./test_photo/anger.jpg va` В этом случае модель классификации инициализируется из двух моделей, в кадре отображается название эмоции и ее численные значения.
<p align="center"><img src="/img/va_from_video.jpg" width="300" alt="Emotion from video with Valence-Arousal values"></p>

### Распознавание эмоций в realtime-видео с веб-камеры
Для определения эмоции в режиме реального времени в видеопотоке с веб-камеры используется только модель классификации эмоции с отнесением ее к одному из 9 имеющихся классов.
Для запуска использовать `python webcam_fer.py`

## Зависимости
- Tensorflow
- OpenCV
- NumPy
- PIL
- sys

## Особенности публикации на GitHub
Поскольку мой проект содержит 2 файла с весами преобученных моделей, каждый около 100мбайт, а GitHub имеет ограничения на объем загружаемого файла в 10мбайт, я был вынужден изучить и применить [Git-LFS](https://git-lfs.github.com/), обеспечивающий поддержку файлов большого объема. 

## Направления для дальнейшей работы
- Объединить все три скрипта в один, с определением типа классификации по передаваемым при запуске параметрам
- Добавить использование архитектуры YOLO
- Получить реальную valence-arousal разметку для получения правильного предсказания численных значений эмоции
