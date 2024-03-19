# Ячеистая нейронная сеть для поиска углов на изображении

Ячеистые нейронные сети (Cellular Neural Networks, CNN) представляют собой парадигму параллельных вычислений, впервые предложенную в 1988 году. Ячеистые нейронные сети похожи на нейронные сети, но отличаются тем, что коммуникация разрешена только между соседними единицами. Обработка изображений является одним из их применений. Процессоры CNN были разработаны для выполнения обработки изображений; в частности, первоначальное применение процессоров CNN заключалось в выполнении обработки в режиме реального времени с ультравысокой частотой кадров (>10 000 кадров/с), недостижимой для цифровых процессоров.

Эта библиотека на Python является реализацией CNN для поиска углов на изображении.

Примечание: Ячеистая нейронная сеть (CNN) не должна путаться с совершенно другой сверточной нейронной сетью (ConvNet).

## Зависимости

Библиотека поддерживается для Python >= 2.7 и Python >= 3.3.

Необходимые модули Python для использования этой библиотеки:

```
Pillow: 3.3.1
Scipy: 0.18.0
Numpy: 1.11.1 + mkl
```

## Пример

**Input:**

![](https://raw.githubusercontent.com/skipper2004/CNN/main/images/input1.bmp)

**Output:**

![](https://raw.githubusercontent.com/skipper2004/CNN/main/images/test.png)