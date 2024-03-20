#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import scipy.signal as sig
import scipy.integrate as sint
from PIL import Image as img
import numpy as np
import os.path
import warnings
from skimage.filters import threshold_otsu

SUPPORTED_FILETYPES = (
    'jpeg', 'jpg', 'png', 'tiff', 'gif', 'bmp',
)

warnings.filterwarnings('ignore')  # Игнорировать незначительные предупреждения


class PyCNN:
    """Обработка изображений с помощью ячеистых нейронных сетей (CNN).

    Ячеистые нейронные сети (CNN) - это парадигма параллельных вычислений,
    впервые предложенная в 1988 году. Ячеистые нейронные сети похожи на
    нейронные сети, но отличаются тем, что коммуникация разрешена только между
    соседними элементами. Обработка изображений - одно из их применений.

    Эта библиотека Python является реализацией CNN для применения в
    обработке изображений.

    Атрибуты:
        n (int): Высота изображения.
        m (int): Ширина изображения.
    """

    def __init__(self):
        """Устанавливает начальные атрибуты класса m (ширина) и n (высота)."""
        self.m = 0  # ширина (количество столбцов)
        self.n = 0  # высота (количество строк)

    def f(self, t, x, Ib, Bu, tempA):
        """Вычисляет производную x в момент времени t.

        Аргументы:
            x: Входные данные.
            Ib (float): Системный сдвиг.
            Bu: Свертка шаблона управления с входом.
            tempA (:obj:`list` of :obj:`list`of :obj:`float`): Обратная
                связь шаблона.
        """
        x = x.reshape((self.n, self.m))
        dx = -x + Ib + Bu + sig.convolve2d(self.cnn(x), tempA, 'same')
        return dx.reshape(self.m * self.n)

    def cnn(self, x):
        """Кусочно-линейная сигмоидная функция.

        Аргументы:
            x : Вход для кусочно-линейной сигмоидной функции.
        """
        return 0.5 * (abs(x + 1) - abs(x - 1))

    def validate(self, inputLocation):
        """Проверяет, существует ли строка пути или поддерживается ли тип файла.

        Аргументы:
            inputLocation (str): Строка с путем к изображению.

        Вызывает:
            IOError: Если `inputLocation` не существует или не является файлом.
            Exception: Если тип файла не поддерживается.
        """
        _, ext = os.path.splitext(inputLocation)
        ext = ext.lstrip('.').lower()
        if not os.path.exists(inputLocation):
            raise IOError('Файл {} не существует.'.format(inputLocation))
        elif not os.path.isfile(inputLocation):
            raise IOError('Путь {} не является файлом.'.format(inputLocation))
        elif ext not in SUPPORTED_FILETYPES:
            raise Exception(
                'Тип файла {} не поддерживается. Поддерживаемые типы: {}'.format(
                    ext, ', '.join(SUPPORTED_FILETYPES)
                )
            )

    # tempA: обратная связь шаблона, tempB: шаблон управления
    def imageProcessing(self, inputLocation, outputLocation,
                        tempA, tempB, initialCondition, Ib, t):
        """Обрабатывает изображение с использованием входных аргументов.

        Аргументы:
            inputLocation (str): Путь к исходному изображению в виде строки.
            outputLocation (str): Путь к обработанному изображению в виде строки.
            tempA (:obj:`list` of :obj:`list`of :obj:`float`): Шаблон обратной связи.
            tempB (:obj:`list` of :obj:`list`of :obj:`float`): Шаблон управления.
            initialCondition (float): Начальное состояние.
            Ib (float): Системный сдвиг.
            t (numpy.ndarray): Массив numpy с равномерно распределенными числами,
                представляющими моменты времени.
        """

        # Конвертация в градации серого
        gray = img.open(inputLocation).convert('L')
        self.m, self.n = gray.size
        u = np.array(gray)

        # Вычисление динамического порога для бинаризации изображения
        threshold = threshold_otsu(u)

        # Применение порога для получения бинарного изображения
        u = np.where(u > threshold, 255, 0)

        z0 = u * initialCondition
        Bu = sig.convolve2d(u, tempB, 'same')
        z0 = z0.flatten()
        tFinal = t.max()
        tInitial = t.min()
        if t.size > 1:
            dt = t[1] - t[0]
        else:
            dt = t[0]
        ode = sint.ode(self.f) \
            .set_integrator('vode') \
            .set_initial_value(z0, tInitial) \
            .set_f_params(Ib, Bu, tempA)
        while ode.successful() and ode.t < tFinal + 0.1:
            ode_result = ode.integrate(ode.t + dt)
        z = self.cnn(ode_result)
        out_l = z[:].reshape((self.n, self.m))
        out_l = out_l / (255.0)
        out_l = np.uint8(np.round(out_l * 255))
        out_l = img.fromarray(out_l).convert('RGB')
        out_l.save(outputLocation)

    # Общая обработка изображений для заданных шаблонов
    def generalTemplates(self,
                         name='Обработка изображений',
                         inputLocation='',
                         outputLocation='output.png',
                         tempA_A=[[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]],
                         tempB_B=[[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]],
                         initialCondition=0.0,
                         Ib_b=0.0,
                         t=np.linspace(0, 10.0, num=2)):
        """Проверка и обработка изображения с использованием входных аргументов.

        Аргументы:
            name (str): Название шаблона.
            inputLocation (str): Путь к исходному изображению в виде строки.
            outputLocation (str): Путь к обработанному изображению в виде строки.
            tempA_A (:obj:`list` of :obj:`list`of :obj:`float`): Шаблон обратной связи.
            tempB_B (:obj:`list` of :obj:`list`of :obj:`float`): Шаблон управления.
            initialCondition (float): Начальное состояние.
            Ib_b (float): Системный сдвиг.
            t (numpy.ndarray): Массив numpy с равномерно распределенными числами,
                представляющими моменты времени.
        """
        self.validate(inputLocation)
        print(f'{name} инициализирована.')
        self.imageProcessing(inputLocation,
        outputLocation,
        np.array(tempA_A),
        np.array(tempB_B),
        initialCondition,
        Ib_b,
        t)
        print(f'Обработка изображения {inputLocation} завершена.')

    def cornerDetection(self, inputLocation='', outputLocation='output.png'):
        """Выполняет обнаружение углов на входном изображении.

        На выходе получается бинарное изображение, где черные пиксели представляют
        выпуклые углы объектов на входном изображении.

        A = [[0.0 0.0 0.0],
             [0.0 1.0 0.0],
             [0.0 0.0 0.0]]

        B = [[-1.0 -1.0 -1.0],
             [-1.0 4.0 -1.0],
             [-1.0 -1.0 -1.0]]

        z = -5.0

        Начальное состояние = 0.0

        Аргументы:
            inputLocation (str): Путь к входному изображению в виде строки.
            outputLocation (str): Путь к обработанному изображению в виде строки.
        """
        name = 'Обнаружение углов'
        tempA = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, -1.0, -1.0], [-1.0, 4.0, -1.0], [-1.0, -1.0, -1.0]]
        Ib = -5.0
        t = np.linspace(0, 10.0, num=11)
        initialCondition = 0.0
        self.generalTemplates(
            name,
            inputLocation,
            outputLocation,
            tempA,
            tempB,
            initialCondition,
            Ib,
            t)

