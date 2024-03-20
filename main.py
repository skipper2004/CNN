#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycnn import PyCNN
import cv2


def main():
    # Инициализация объекта
    cnn = PyCNN()

    input_img = "images/" + input("Введите название изображения: ").strip()

    output_img = 'images/output.png'

    try:
        # Выполнение соответствующих техник обработки изображения для данного изображения
        cnn.cornerDetection(input_img, output_img)
    except Exception as f:
        print(f"Файл {input_img} не существует. Ошибка:", f)

    # Загрузка изображений
    img1 = cv2.imread(output_img, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(input_img)

    # Адаптивная бинаризация первого изображения
    thresh = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создание копии второго изображения
    overlay = img2.copy()

    # Определение коэффициента масштабирования для радиуса окружностей
    fixed_radius = int(5 * min(img2.shape[0], img2.shape[1]) / 256)

    # Обводка чёрных элементов окружностью на втором изображении
    for contour in contours:
        (x, y), _ = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        cv2.circle(overlay, center, fixed_radius, (0, 0, 255), 3)

    # Наложение обработанного изображения поверх второго
    result = cv2.addWeighted(overlay, 0.5, img2, 0.5, 0)

    # Вывод результата
    cv2.imshow('Результат', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
