# Моделирование вращения твердого тела с использованием уравнений Эйлера и Пуассона и 3D визуализацией

Этот репозиторий содержит Python-скрипт для моделирования вращения твердого тела вокруг неподвижной точки с использованием уравнений Эйлера и Пуассона. Модель реализована с учетом вращения в трехмерном пространстве, включает в себя 3D визуализацию, вычисление координат центра масс.

## Краткое описание проекта

Проект моделирует поведение твердого тела с произвольным тензором инерции, вращающегося вокруг неподвижной точки.  В модели учитываются:

* **Динамика вращения:**  Изменение угловой скорости тела во времени определяется уравнениями Эйлера, которые учитывают момент инерции тела, и уравнение Пуассона.
* **Кинематика вращения:** Ориентация тела в пространстве точно описывается с помощью кватернионов.
* **Визуализация:** Результаты моделирования представлены в виде интерактивной 3D-анимации, демонстрирующей вращение тела и изменение положения его центра масс.

## Теоретическая основа

Моделирование основано на двух ключевых уравнениях:

**1. Уравнения Эйлера:** Эти уравнения описывают динамику вращения твердого тела, связывая изменение угловой скорости с моментом инерции и внешними моментами.  В общем виде уравнения Эйлера записываются как:

**I₀ ⋅ dω/dt + ω x (I₀ ⋅ ω) = M**

где:

* **I₀**: Тензор инерции тела (3x3 матрица).  В общем случае,  **I₀**  имеет вид:
I₀ = [[Ixx, Ixy, Ixz], [Iyx, Iyy, Iyz], [Izx, Izy, Izz]]


где **Ixx**, **Iyy**, **Izz** - моменты инерции относительно осей x, y, z соответственно, а **Ixy**, **Ixz**, ... - центробежные моменты инерции.  Если оси координат совпадают с главными осями инерции, тензор **I₀** становится диагональным:
I₀ = [[A, 0, 0], [0, B, 0], [0, 0, C]]


где A, B, C - главные моменты инерции.  В этом случае уравнения Эйлера упрощаются (см. ниже).

* **ω**: Вектор угловой скорости (ω₁, ω₂, ω₃).
* **x**: Векторное произведение.
* **M**: Вектор внешнего момента (3x1 вектор), в этой модели для упрощения  **M = 0**.
* **.**: Обозначает скалярное произведение матриц или умножение скаляра на вектор/матрицу

В случае диагонального тензора инерции (оси координат совпадают с главными осями инерции), уравнения Эйлера значительно упрощаются:

**A ⋅ dω₁/dt = (B - C) ⋅ ω₂ ⋅ ω₃**
**B ⋅ dω₂/dt = (C - A) ⋅ ω₃ ⋅ ω₁**
**C ⋅ dω₃/dt = (A - B) ⋅ ω₁ ⋅ ω₂**

**2. Уравнения Пуассона (с использованием кватернионов):** Эти уравнения описывают кинематику вращения, то есть изменение ориентации тела во времени, используя кватернионы:

**dΛ/dt = 0.5 ⋅ Λ ο ω**

где:

* **Λ**: Кватернион, представляющий ориентацию тела (q₀, q₁, q₂, q₃).
* **ω**: Вектор угловой скорости.
* **ο**: Умножение кватернионов.
* **.**: Обозначает скалярное произведение.

Это уравнение позволяет вычислить производную кватерниона ориентации по времени.  Разложив уравнение на скалярную и векторную части, можно получить систему из четырех уравнений для компонент кватерниона.

## Структура кода

Код организован в несколько классов и функций для обеспечения модульности и читаемости:

* ****Quaternion**:** Класс для работы с кватернионами, включает методы для математических операций (сложение, умножение, сопряжение, нормализация).  Нормализация кватерниона важна для поддержания его единичной длины и предотвращения накопления ошибок при численном интегрировании.

* ****Solid_body**:** Класс, представляющий твердое тело.  Хранит информацию о тензоре инерции (**I**), начальной угловой скорости (**omega**), и начальных координатах центра масс (**R0**).  Включает методы для:
    * вычисления главных моментов инерции и осей;
    * получения координат центра масс в инерциальной системе координат (**get_coordinates**);
    * получения базиса тела в инерциальной системе координат (**get_basis**).

* ****External_force**:** Класс для задания внешних сил и вычисления момента сил (в текущей версии не используется).

* ****g(t, U)**:** Функция, реализующая систему дифференциальных уравнений (Эйлера и Пуассона) для численного интегрирования.  Она принимает текущее время **t** и вектор состояния **U** (включающий компоненты кватерниона и угловой скорости) и возвращает вектор производных.

* **Главная часть программы:** Инициализирует параметры модели, решает систему дифференциальных уравнений с помощью **scipy.integrate.solve_ivp**,  вычисляет координаты центра масс и создает 3D анимацию с использованием **matplotlib.animation**.

## Визуализация

Визуализация результатов моделирования реализована с использованием библиотеки **matplotlib.animation**, которая позволяет создавать анимацию в Python.  Анимация отображается в трехмерном пространстве и показывает:

* **Вращение системы координат, связанной с телом:**  Три взаимно перпендикулярные стрелки представляют оси x, y и z системы координат, жестко связанной с вращающимся телом.  Изменение ориентации этих стрелок во времени отображает вращение тела.

* **Траектория движения центра масс:**  Центр масс тела отмечается точкой, которая перемещается в трехмерном пространстве в соответствии с вращением.  Положение точки вычисляется с помощью кватернионов и начальных координат центра масс в системе координат, связанной с телом.  Эта траектория отображается как след точек, показывающий путь центра масс за время симуляции.

**Подробное описание визуализации:**

Анимация создается функцией **animate** в коде.  Эта функция вызывается на каждом шаге анимации функцией **FuncAnimation**.  Для каждого кадра анимации функция **animate** обновляет положение трех стрелок (осей координат тела) и добавляет новую точку к траектории центра масс.  Положение стрелок вычисляется на основе кватерниона ориентации тела, полученного из решения системы дифференциальных уравнений.

В начале анимации на графике отображаются оси координат инерциальной системы координат (красная, зеленая, синяя стрелки), что служит для контекста и сравнения с движением осей координат тела.

## Использование

Для работы кода необходимы библиотеки **numpy**, **scipy**, **matplotlib**. Запустите Python скрипт. Программа выведет 3D-анимацию вращения твердого тела, показывая изменение ориентации и траекторию движения центра масс.


## Дальнейшие улучшения

* Добавление моделирования внешних сил и моментов.
* Реализация моделирования трения.
* Улучшение 3D-визуализации (например, добавление визуальной модели тела, а не только осей координат).
* Добавление возможности изменения параметров тела (массы, формы) во время симуляции.
* Использование более точных методов численного интегрирования.
