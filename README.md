# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
- Ахидов Роман Игоревич
- РИ210934
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.
В данной лабораторной работе мы создадим ML-агент и будем тренировать нейросеть, задача которой будет заключаться в управлении шаром. Задача шара заключается в том, чтобы оставаясь на плоскости находить кубик, смещающийся в заданном случайном диапазоне координат.
## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity.
- Установил ML Agents
![image](https://user-images.githubusercontent.com/105049918/200978242-6c8911b9-6e68-4f68-bfb6-8ef4862c9ab9.png)
- Создал и активировал новый ML Agent
![image](https://user-images.githubusercontent.com/105049918/200979224-940480e4-2beb-48b4-a9e6-8f12cbf4e57c.png)
- Скачал и установил библиотеку torch
![image](https://user-images.githubusercontent.com/105049918/200979329-e7104eb1-b292-456d-809f-f4767d80e1c9.png)
- Настроил все файлы и скрипты, запустил сцену
![image](https://user-images.githubusercontent.com/105049918/201990525-27ace069-a489-401f-9327-3eb5bb439abd.png)
![image](https://user-images.githubusercontent.com/105049918/201990636-a3636b24-fd4d-45c5-9a85-c92befc53e07.png)
- Сделал 3 копии модели
![image](https://user-images.githubusercontent.com/105049918/201991317-e5bd6549-ec4c-42e0-83da-244a3632de0c.png)
- Сделал 9 копий модели
![image](https://user-images.githubusercontent.com/105049918/201992146-e4a58120-075e-4e62-af7e-7bae5a98a57e.png)
- Сделал 27 копий модели
![image](https://user-images.githubusercontent.com/105049918/201992908-2713ca1e-8061-41cc-ae4b-2c3524bd9ef3.png)

## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети, доступного в папке с файлами проекта по ссылке. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных на сфере.

```
behaviors: #
  RollerBall: #Объект
    trainer_type: ppo #Тип тренера
    hyperparameters: # Гиперпараметры
      batch_size: 10 # Количество опытов в каждой итерации градиентного спуска
      buffer_size: 100 # Количество опытов, которые необходимо собрать перед обновлением модели политики
      learning_rate: 3.0e-4 # Начальная скорость обучения для градиентного спуска. Соответствует силе каждого шага обновления градиентного спуска
      beta: 5.0e-4 # Сила регуляризации энтропии, которая делает политику «более случайной»
      epsilon: 0.2 # Допустимый порог расхождения между старой и новой политикой при обновлении градиентного спуска
      lambd: 0.99 # Параметр регуляризации
      num_epoch: 3 # Количество проходов через буфер опыта при выполнении оптимизации градиентного спуска
      learning_rate_schedule: linear # Определяет, как скорость обучения изменяется с течением времени
    network_settings: # Спецификации сети
      normalize: false # Применяется ли нормализация к входным данным векторных наблюдений
      hidden_units: 128 # Количество единиц в скрытых слоях нейронной сети
      num_layers: 2 # Количество скрытых слоев в нейронной сети
    reward_signals: # Настройки сигналов вознаграждения
      extrinsic: # Внешние награды
        gamma: 0.99 # Фактор скидки для будущих вознаграждений, поступающих из окружающей среды
        strength: 1.0 # Коэффициент, на который умножается вознаграждение, данное средой
    max_steps: 500000 # Общее количество шагов
    time_horizon: 64 # Сколько шагов опыта необходимо собрать для каждого агента, прежде чем добавить его в буфер опыта
    summary_freq: 10000 # Количество опытов, которое необходимо собрать перед созданием и отображением статистики обучения

```
- Компонент DecisionRequester предоставляет удобный и гибкий способ запуска процесса принятия решения агентом. Без DecisionRequester реализация вашего агента должна вручную вызывать функцию RequestDecision().
- Behavior Parameters - компонент для настройки поведения экземпляра агента и свойств мозга.

## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.
![video](https://user-images.githubusercontent.com/105049918/202019445-1f81d197-8457-41cd-bdb7-097f6a337aa3.mp4)

```py

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.NewProject()
oProject.Rename("C:/Users/denisov.dv/Documents/Ansoft/SphereDIffraction.aedt", True)
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.CreateSphere(
	[
		"NAME:SphereParameters",
		"XCenter:="		, "0mm",
		"YCenter:="		, "0mm",
		"ZCenter:="		, "0mm",
		"Radius:="		, "1.0770329614269mm"
	], 
)

```

## Выводы

Абзац умных слов о том, что было сделано и что было узнано.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
