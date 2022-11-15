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

``` python
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
- Результат
![ezgif-4-c4559e9e2e](https://user-images.githubusercontent.com/105049918/202020222-4e44e3b3-e3d9-4a58-97da-fe7f7eb57c01.gif)
- Настройки Юнити
![image](https://user-images.githubusercontent.com/105049918/202020844-5a441eb8-da82-4263-b0ea-9003dba10364.png)
![image](https://user-images.githubusercontent.com/105049918/202020897-d4cf133e-e7fb-4509-9823-44bd8ef36cd7.png)
- Переделанный код
```C#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target1;
    public Transform Target2;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target1.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
        Target2.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target1.localPosition);
        sensor.AddObservation(Target2.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget1 = Vector3.Distance(this.transform.localPosition, Target1.localPosition);
        float distanceToTarget2 = Vector3.Distance(this.transform.localPosition, Target2.localPosition);
        if (distanceToTarget1 < 1.42f || distanceToTarget2 < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
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
