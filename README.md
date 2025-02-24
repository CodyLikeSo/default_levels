# Dataset: data.csv

# Tasks: 
- build a model that will produce a def default rate of no more than 3.5% on the test sample

- make a graph of the dependence of the approval level on the cutoff level (i.e., the default level)

# Solution:
To start skrip.py fallow this steps:
1. Clone repo -> ```git clone https://github.com/CodyLikeSo/default_levels.git``` and then ```cd default_levels/```
2. Create VENV if needed: ```python -m venv .venv``` and activate ```source .venv/bin/activate```
3. Install packages from requirements.txt -> ```pip install -r requirements.txt```
4. start -> ```python sript.py```


## Description(rus):

1. В данной задаче решается задача бинарной классификации с использованием модели логистической регрессии. Сперва я преоброзовал файл .xlsb в .csv через оналйн ресурс ```https://products.aspose.app/cells/conversion``` (Заново конвертировать не нужно - data.csv уже в репозитории). 
2. Цель - предсказать уровень дефолта, не превышающий 3,5 % на тестовой выборке. Набор данных обрабатывается путем удаления нулевых столбцов, заполнения пропущенных значений и кодирования категориальных признаков в числовые.

3. Данные делятся на обучающую и тестовую выборки в соотношении 80:20. Затем модель обучается и используется для прогнозирования вероятностей на тестовом множестве. Для каждого порога рассчитываются соответствующие показатели дефолта и одобрения, и определяется порог, при котором показатель дефолта не превышает заданные 3,5 %.

4. В завершение проекта был построен график, на котором отображается процент одобрения относительно пороговых значений

5. Хотя модель логистической регрессии эффективно выполняет требования задачи, она может плохо справляться с нелинейными закономерностями, что может снизить ее точность при работе со сложными данными. Кроме того, в текущей модели отсутствует оценка эффективности и перекрестная валидация, и эти аспекты могут быть рассмотрены для дальнейшего улучшения. Обработка пропущенных значений также может внести погрешность, и поэтому требует тщательного анализа.
