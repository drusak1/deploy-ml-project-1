bake_sharing_analysis
==============================

practice deploy ml models in production way

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

### Первое задание
 - Реализовать модель в виде sklearn-pipeline 
 - Реализовать модель в виде procedure pipeline
 - Реализовать модель в виде OOP pipeline 
 - Выбрать один пайплайн и реализовать трек экспериментов в MLFlow с подбором гиперпараметров(optune или аналогами)

**Результат**
 - Автоматизация предобработки, обучения и инференса

---


### Второе задание

- Взять пайплайн с MLflow для дальнейшей работы с [Kedro](https://kedro.readthedocs.io/en/stable/)
- Выбранный пайплайн сделать по правилам Kedro
- Реализовать Data engineering pipeline
- Реализовать Machine learning pipeline
- Связать два pipeline и получить такой же скоринг 

**Результат**
- Kredo визуализирует pipeline
- ML pipeline показывает такой же скоринг как базовый pipeline 


---

### Третье задание
- Добавить в pipeline модели шаг,который будет работать с APIDataSet
- Данный шаг должен брать данные с API и выдавать предикт
- Все должно быть оформленно в FastAPI, выкидывать результат на localhost:port 

---

### Четвертое задание
- Покрыть pipeline тестами используя python hyposises, pytest 
- Покрытие должно быть не ниже 70% кода
- Использовать GitHub Action для CI pipeline

---

### Пятое задание
- Создать Dockerfile.
- Используя dockerignore скопировать в image только нужный код(инференс + передача инфы)
- Создать Docker image c моделью
- Запустить Docker Container  на основе Docker image .
Модель должна возвращать результат по API
---

#### Шестое задание 
- создать DAG для Data engineer pipeline
- создать DAG для Machine learning pipeline
- использовать docker compore для деплоя airflow и postgres
---
### Седьмое задание
- поставить grafana и настроить дашборд
---

### Восьмое задание 
- Построить дашборды,которые чекают качество модели
- Чекать данные на Data shift, Concept Shift с помощью [Evedantly](https://github.com/evidentlyai/evidently)

---
