# 05. Контроль качества и выбор модели: валидация, кросс-валидация, подбор гиперпараметров

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 04. Линейные модели](../04_linear_models/README.md) · [06. Bias–variance →](../06_bias_variance/README.md)

---

## 1. Проблема контроля качества

Ошибка на обучении (**train error**) и ошибка на новых данных (**test error**), как правило, очень различаются:
модель подстраивается под обучающую выборку, и train error **оптимистична** ([насколько — формула](deep.md#1-почему-ошибка-на-обучении-оптимистична)).
Нужен способ оценить, как алгоритм будет работать **в будущем** на реальных данных.

**Базовая идея любого контроля:** разбить выборку на обучающую (TRAIN) и контрольную (TEST),
обучить на первой, оценить на второй.

---

## 2. Три золотых правила разбиения выборки

1. **Тест/валидация моделирует реальную работу алгоритма:**
   - **деление по группам** — если в проде модель увидит новых клиентов, то и в тесте должны быть клиенты, которых не было в обучении (`GroupKFold`);
   - **предсказание будущего** — если модель прогнозирует будущее, тест должен быть позже обучения (out-of-time);
   - **стратификация** — сохранять пропорции классов (`stratify=y`).

   Схема из лекции: данные можно резать по пользователям (out of sample) и по времени (out of time):
   *in sample / out of time*, *out of sample / in time*, *out of sample / out of time*.
2. **Тест должен быть случайным** (`shuffle=True`) **или специально подготовленным** (например, убрать дубликаты, которые иначе окажутся и в train, и в test).
3. **Нет утечкам:** нельзя явно или неявно использовать метки объектов, на которых оцениваем качество.
   Отбор признаков, mean target encoding (MTE), нормировка, заполнение пропусков — **только по train-части** каждого фолда.

---

## 3. Способы контроля

### 3.1. Отложенная выборка (Hold-out)

Делим на обучение и отложенный контроль, обычно **80/20**.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)   # по порядку (время)
```

- Больше тест — надёжнее оценка; больше обучение — модель ближе к итоговой.
- Оценка **зависит от конкретного разбиения**, при другом разбиении может сильно измениться.
- Если затем переобучить модель на всех данных, оценки её ошибки у нас нет — в каком-то смысле
  неустранимый недостаток (на практике считают, что на большем объёме будет не хуже).

### 3.2. Random subsampling (Monte-Carlo CV)

$k$ раз случайно выбираем отложенную выборку и усредняем ошибки.

```python
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit
ShuffleSplit(n_splits=4, test_size=0.3)               # просто случайно
GroupShuffleSplit(n_splits=4, test_size=0.3)          # группы не разбиваются: cv.split(X, groups=g)
StratifiedShuffleSplit(n_splits=4, test_size=0.3)     # сохраняем пропорции классов: cv.split(X, y)
```

### 3.3. k-Fold кросс-валидация

1. Делим выборку на $k$ примерно равных частей (обычно $k = 5$ или $10$).
2. Цикл $i = 1..k$: $i$-я часть — тест, объединение остальных — обучение.
3. Усредняем $k$ ошибок. **Дисперсию** по фолдам можно использовать для оценки доверия к результату.

| Вариант | sklearn | Когда |
|---|---|---|
| Обычный | `KFold(n_splits=5, shuffle=True)` | регрессия, перемешанные данные |
| Стратифицированный | `StratifiedKFold(n_splits=5, shuffle=True)` | классификация, особенно с дисбалансом |
| По группам | `GroupKFold(n_splits=5)` | несколько объектов на одного клиента/пациента/сессию |
| Заданное разбиение | `PredefinedSplit` | фолды заданы заранее |
| Leave-One-Out | `LeaveOneOut()` = k-fold при $k = \ell$ | очень маленькие выборки; хорош в теории, долго считается |
| Leave-P-Out | `LeavePOut(p)` | все возможные наборы из $P$ объектов — очень дорого |
| Leave-One-Group-Out | `LeaveOneGroupOut()` | по одной группе в тест; при оценке учитывать размеры групп |
| С повторениями | `RepeatedKFold`, `RepeatedStratifiedKFold(n_splits, n_repeats)` | «сделать несколько раз» с новым перемешиванием — стабильнее оценка |

### 3.4. Бутстрэп

Выбором **с возвращением** формируем выборку того же объёма $m$, обучаемся на ней; контроль — на
объектах, которые в неё не попали (**out-of-bag**). В контроль попадает примерно
$\big(1 - \frac1m\big)^m \approx e^{-1} \approx 37\%$ выборки ([вывод](deep.md#2-бутстрэп-откуда-368)).

- модель учится на выборке **того же объёма**, что итоговая;
- используются не все данные, есть дубликаты;
- по распределению бутстрэп-выборка похожа на исходную.

Бутстрэп — основа бэггинга и случайного леса (тема [08](../08_bagging_random_forest/README.md)).

### 3.5. Контроль по времени (Out-of-Time)

Для временных рядов **нельзя допускать утечек из будущего в прошлое**.

| Схема | Как | Когда |
|---|---|---|
| Последовательный hold-out | train: 2020–2022, test: 2023–2024 | базовый вариант |
| **Expanding window** | Fold 1: train [2019] → val [2020]; Fold 2: train [2019–2020] → val [2021]; … | важен весь исторический контекст |
| **Sliding window** | Fold 1: train [2019–2020] → val [2021]; Fold 2: train [2020–2021] → val [2022]; … | старые данные теряют актуальность |

```python
from sklearn.model_selection import TimeSeriesSplit
TimeSeriesSplit(n_splits=4,
                max_train_size=None,   # ограничение train -> sliding window; None -> expanding
                test_size=None,        # по умолчанию n_samples // (n_splits + 1)
                gap=0)                 # сколько объектов «откусить» между train и test (embargo)
```

Часто много проверок не сделать — слишком короткая предыстория. Разбиение организуют «под контекст»:
прогноз на следующий день, неделю, месяц.

### 3.6. Nested (вложенная) кросс-валидация

Для **одновременного** подбора гиперпараметров и честной оценки итогового качества:
- **внешний цикл** (outer loop) оценивает обобщающую способность;
- **внутренний цикл** (inner loop) подбирает гиперпараметры/признаки на train-части внешнего фолда.

Зачем: оценка лучшей конфигурации на той же валидации, где её выбирали, **смещена вверх**
(«проклятие победителя», [deep.md](deep.md#4-почему-нужен-отдельный-тест-смещение-отбора)).

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
inner = GridSearchCV(model, param_grid, cv=KFold(5, shuffle=True, random_state=0))
outer_scores = cross_val_score(inner, X, y, cv=KFold(5, shuffle=True, random_state=1))
```

---

## 4. Выбор модели (Model Selection)

**Модель** — параметрическое семейство алгоритмов (kNN, SVM, линейная, нейросеть…). **Параметры**
настраиваются обучением; **гиперпараметры** выбираются до обучения — экспертно или перебором.
Моделей много (но конечное число), алгоритмов — бесконечно много: 1NN, 3NN, …, SVM(C=0.27), SVM(C=0.15)…

Выбираем в широком смысле **пайплайн**: модель, гиперпараметры, признаки, предобработку.

### Train / Validation / Test

| Часть | Зачем |
|---|---|
| **Train** | обучение модели (настройка параметров) |
| **Validation** | выбор пайплайна: модели, гиперпараметров, признаков; иногда — локальный контроль |
| **Test** | итоговая оценка качества выбранного решения |

Можно комбинировать: **k-Fold CV для валидации + hold-out для итогового теста**. Только валидации
доверяют редко — для уверенной оценки нужен тест, к которому не прикасались при выборе.

### Где используется кросс-валидация

- оценка качества модели;
- настройка гиперпараметров;
- **«честные» ответы на обучении** — как алгоритм отвечал бы на объекте, если бы не учился на нём
  (`cross_val_predict`; нужно для стекинга, target encoding, анализа ошибок);
- контроль алгоритмов, построение **кривых валидации** и **кривых обучения**;
- предобработка (кодирование категорий) и ансамблирование (метапризнаки).

```python
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, permutation_test_score

cross_val_score(model, X, y, cv=5, scoring="roc_auc")                       # массив метрик по фолдам
cross_validate(model, X, y, cv=5, scoring=["roc_auc", "f1"],
               return_train_score=True)                                     # + время обучения/предсказания
oof = cross_val_predict(model, X, y, cv=5, method="predict_proba")          # out-of-fold ответы
score, perm_scores, pvalue = permutation_test_score(model, X, y, cv=5)      # неслучайность результата
```

### Кривые обучения и валидации

- **Learning curve** — качество на train и CV в зависимости от **объёма** обучающей выборки:
  большой зазор между train и CV → переобучение (поможет больше данных / регуляризация);
  обе кривые низкие и сошлись → недообучение (нужна сложнее модель / признаки).
- **Validation curve** — качество от **значения гиперпараметра**: видно, где начинается переобучение.

```python
from sklearn.model_selection import learning_curve, validation_curve
sizes, tr, cv = learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8))
tr, cv = validation_curve(model, X, y, param_name="max_depth", param_range=range(1, 20), cv=5)
```

---

## 5. Подбор гиперпараметров

| Метод | Принцип | Где применим | Плюсы | Минусы |
|---|---|---|---|---|
| **Grid Search** | полный перебор по сетке | маленькие пространства, бейзлайн | простота, воспроизводимость | экспоненциальный рост времени с числом параметров |
| **Randomized Search** | случайные комбинации из распределений | большие пространства, быстрый прототип | быстрее Grid при том же бюджете ([почему](deep.md#5-почему-случайный-поиск-лучше-сетки)) | может пропустить оптимум |
| **Hyperopt** | байесовская оптимизация (TPE) | сложные иерархические пространства | эффективен, гибок | устаревший интерфейс |
| **Optuna** | байесовская оптимизация (TPE) + **pruning** | современные задачи, Kaggle, прод | early stopping проб, гибкость, визуализация | требует настройки |

Пример из лекции (kNN, accuracy % на CV):

| metric \ k | 1 | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|
| euclidean | 76.0 | 77.0 | 79.0 | 78.5 | 80.5 | **82.5** |
| manhattan | 74.0 | 74.0 | 79.0 | 79.5 | 80.5 | 81.0 |
| chebyshev | 76.5 | 78.5 | 80.0 | 80.0 | 81.0 | 81.5 |

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform, randint

grid = GridSearchCV(KNeighborsClassifier(),
                    {"metric": ["euclidean", "manhattan", "chebyshev"], "n_neighbors": [1, 3, 5, 7, 9, 11]},
                    cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X, y); grid.best_params_, grid.cv_results_["mean_test_score"]

rnd = RandomizedSearchCV(lgb.LGBMClassifier(n_estimators=100, subsample=0.75, colsample_bytree=0.75),
                         {"num_leaves": randint(2, 64), "learning_rate": loguniform(1e-2, 0.3)},
                         n_iter=50, cv=5, scoring="roc_auc", n_jobs=-1, random_state=0)
model.get_params()          # все гиперпараметры модели
```

### Optuna (из семинара по деревьям регрессии)

```python
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor

def objective(trial):
    params = {
        "max_depth":         trial.suggest_int("max_depth", 2, 25),
        "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 2, 200, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 100),
        "criterion":         trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error"]),
        "max_features":      trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "ccp_alpha":         trial.suggest_float("ccp_alpha", 1e-6, 1e-1, log=True),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(DecisionTreeRegressor(random_state=42, **params), X, y, cv=cv, scoring="r2").mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
print(study.best_value, study.best_params)
optuna.visualization.plot_optimization_history(study)
```

### Советы

- Явно указывайте метрику (`scoring`), а лучше несколько.
- Распараллеливание: `n_jobs=-1`; устойчивость к падениям: `error_score=0` (или `np.nan`).
- **Оптимизируйте весь пайплайн** (предобработка + модель), а не только модель — так нет утечек.
- Используйте современные методы (Optuna).
- Фиксируйте `random_state` разбиений, чтобы сравнения моделей были честными (на одних и тех же фолдах).

---

## 6. Типичные ошибки (утечки)

- Масштабирование / заполнение пропусков / отбор признаков **по всей выборке до** разбиения.
- Target encoding без out-of-fold.
- Случайное перемешивание временного ряда.
- Один клиент (пациент, сессия) и в train, и в test.
- Дубликаты в train и test.
- Многократный подбор гиперпараметров «по тесту» — тест превращается в валидацию.
- Выбор лучшей из сотни конфигураций и отчёт её валидационного качества как итогового.

---

## Что почитать / посмотреть

- 📘 Учебник ML Яндекса: [Кросс-валидация](https://education.yandex.ru/handbook/ml/article/kross-validaciya) · [Подбор гиперпараметров](https://education.yandex.ru/handbook/ml/article/podbor-giperparametrov)
- 📄 [sklearn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) · [Tuning hyper-parameters](https://scikit-learn.org/stable/modules/grid_search.html)
- 📄 [Optuna docs](https://optuna.readthedocs.io/en/stable/)

## Источники курса

- `Контроль качества и выбор модели.pdf` — весь (36 с.): способы контроля, три золотых правила, кривые, перебор параметров.
- `Лекция 2_072913.pdf`, с. 43–82: кросс-валидация, бутстрэп, временные ряды, nested CV, подбор гиперпараметров (таблица методов).
- Colab-семинар «Деревья регрессии»: подбор гиперпараметров `DecisionTreeRegressor` через Optuna.


<!-- materials:start -->
**Файлы курса в репозитории:**

- Лекция 2 (ML): метрики классификации (лекция курса, файл в репозиторий не включён)
- Контроль качества и выбор модели (лекция курса, файл в репозиторий не включён)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [06. Сложность моделей и bias–variance →](../06_bias_variance/README.md)
