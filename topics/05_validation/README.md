# 05. Контроль качества и выбор модели

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Ошибка на обучении оптимистична, поэтому качество оценивают на данных, **не участвовавших
в обучении**: hold-out, k-fold CV, бутстрэп, контроль по времени. Гиперпараметры выбирают на валидации,
итоговое качество — на отдельном тесте (или nested CV). Главное — разбиение должно **моделировать
реальное применение** и не допускать **утечек**.

| Термин | Определение |
|---|---|
| Hold-out | одно разбиение train/test (обычно 80/20) |
| k-Fold CV | $k$ фолдов, каждый по разу в тесте, ошибки усредняются |
| Stratified / Group k-Fold | сохраняет доли классов / не разрывает группы (клиент, пациент) |
| LOO | k-fold при $k=\ell$ |
| Бутстрэп / OOB | выборка с возвращением; ≈36.8% объектов не попадают — на них контроль |
| TimeSeriesSplit | train всегда раньше test; expanding / sliding window; `gap` |
| Nested CV | внутренний цикл подбирает гиперпараметры, внешний оценивает процедуру целиком |
| Learning / validation curve | качество от объёма выборки / от значения гиперпараметра |
| Grid / Random search / Optuna (TPE) | перебор сетки / случайные пробы / байесовская оптимизация + прунинг |

**Три золотых правила:** (1) тест моделирует реальность — группы, время, стратификация;
(2) тест случаен или специально подготовлен (без дубликатов); (3) никаких утечек меток из теста.

$$
P(\text{объект не попал в бутстрэп}) = \Big(1-\frac1m\Big)^m \to e^{-1}\approx 0.368,\qquad
P(\text{random search попал в лучшие } q) = 1-(1-q)^n
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>1. Проблема контроля качества</b></summary>

Ошибка на обучении (**train error**) и ошибка на новых данных (**test error**), как правило, очень различаются:
модель подстраивается под обучающую выборку, и train error **оптимистична** (насколько — формула — см. раздел «1. Почему ошибка на обучении оптимистична»).
Нужен способ оценить, как алгоритм будет работать **в будущем** на реальных данных.

**Базовая идея любого контроля:** разбить выборку на обучающую (TRAIN) и контрольную (TEST),
обучить на первой, оценить на второй.

</details>

<details>
<summary><b>2. Три золотых правила разбиения выборки</b></summary>

1. **Тест/валидация моделирует реальную работу алгоритма:**
   - **деление по группам** — если в проде модель увидит новых клиентов, то и в тесте должны быть клиенты, которых не было в обучении (`GroupKFold`);
   - **предсказание будущего** — если модель прогнозирует будущее, тест должен быть позже обучения (out-of-time);
   - **стратификация** — сохранять пропорции классов (`stratify=y`).

   Схема из лекции: данные можно резать по пользователям (out of sample) и по времени (out of time):
   *in sample / out of time*, *out of sample / in time*, *out of sample / out of time*.
2. **Тест должен быть случайным** (`shuffle=True`) **или специально подготовленным** (например, убрать дубликаты, которые иначе окажутся и в train, и в test).
3. **Нет утечкам:** нельзя явно или неявно использовать метки объектов, на которых оцениваем качество.
   Отбор признаков, mean target encoding (MTE), нормировка, заполнение пропусков — **только по train-части** каждого фолда.

</details>

<details>
<summary><b>3. Способы контроля</b></summary>

**3.1. Отложенная выборка (Hold-out)**

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

**3.2. Random subsampling (Monte-Carlo CV)**

$k$ раз случайно выбираем отложенную выборку и усредняем ошибки.

```python
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit
ShuffleSplit(n_splits=4, test_size=0.3)               # просто случайно
GroupShuffleSplit(n_splits=4, test_size=0.3)          # группы не разбиваются: cv.split(X, groups=g)
StratifiedShuffleSplit(n_splits=4, test_size=0.3)     # сохраняем пропорции классов: cv.split(X, y)
```

**3.3. k-Fold кросс-валидация**

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

**3.4. Бутстрэп**

Выбором **с возвращением** формируем выборку того же объёма $m$, обучаемся на ней; контроль — на
объектах, которые в неё не попали (**out-of-bag**). В контроль попадает примерно
$\big(1 - \frac1m\big)^m \approx e^{-1} \approx 37\%$ выборки (вывод — см. раздел «2. Бутстрэп: откуда 36.8%»).

- модель учится на выборке **того же объёма**, что итоговая;
- используются не все данные, есть дубликаты;
- по распределению бутстрэп-выборка похожа на исходную.

Бутстрэп — основа бэггинга и случайного леса (тема [08](../08_bagging_random_forest/README.md)).

**3.5. Контроль по времени (Out-of-Time)**

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

**3.6. Nested (вложенная) кросс-валидация**

Для **одновременного** подбора гиперпараметров и честной оценки итогового качества:
- **внешний цикл** (outer loop) оценивает обобщающую способность;
- **внутренний цикл** (inner loop) подбирает гиперпараметры/признаки на train-части внешнего фолда.

Зачем: оценка лучшей конфигурации на той же валидации, где её выбирали, **смещена вверх**
(«проклятие победителя», см. раздел «4. Почему нужен отдельный тест: смещение отбора»).

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
inner = GridSearchCV(model, param_grid, cv=KFold(5, shuffle=True, random_state=0))
outer_scores = cross_val_score(inner, X, y, cv=KFold(5, shuffle=True, random_state=1))
```

</details>

<details>
<summary><b>4. Выбор модели (Model Selection)</b></summary>

**Модель** — параметрическое семейство алгоритмов (kNN, SVM, линейная, нейросеть…). **Параметры**
настраиваются обучением; **гиперпараметры** выбираются до обучения — экспертно или перебором.
Моделей много (но конечное число), алгоритмов — бесконечно много: 1NN, 3NN, …, SVM(C=0.27), SVM(C=0.15)…

Выбираем в широком смысле **пайплайн**: модель, гиперпараметры, признаки, предобработку.

**Train / Validation / Test**

| Часть | Зачем |
|---|---|
| **Train** | обучение модели (настройка параметров) |
| **Validation** | выбор пайплайна: модели, гиперпараметров, признаков; иногда — локальный контроль |
| **Test** | итоговая оценка качества выбранного решения |

Можно комбинировать: **k-Fold CV для валидации + hold-out для итогового теста**. Только валидации
доверяют редко — для уверенной оценки нужен тест, к которому не прикасались при выборе.

**Где используется кросс-валидация**

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

**Кривые обучения и валидации**

- **Learning curve** — качество на train и CV в зависимости от **объёма** обучающей выборки:
  большой зазор между train и CV → переобучение (поможет больше данных / регуляризация);
  обе кривые низкие и сошлись → недообучение (нужна сложнее модель / признаки).
- **Validation curve** — качество от **значения гиперпараметра**: видно, где начинается переобучение.

```python
from sklearn.model_selection import learning_curve, validation_curve
sizes, tr, cv = learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8))
tr, cv = validation_curve(model, X, y, param_name="max_depth", param_range=range(1, 20), cv=5)
```

</details>

<details>
<summary><b>5. Подбор гиперпараметров</b></summary>

| Метод | Принцип | Где применим | Плюсы | Минусы |
|---|---|---|---|---|
| **Grid Search** | полный перебор по сетке | маленькие пространства, бейзлайн | простота, воспроизводимость | экспоненциальный рост времени с числом параметров |
| **Randomized Search** | случайные комбинации из распределений | большие пространства, быстрый прототип | быстрее Grid при том же бюджете (почему — см. раздел «5. Почему случайный поиск лучше сетки») | может пропустить оптимум |
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

**Optuna (из семинара по деревьям регрессии)**

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

**Советы**

- Явно указывайте метрику (`scoring`), а лучше несколько.
- Распараллеливание: `n_jobs=-1`; устойчивость к падениям: `error_score=0` (или `np.nan`).
- **Оптимизируйте весь пайплайн** (предобработка + модель), а не только модель — так нет утечек.
- Используйте современные методы (Optuna).
- Фиксируйте `random_state` разбиений, чтобы сравнения моделей были честными (на одних и тех же фолдах).

</details>

<details>
<summary><b>6. Типичные ошибки (утечки)</b></summary>

- Масштабирование / заполнение пропусков / отбор признаков **по всей выборке до** разбиения.
- Target encoding без out-of-fold.
- Случайное перемешивание временного ряда.
- Один клиент (пациент, сессия) и в train, и в test.
- Дубликаты в train и test.
- Многократный подбор гиперпараметров «по тесту» — тест превращается в валидацию.
- Выбор лучшей из сотни конфигураций и отчёт её валидационного качества как итогового.

</details>

**Математика и выводы**

<details>
<summary><b>1. Почему ошибка на обучении оптимистична</b></summary>

Пусть $y = f(x) + \varepsilon$, $\operatorname{Var}\varepsilon = \sigma^2$, и мы обучили линейную регрессию с $d$
параметрами по МНК. Сравним ошибку на обучении $\overline{\text{err}} = \frac{1}{\ell}\sum_i(y_i - \hat y_i)^2$ с
«внутривыборочной» ошибкой $\text{Err}_{\text{in}}$ — ожидаемой ошибкой на **новых** ответах $y_i^0$ в тех же точках $x_i$.

**Утверждение (оптимизм):**

$$
\mathbb{E}\big[\text{Err}_{\text{in}} - \overline{\text{err}}\big] = \frac{2}{\ell}\sum_{i=1}^{\ell}\operatorname{cov}(\hat y_i, y_i) = \frac{2d}{\ell}\sigma^2 .
$$

*Вывод для МНК:* $\hat y = Hy$, $H$ — проектор ранга $d$. Тогда
$\sum_i\operatorname{cov}(\hat y_i, y_i) = \operatorname{tr}\operatorname{cov}(Hy, y) = \operatorname{tr}(H)\sigma^2 = d\sigma^2$
(след проектора равен его рангу). Первое равенство получается раскрытием
$\mathbb{E}(y_i^0 - \hat y_i)^2 - \mathbb{E}(y_i - \hat y_i)^2$: отличие только в члене $2\operatorname{cov}(y_i, \hat y_i)$,
так как новый $y_i^0$ не зависит от $\hat y_i$.

Смысл: чем больше параметров ($d$) и меньше данных ($\ell$), тем сильнее ошибка на обучении занижает
реальную. Отсюда критерии $C_p = \overline{\text{err}} + \frac{2d}{\ell}\hat\sigma^2$, AIC, BIC — аналитические поправки
вместо кросс-валидации. Для сложных моделей (деревья, бустинг) таких формул нет — нужен **контроль на отложенных данных**.

</details>

<details>
<summary><b>2. Бутстрэп: откуда 36.8%</b></summary>

Выбираем $m$ объектов **с возвращением** из $m$. Вероятность, что конкретный объект ни разу не выбран:

$$
\Big(1 - \frac{1}{m}\Big)^m \xrightarrow[m\to\infty]{} e^{-1} \approx 0.368 .
$$

(Логарифм: $m\ln(1 - \frac1m) = m\big(-\frac1m - \frac{1}{2m^2} - \dots\big) \to -1$.) Значит, в бутстрэп-выборку попадает
≈ 63.2% **уникальных** объектов, а ≈ 36.8% остаются для контроля (out-of-bag). Уже при $m = 20$:
$(0.95)^{20} \approx 0.358$.

**Оценка .632.** OOB-ошибка пессимистична (модель видела только 63.2% уникальных объектов), ошибка
на обучении — оптимистична. Эфрон предложил смесь:

$$
\widehat{\text{Err}}^{(.632)} = 0.368\cdot\overline{\text{err}} + 0.632\cdot\widehat{\text{Err}}_{\text{OOB}} .
$$

Для сильно переобучающихся моделей (1NN с $\overline{\text{err}} = 0$) и она оптимистична, поэтому есть
поправка **.632+**, учитывающая степень переобучения.

</details>

<details>
<summary><b>3. Смещение и дисперсия оценок кросс-валидации</b></summary>

- **Смещение.** В $k$-fold модель учится на доле $\frac{k-1}{k}$ данных. Если кривая обучения ещё растёт,
  CV-оценка **пессимистична** (модель на всех данных будет лучше). Чем больше $k$, тем меньше смещение.
- **Дисперсия.** При LOO ($k = \ell$) обучающие выборки почти совпадают, модели сильно коррелированы,
  и среднее их ошибок имеет **большую** дисперсию; при малом $k$ — меньше обучающих данных, но
  разнообразнее модели. Практический компромисс — $k = 5$ или $10$.
- **Разброс по фолдам** $\text{std}$ даёт ощущение неопределённости, но стандартная формула
  $\text{std}/\sqrt{k}$ его **занижает**: фолды не независимы (обучающие части пересекаются).
  Для сравнения моделей лучше повторять CV с разными разбиениями (`RepeatedKFold`) и смотреть на
  **парные** разности метрик на одних и тех же фолдах.

**LOO для линейной регрессии без переобучения**

Для МНК остаток leave-one-out выражается через обычный остаток и диагональ hat-матрицы:

$$
y_i - \hat y_{(-i)} = \frac{y_i - \hat y_i}{1 - h_{ii}}, \qquad h_{ii} = x_i^\top(X^\top X)^{-1}x_i .
$$

*Набросок вывода.* Удаление строки $x_i$: $(X^\top X - x_ix_i^\top)^{-1}$ по формуле Шермана–Моррисона равна
$A^{-1} + \frac{A^{-1}x_ix_i^\top A^{-1}}{1 - h_{ii}}$, где $A = X^\top X$. Подставляя в
$\hat w_{(-i)} = (A - x_ix_i^\top)^{-1}(X^\top y - x_iy_i)$ и упрощая, получаем
$\hat w_{(-i)} = \hat w - \frac{A^{-1}x_i(y_i - \hat y_i)}{1 - h_{ii}}$; отсюда формула. Сумма квадратов этих остатков —
**PRESS**; LOO-ошибка считается за одно обучение. Аналог для Ridge — обобщённая кросс-валидация (GCV),
этим пользуется `RidgeCV` по умолчанию.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

rng = np.random.default_rng(0)
X = np.c_[np.ones(50), rng.normal(size=(50, 3))]; y = X @ [1, 2, -1, 0.5] + rng.normal(size=50)
H = X @ np.linalg.solve(X.T @ X, X.T)
loo_fast = (y - H @ y) / (1 - np.diag(H))
loo_slow = y - cross_val_predict(LinearRegression(fit_intercept=False), X, y, cv=LeaveOneOut())
print(np.allclose(loo_fast, loo_slow))   # True
```

</details>

<details>
<summary><b>4. Почему нужен отдельный тест: смещение отбора</b></summary>

Пусть мы сравниваем $M$ конфигураций, и у всех **одинаковое** истинное качество $\mu$, а оценка на
валидации шумная: $\hat\mu_m = \mu + \xi_m$, $\xi_m \sim \mathcal{N}(0, s^2)$ независимо. Мы выбираем лучшую:

$$
\mathbb{E}\max_m\hat\mu_m = \mu + s\,\mathbb{E}\max_m z_m \le \mu + s\sqrt{2\ln M} ,
$$

где $z_m = \xi_m / s \sim \mathcal{N}(0, 1)$.

При $M = 100$ среднее максимума 100 стандартных нормальных величин ≈ 2.5, и при $s = 0.01$ (1% accuracy)
«лучшая» модель в среднем выглядит на ≈ 2.5 п.п. лучше, чем есть на самом деле.
Это **проклятие победителя** (winner's curse): валидационная оценка выбранной модели **смещена вверх**.
Поэтому:
- итоговое качество оценивают на **тесте**, который не участвовал в выборе;
- либо используют **вложенную (nested) CV**: внутренний цикл выбирает гиперпараметры, внешний —
  оценивает всю процедуру выбора целиком.

Классический пример утечки того же типа: **отбор признаков по всей выборке до CV** (Ambroise &
McLachlan, 2002). На данных из чистого шума с 5000 признаков и 50 объектами отбор 20 «лучших» по
корреляции с таргетом и последующая CV дают «точность» ≈ 90%+, хотя истинная — 50%. Отбор признаков
должен выполняться **внутри** каждого фолда (в пайплайне).

</details>

<details>
<summary><b>5. Почему случайный поиск лучше сетки</b></summary>

**Аргумент Бергстры–Бенджио (2012).** Пусть «хорошая» область гиперпараметров занимает долю $q$ объёма
пространства поиска. При $n$ независимых случайных пробах вероятность хотя бы раз в неё попасть:

$$
1 - (1 - q)^n .
$$

Для $q = 5\%$ и $n = 60$: $1 - 0.95^{60} \approx 0.954$ — **независимо от размерности** пространства.

**Низкая эффективная размерность.** Обычно на качество сильно влияют 1–2 гиперпараметра из многих.
Сетка $k \times k$ по двум параметрам, из которых важен один, проверяет лишь $k$ различных значений важного
параметра; случайный поиск с тем же бюджетом $k^2$ — все $k^2$ разных значений.

</details>

<details>
<summary><b>6. Байесовская оптимизация и TPE (Optuna, Hyperopt)</b></summary>

Идея: строим **суррогатную модель** зависимости качества от гиперпараметров по уже сделанным пробам
и выбираем следующую точку, максимизируя **функцию приобретения** (acquisition), например ожидаемое улучшение:

$$
\text{EI}(x) = \mathbb{E}\big[\max(0,\; y^* - y(x))\big]
$$

(для минимизации ошибки $y$; $y^*$ — порог, например лучшее найденное значение).

**TPE (Tree-structured Parzen Estimator)** моделирует не $p(y \mid x)$, как гауссовский процесс, а $p(x \mid y)$:
делим пробы на «хорошие» ($y < y^*$, обычно лучшие $\gamma \approx 15$–$25\%$) и «плохие»; по каждой группе
строим ядерную оценку плотности $l(x) = p(x \mid y < y^*)$ и $g(x) = p(x \mid y \ge y^*)$. Можно показать, что

$$
\text{EI}(x) \propto \Big(\gamma + \frac{g(x)}{l(x)}(1 - \gamma)\Big)^{-1},
$$

т.е. максимизация EI ⇔ **максимизация отношения $l(x)/g(x)$**: пробуем там, где хороших конфигураций
плотно, а плохих мало. TPE естественно работает с условными («древовидными») пространствами
(например, `kernel` выбран → появляется `gamma`) — отсюда название.

**Прунинг (Optuna).** Неперспективные пробы обрываются досрочно по промежуточным значениям (например,
после части эпох / деревьев): `MedianPruner` останавливает пробу, если её промежуточный результат хуже
медианы предыдущих проб на том же шаге. Экономит большую часть бюджета.

</details>

<details>
<summary><b>7. Перестановочный тест</b></summary>

Проверяем гипотезу $H_0$: «таргет не связан с признаками». Перемешиваем $y$ $n$ раз, каждый раз считаем
CV-качество $s_b$; p-value:

$$
p = \frac{1 + \#\lbrace b: s_b \ge s\rbrace}{1 + n} .
$$

Если $p$ мало, качество $s$ не объясняется случайностью. В sklearn — `permutation_test_score`.

</details>

<details>
<summary><b>8. Временные ряды: почему нельзя перемешивать</b></summary>

При случайном разбиении соседние по времени наблюдения (сильно коррелированные) попадают и в train,
и в test: модель фактически «видит будущее» и интерполирует, а не прогнозирует. Оценка получается
оптимистичной. Правильно — **train всегда раньше test** (expanding / sliding window, `TimeSeriesSplit`).
Если признаки строятся по окну длины $w$ (лаги, скользящие средние), между train и test оставляют
**зазор** (`gap`, embargo) не меньше горизонта признаков/таргета, иначе информация протекает через
перекрывающиеся окна.

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему лучший результат подбора гиперпараметров на валидации — завышенная оценка?</summary>

выбор максимума из шумных оценок («проклятие победителя»); нужен тест или nested CV.

</details>

<details>
<summary>Временной ряд + `KFold(shuffle=True)`</summary>

утечка из будущего; нужен `TimeSeriesSplit` и зазор.

</details>

<details>
<summary>Почему random search лучше grid?</summary>

важны 1–2 параметра; с тем же бюджетом random пробует больше разных значений важных параметров.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- Оптимизм train-ошибки для МНК: $\mathbb{E}[\text{Err}_{\text{in}}-\overline{\text{err}}]=2d\sigma^2/\ell$ (основа $C_p$, AIC).
- LOO для линейной регрессии считается за одно обучение: $e_{(-i)}=e_i/(1-h_{ii})$.
- k-fold с малым $k$ — пессимистичен (модель видит меньше данных); LOO — почти несмещён, но с большой дисперсией. Обычно $k=5$–$10$.
- TPE (Optuna) максимизирует $l(x)/g(x)$ — отношение плотностей «хороших» и «плохих» конфигураций.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (StratifiedKFold, cross_val_score, GridSearchCV,
                                     RandomizedSearchCV, TimeSeriesSplit)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform

X, y = make_classification(n_samples=600, n_features=20, n_informative=5, random_state=0)
pipe = make_pipeline(StandardScaler(), SVC())              # скейлер внутри пайплайна — без утечки
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv)
print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

space = {"svc__C": loguniform(1e-2, 1e2), "svc__gamma": loguniform(1e-3, 1e0)}
search = RandomizedSearchCV(pipe, space, n_iter=20, cv=cv, random_state=0, n_jobs=-1)
# Nested CV: честная оценка всей процедуры «подбор + обучение»
nested = cross_val_score(search, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1))
search.fit(X, y)
print(f"best inner CV = {search.best_score_:.3f} (оптимистично), nested CV = {nested.mean():.3f}")

for tr, te in TimeSeriesSplit(n_splits=3, gap=2).split(np.arange(12)):
    print("TimeSeriesSplit train:", tr, "test:", te)
```

Результат запуска:

```text
CV accuracy: 0.890 ± 0.019
best inner CV = 0.892 (оптимистично), nested CV = 0.875
TimeSeriesSplit train: [0] test: [3 4 5]
TimeSeriesSplit train: [0 1 2 3] test: [6 7 8]
TimeSeriesSplit train: [0 1 2 3 4 5 6] test: [9 10 11]
```

```python
# Утечка: отбор признаков ДО кросс-валидации на чистом шуме даёт «качество» намного выше 50%
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
Xn, yn = rng.normal(size=(60, 5000)), rng.integers(0, 2, 60)          # таргет не связан с признаками
X_leak = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)          # отбор по всей выборке — ошибка!
print("с утечкой:   ", cross_val_score(LogisticRegression(), X_leak, yn, cv=5).mean().round(3))
print("без утечки (≈0.5 — угадывание):", cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression()),
                                       Xn, yn, cv=5).mean().round(3))
```

Результат запуска:

```text
с утечкой:    0.983
без утечки (≈0.5 — угадывание): 0.583
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Кросс-валидация](https://education.yandex.ru/handbook/ml/article/kross-validaciya) · [Подбор гиперпараметров](https://education.yandex.ru/handbook/ml/article/podbor-giperparametrov)
- [sklearn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) · [Tuning hyper-parameters](https://scikit-learn.org/stable/modules/grid_search.html)
- [Optuna docs](https://optuna.readthedocs.io/en/stable/)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Контроль качества и выбор модели.pdf` — весь (36 с.): способы контроля, три золотых правила, кривые, перебор параметров.
- `Лекция 2_072913.pdf`, с. 43–82: кросс-валидация, бутстрэп, временные ряды, nested CV, подбор гиперпараметров (таблица методов).
- Colab-семинар «Деревья регрессии»: подбор гиперпараметров `DecisionTreeRegressor` через Optuna.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Hastie, Tibshirani, Friedman. *ESL*, гл. 7 (оптимизм, $C_p$, AIC, CV, бутстрэп .632)
- Efron B., Tibshirani R. *Improvements on Cross-Validation: The .632+ Bootstrap Method* (1997)
- Ambroise C., McLachlan G. *Selection bias in gene extraction…* (PNAS 2002)
- Cawley G., Talbot N. *On Over-fitting in Model Selection and Subsequent Selection Bias…* (JMLR 2010)
- Bergstra J., Bengio Y. *Random Search for Hyper-Parameter Optimization* (JMLR 2012)
- Bergstra J. et al. *Algorithms for Hyper-Parameter Optimization* (TPE, NeurIPS 2011)

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): Лекция 2 (ML): метрики классификации · Контроль качества и выбор модели

</details>

---

[← 04. Линейные модели: регрессия, регуляризация, логрег, SVM](../04_linear_models/README.md) · [Оглавление курса](../../README.md) · [06. Сложность моделей, bias–variance, переобучение →](../06_bias_variance/README.md)
