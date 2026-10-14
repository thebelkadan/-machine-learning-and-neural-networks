# 09. Градиентный бустинг

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Строим сумму деревьев $a_T=a_0+\eta\sum_t b_t$, где каждое новое дерево приближает
**антиградиент** функции потерь по текущим ответам (для MSE — просто невязку $y-a_{T-1}(x)$).
Это градиентный спуск в пространстве функций. Бустинг снижает **смещение**, переобучается с ростом числа
деревьев → нужен **early stopping**. Лучшие реализации: XGBoost, LightGBM, CatBoost.

| Термин | Определение |
|---|---|
| Псевдоостатки | $r_i=-\partial L(y_i,z)/\partial z$ в точке $z=a_{T-1}(x_i)$ |
| Shrinkage / learning rate | множитель $\eta$ при каждом дереве; меньше $\eta$ → больше деревьев |
| Stochastic GB | каждое дерево на случайной доле объектов (`subsample`) |
| XGBoost-формулы | $w_j^*=-G_j/(H_j+\lambda)$, Gain из $G^2/(H+\lambda)$ |
| Histogram / GOSS / EFB | бины вместо сортировки / сэмплинг по градиентам / склейка разреженных признаков (LightGBM) |
| Leaf-wise / oblivious | рост по листьям (LightGBM) / симметричные деревья (CatBoost) |
| Ordered boosting / TS | CatBoost: статистики и градиенты только по «прошлым» объектам перестановки — без утечки таргета |

$$
r_i=-\frac{\partial L(y_i,z)}{\partial z}\Big|_{a_{T-1}(x_i)},\qquad
w_j^*=-\frac{\sum_{i\in R_j}g_i}{\sum_{i\in R_j}h_i+\lambda},\qquad
\text{Gain}=\tfrac12\Big[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}-\tfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>1. Идея</b></summary>

**Бустинг** — последовательный ансамбль: каждый следующий базовый алгоритм строится так, чтобы **исправлять
ошибки** уже построенной композиции. «Шагаем последовательно к таргету».

$$
a_T(x) = a_0(x) + \eta\sum_{t=1}^{T} b_t(x) .
$$

Если бэггинг/лес усредняет **сложные** модели и снижает дисперсию, то бустинг складывает **простые**
(неглубокие деревья, high bias) и снижает в первую очередь **смещение**.

</details>

<details>
<summary><b>2. Алгоритм градиентного бустинга</b></summary>

Есть дифференцируемая функция ошибки $L(y, z)$. Уже построен $a_{T-1}$. Хотим добавить $b_T$, минимизируя
$\sum_i L(y_i, a_{T-1}(x_i) + b(x_i))$ — в общем случае аналитически не решается.

**Решение:** функция ошибки убывает в направлении **антиградиента** по ответам. Считаем

$$
r_i = -\frac{\partial L(y_i, z)}{\partial z}\Big|_{z = a_{T-1}(x_i)}
$$

и обучаем $b_T$ **приближать** $r_i$ (обычно по MSE) на выборке $\lbrace (x_i, r_i)\rbrace$. Отсюда название.

```
a_0 = оптимальная константа (среднее / медиана / логит доли класса 1)
для t = 1..T:
    r_i = -∂L(y_i, z)/∂z  при z = a_{t-1}(x_i)       # псевдоостатки (антиградиент)
    b_t = дерево, обученное на {(x_i, r_i)}            # структура по MSE
    значения в листьях уточнить по исходной L         # TreeBoost / шаг Ньютона
    a_t = a_{t-1} + η · b_t
```

| Задача | Функция ошибки | Антиградиент — на что учится следующее дерево |
|---|---|---|
| Регрессия | MSE $\frac12(y - z)^2$ | **невязка** $y - z$ — «настраиваемся на невязку» |
| Регрессия | MAE $\lvert y - z\rvert$ | $\operatorname{sign}(y - z)$ — устойчиво к выбросам |
| Классификация | log loss (BinomialBoost), $p = \sigma(z)$ | $y - p$ |
| Классификация | экспоненциальная $e^{-yz}$ (AdaBoost) | $y e^{-yz}$ |

Для классификации модель выдаёт вещественные «сырые скоры» $z$ (сумма деревьев), вероятность — $\sigma(z)$.
Используются **суррогатные** гладкие функции ошибки.

Подробный вывод и связь с AdaBoost (см. раздел «2. Идея: градиентный спуск в пространстве ответов»)

</details>

<details>
<summary><b>3. Регуляризация бустинга</b></summary>

| Приём | Смысл | Параметры |
|---|---|---|
| **Shrinkage** (темп обучения) | $a_t = a_{t-1} + \eta b_t$; шаг по антиградиенту всё равно не ведёт сразу в минимум, а шагаем мы лишь по приближению алгоритма | `learning_rate` / `eta` |
| **Число деревьев** | бустинг **переобучается** с ростом $T$ → контролируем ранним остановом | `n_estimators`, `early_stopping` |
| **Stochastic GB** (Friedman, 1999) | каждое дерево на случайной доле объектов (bag fraction, обычно без бутстрэпа): быстрее и часто лучше | `subsample`, `bagging_fraction` |
| **Подвыборка признаков** | на дерево / на уровень / на вершину | `colsample_bytree`, `colsample_bynode`, `rsm` (CatBoost) |
| **Сложность деревьев** | глубина / число листьев / минимум объектов в листе | `max_depth`, `num_leaves`, `min_data_in_leaf` |
| **L1/L2 на веса листьев** | $w_j = -\frac{G_j}{H_j + \lambda}$, с L1 — мягкий порог | `reg_alpha`, `reg_lambda` |
| **Минимальный gain** | штраф $\gamma$ за каждый лист | `gamma`, `min_split_gain` |

Чем меньше `learning_rate`, тем больше нужно деревьев (они связаны).

</details>

<details>
<summary><b>4. Продвинутые методы: второй порядок (XGBoost)</b></summary>

Раскладываем функцию ошибки в ряд Тейлора до второго порядка ($g_i$, $h_i$ — первая и вторая производные
по ответу) и добавляем регуляризацию дерева $\gamma J + \frac\lambda2\sum w_j^2$. Для дерева с фиксированной структурой:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad
\text{Gain} = \frac12\Big[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Big] - \gamma ,
$$

$G_j = \sum_{i \in R_j} g_i$, $H_j = \sum_{i \in R_j} h_i$. **Критерий расщепления берётся из самой функции ошибки**,
а не из Джини/энтропии; можно строить дерево и рекурсивно обрезать ветви с Gain < 0
(вывод (см. раздел «4. XGBoost: вторые производные и регуляризация»)).

</details>

<details>
<summary><b>5. Современные реализации</b></summary>

| | XGBoost (2014, Tianqi Chen, Higgs Boson) | LightGBM (2017, Microsoft) | CatBoost (2017, Яндекс) |
|---|---|---|---|
| **Рост деревьев** | по уровням (level-wise), потом добавили по листьям (для гистограмм) | **по листьям** (leaf-wise, best-first) | по уровням однородно — **симметричные (oblivious) деревья** |
| **Поиск сплитов** | exact greedy (полный перебор) + гистограммы (`tree_method='hist'`) | **гистограммы** (бины) | предварительный биннинг |
| **Фишки** | sparsity-aware split finding; random forest mode | **EFB** (склейка разреженных признаков), **GOSS** | **Ordered boosting**, overfitting detector, работает «из коробки» |
| **Сэмплирование** | — | GOSS: среди малых градиентов сэмплируем, но с большим весом | Бернулли, байесовская, MVS |
| **Важности признаков** | gain, frequency (weight), coverage | gain, split | PredictionValuesChange, LossFunctionChange, Interaction, SHAP |
| **Пропуски (NaN)** | на оптимальную сторону сплита | на оптимальную сторону (`use_missing=True`); нули можно считать NaN | Min / Max |
| **Категориальные признаки** | с версии 1.5 | разбиение на две части с учётом mean target (категории сортируются по `(sum_g + cat_smooth) / (sum_h + cat_l2)`) | **Ordered Target Statistics**, OHE при числе категорий < `one_hot_max_size`, жадные **комбинации** категорий |

В sklearn: `GradientBoostingRegressor/Classifier` (классика, медленно) и `HistGradientBoosting*`
(гистограммный, быстрый, нативные NaN и категории).

**Ключевые идеи реализаций**

- **Гистограммный подход**: каждый вещественный признак заранее разбивается на бины (≤ 255) → число
  проверяемых порогов ~ число бинов, а не число уникальных значений.
- **EFB** (Exclusive Feature Bundling): признаки с большим числом нулей, которые одновременно не ненулевые,
  жадно склеиваются в бандлы (оптимум — NP-полная задача).
- **GOSS**: объекты с большими градиентами берём все, из остальных сэмплируем, но с повышенным весом.
- **Игнорирование нулей/NaN**: убираем их, выбираем сплит, добавляем в «выгодное» поддерево.
- **CatBoost и проблема смещения**: значение в листе — сумма антиградиентов, и мы оцениваем значение
  в точке, **зная метку в ней** → утечка. Решение — **ordered boosting** (динамический бустинг): случайная
  перестановка объектов, для каждого объекта используем только информацию «до него»; то же для
  **smoothed target encoding** категорий (подробно (см. раздел «8. CatBoost: сдвиг предсказаний и ordered boosting»)).
- **DART**: при построении нового дерева «выключаем» случайное подмножество старых (dropout для деревьев).
- **GBDT-PL** (`linear_tree=True` в LightGBM): линейные регрессии в листьях вместо констант — бустинг
  получает способность к экстраполяции.

</details>

<details>
<summary><b>6. Как настраивать (итог лекции + семинары)</b></summary>

1. Выбрать вид бустинга, функцию ошибки и метрику **под задачу**.
2. **Три самых важных параметра: сложность деревьев, темп обучения, число деревьев.**
3. Для нескольких значений сложности (глубина / число листьев) настроить два остальных связанных
   параметра; для перебора можно использовать **немного деревьев** (большой `learning_rate`).
4. В продакшене — **увеличить число деревьев и уменьшить темп** (с ранним остановом).
5. Попробовать **сумму нескольких GBM** (разные seed / параметры) — проверить, помогает ли.

Бустинг **неплохо работает на параметрах «из коробки»** и не очень чувствителен к гиперпараметрам —
перебирать их стоит в последнюю очередь. Категории гиперпараметров LightGBM (из семинара):

| Категория | Параметры |
|---|---|
| Настраиваем всегда | `learning_rate` (`eta`) — самый важный, зависимость унимодальная; `max_cat_to_onehot` |
| По желанию | `num_leaves` (`max_leaves`), `zero_as_missing`, `reg_alpha`, `reg_lambda` |
| Для ускорения | `data_sample_strategy` (`bagging` / `goss`), `colsample_by*` |
| Для «надёжности» | `min_data_in_leaf`, `min_data_in_bin`, `min_data_per_group` |
| Не трогать без нужды | `min_split_gain=0`, `max_depth` (для leaf-wise деревьев ограничение глубины искажает идею), `max_bin=255` |

**Обязательно всегда:** early stopping и логирование метрик; кастомная метрика, если целевая не реализована;
интерпретация модели после обучения (важности, SHAP) для проверки адекватности.

</details>

<details>
<summary><b>7. Код</b></summary>

**sklearn**

```python
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8)
hgb = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05, max_leaf_nodes=31,
                                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
                                     categorical_features="from_dtype")       # нативные категории и NaN
```

**LightGBM (Training API, семинар 1)**

```python
import lightgbm as lgb

lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_index)    # или dtype 'category' в pandas
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_tr)
params = {
    "objective": "binary",              # лосс = logloss
    "metric": ["auc", "binary_logloss"],
    "eta": 0.05,                        # learning rate
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,            # colsample_bytree
    "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}
evals = {}
model = lgb.train(params, lgb_tr, num_boost_round=5000, valid_sets=[lgb_val],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100), lgb.record_evaluation(evals)])
proba = model.predict(X_val, num_iteration=model.best_iteration)          # вероятности
raw = model.predict(X_val, raw_score=True)                                # «сырые» скоры (удобнее для анализа)
model.feature_importance(importance_type="gain")                         # 'split' | 'gain'
model.trees_to_dataframe()                                                # все деревья как таблица

# Кастомная метрика: (preds, data) -> (name, value, is_higher_better)
def f1_metric(preds, data):
    y_true = data.get_label()
    y_pred = (preds > 0.5).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum(); fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return "f1", 2 * tp / (2 * tp + fp + fn + 1e-12), True

# Кастомный лосс: возвращает градиент и гессиан по сырому скору
from scipy.special import expit
def logloss_obj(preds, data):
    p = expit(preds); y = data.get_label()
    return p - y, p * (1 - p)

# Веса объектов (сумма весов ≈ N!), дисбаланс классов
lgb.Dataset(X_tr, y_tr, weight=w)          # или params: is_unbalance=True / scale_pos_weight=...
lgb.cv(params, lgb_tr, num_boost_round=500, nfold=5)                       # встроенная CV
# Дообучение «поверх» другой модели: init_score = сырые скоры предыдущей модели
```

**CatBoost (семинар 2)**

```python
import catboost as cb

train_pool = cb.Pool(X_tr, y_tr, cat_features=cat_cols)        # категории указывать явно
val_pool = cb.Pool(X_val, y_val, cat_features=cat_cols)
model = cb.CatBoostClassifier(
    loss_function="Logloss", eval_metric="AUC",
    iterations=5000, learning_rate=0.05, depth=6,
    od_type="Iter", early_stopping_rounds=200, use_best_model=True,   # overfitting detector
    auto_class_weights="Balanced",           # или class_weights / scale_pos_weight
    custom_metric=["Precision:hints=skip_train~true", "F1:hints=skip_train~true"],
    one_hot_max_size=4,                       # OHE для признаков с малым числом категорий
    boosting_type="Plain",                    # 'Ordered' — для малых данных (< 10k)
    verbose=200)
model.fit(train_pool, eval_set=val_pool)
model.get_feature_importance(type="PredictionValuesChange")   # 'LossFunctionChange', 'Interaction', 'ShapValues'
model.get_evals_result()
# has_time=True + timestamp — использовать порядок по времени вместо случайных перестановок
```

**Бустинг с нуля (регрессия, MSE)**

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class MyGradientBoostingRegressor:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3):
        self.T, self.lr, self.depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y):
        self.f0 = y.mean()                                 # начальное приближение: оптимальная константа MSE
        pred = np.full(len(y), self.f0)
        self.trees = []
        for _ in range(self.T):
            residual = y - pred                            # антиградиент MSE = невязка
            tree = DecisionTreeRegressor(max_depth=self.depth).fit(X, residual)
            pred += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        return self.f0 + self.lr * sum(t.predict(X) for t in self.trees)
```

</details>

<details>
<summary><b>8. Бустинг vs нейросети на таблицах</b></summary>

ДЗ 1 курса DL (`task_for_students.ipynb`, Covertype: 581 тыс. участков леса, 7 классов): CatBoost «по умолчанию»
(1000 деревьев глубины 6) даёт **0.890** точности на тесте; MLP нужно много настраивать, чтобы его догнать.
На таблицах бустинг обычно сильнее: ему всё равно, в каких шкалах признаки, он устойчив к неинформативным
признакам и почти не требует настройки. Подробнее про MLP — темы [14](../14_neural_networks_backprop/README.md)–[15](../15_nn_training/README.md).

</details>

<details>
<summary><b>Типичные ошибки</b></summary>

- Обучать без раннего останова (бустинг переобучается с числом деревьев, в отличие от леса).
- Сравнивать модели, подобранные с разным числом деревьев/темпом, без контроля на валидации.
- Подавать категории в LightGBM/CatBoost как числа (Label Encoding), не указав, что это категории.
- Делать target encoding вручную без out-of-fold и отдавать его бустингу (утечка) — CatBoost решает это сам.
- Задавать веса объектов с суммой, сильно отличающейся от $N$, или нулевые веса.
- Интерпретировать `predict_proba` как откалиброванные вероятности при сильных весах классов (`is_unbalance`).

</details>

**Математика и выводы**

<details>
<summary><b>1. Forward Stagewise Additive Modeling</b></summary>

Строим модель как сумму:

$$
a_T(x) = \sum_{t=0}^{T} \gamma_t b_t(x) .
$$

**FSAM**: добавляем слагаемые по одному, не меняя уже построенные. Пусть построено $a_{T-1}$. Формально нужно решить

$$
(\gamma_T, b_T) = \arg\min_{\gamma, b}\sum_{i=1}^{\ell} L\big(y_i,\; a_{T-1}(x_i) + \gamma b(x_i)\big),
$$

**а не** просто «обучить $b$ на невязку» $y_i - a_{T-1}(x_i)$ (это совпадает лишь для MSE). Проблема: для
произвольной $L$ эта задача аналитически не решается.

</details>

<details>
<summary><b>2. Идея: градиентный спуск в пространстве ответов</b></summary>

Рассмотрим ошибку как функцию от **вектора ответов** на обучающих объектах $\hat y = (\hat y_1, \dots, \hat y_\ell)$:
$F(\hat y) = \sum_i L(y_i, \hat y_i)$. Сейчас $\hat y_i = a_{T-1}(x_i)$. Функция убывает быстрее всего в
направлении антиградиента:

$$
g_i = \frac{\partial L(y_i, z)}{\partial z}\Big|_{z = a_{T-1}(x_i)}, \qquad \hat y \leftarrow \hat y - \eta\, g .
$$

Но мы не можем сдвинуть ответы на обучающих объектах произвольно — нужен **алгоритм**, который умеет
предсказывать и на новых объектах. Поэтому **приближаем антиградиент** базовым алгоритмом (обычно MSE):

$$
b_T = \arg\min_b \sum_{i=1}^{\ell}\big(b(x_i) - (-g_i)\big)^2, \qquad a_T = a_{T-1} + \eta\, b_T .
$$

Отсюда название: **градиентный бустинг = градиентный спуск в функциональном пространстве**, где шаг —
это обученный на антиградиент базовый алгоритм.

**Частные случаи антиградиента** ($z$ — текущий ответ):

| Задача | $L(y, z)$ | $-\partial L/\partial z$ |
|---|---|---|
| Регрессия, MSE | $\frac12(y - z)^2$ | $y - z$ — **невязка** (residual) |
| Регрессия, MAE | $\lvert y - z\rvert$ | $\operatorname{sign}(y - z)$ |
| Классификация, log loss (BinomialBoost), $y \in \lbrace 0,1\rbrace$, $p = \sigma(z)$ | $-y\ln p - (1-y)\ln(1-p)$ | $y - \sigma(z)$ |
| То же для $y \in \lbrace -1, +1\rbrace$ | $\ln(1 + e^{-yz})$ | $\dfrac{y}{1 + e^{yz}}$ |
| AdaBoost-подобная, $y \in \lbrace -1,+1\rbrace$ | $e^{-yz}$ | $y\,e^{-yz}$ |

Для MSE «настраиваемся на невязку» буквально. Для классификации нужны **суррогатные** дифференцируемые функции
ошибки — гладкие верхние оценки для $[yz < 0]$ (тема [04](../04_linear_models/README.md)).

**Начальное приближение** $a_0$ — оптимальная константа: среднее для MSE, медиана для MAE,
$\ln\frac{\bar y}{1-\bar y}$ (логит доли класса 1) для log loss — это и делает `boost_from_average`.

</details>

<details>
<summary><b>3. TreeBoost: оптимальные значения в листьях</b></summary>

Базовый алгоритм — дерево $b(x) = \sum_{j=1}^{J} c_j[x \in R_j]$. Структуру дерева (области $R_j$) строим по
антиградиенту (MSE-критерий), а **значения в листьях** подбираем уже по исходной функции потерь — каждое
отдельно, так как области не пересекаются (Friedman, 2001):

$$
c_j = \arg\min_c \sum_{x_i \in R_j} L\big(y_i,\; a_{T-1}(x_i) + c\big) .
$$

- MSE → среднее невязок в листе; MAE → медиана невязок в листе.
- Log loss — решения в явном виде нет; делаем **один шаг Ньютона** от $c = 0$:

$$
c_j \approx -\frac{\sum_{i \in R_j} g_i}{\sum_{i \in R_j} h_i} = \frac{\sum_{i \in R_j}(y_i - p_i)}{\sum_{i \in R_j} p_i(1 - p_i)}, \qquad h_i = \frac{\partial^2 L}{\partial z^2} = p_i(1-p_i) .
$$

</details>

<details>
<summary><b>4. XGBoost: вторые производные и регуляризация</b></summary>

Оптимизируем на шаге $t$ функционал с регуляризацией сложности дерева ($J$ листьев, веса $w_j$):

$$
\mathcal{L}^{(t)} = \sum_{i=1}^{\ell} L\big(y_i,\; \hat y_i^{(t-1)} + b(x_i)\big) + \gamma J + \frac{\lambda}{2}\sum_{j=1}^{J} w_j^2 .
$$

**Разложение Тейлора до второго порядка** по добавке $b(x_i)$ (малой — потому и нужен маленький шаг):

$$
\mathcal{L}^{(t)} \approx \sum_i\Big[L(y_i, \hat y_i^{(t-1)}) + g_i b(x_i) + \tfrac12 h_i b(x_i)^2\Big] + \gamma J + \frac{\lambda}{2}\sum_j w_j^2 ,
$$

где $g_i = \partial_z L$, $h_i = \partial_z^2 L$ в точке $\hat y_i^{(t-1)}$. Первое слагаемое — константа. Сгруппируем объекты по листьям:
$b(x_i) = w_j$ при $x_i \in R_j$; обозначим $G_j = \sum_{i \in R_j} g_i$, $H_j = \sum_{i \in R_j} h_i$:

$$
\tilde{\mathcal{L}} = \sum_{j=1}^{J}\Big[G_j w_j + \tfrac12(H_j + \lambda)w_j^2\Big] + \gamma J .
$$

**Оптимальные веса листьев** (производная по $w_j$ равна нулю):

$$
w_j^* = -\frac{G_j}{H_j + \lambda} .
$$

**Значение функционала** при оптимальных весах — оценка качества структуры дерева:

$$
\tilde{\mathcal{L}}^* = -\frac12\sum_{j=1}^{J}\frac{G_j^2}{H_j + \lambda} + \gamma J .
$$

**Gain сплита** листа на левую и правую части — уменьшение $\tilde{\mathcal{L}}^*$:

$$
\text{Gain} = \frac12\Big[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Big] - \gamma .
$$

Следствия (как на слайдах «Продвинутые методы оптимизации»):
- критерий расщепления берётся **не из традиционных impurity, а из самой функции ошибки**;
- $\gamma$ — минимальный выигрыш для сплита (`min_split_gain`); можно строить дерево и рекурсивно
  **обрезать**, если Gain < 0;
- $\lambda$ сжимает веса листьев (L2), особенно в листьях с малым $H_j$ (мало объектов / уверенные предсказания);
- для MSE $h_i = 1$, $H_j = n_j$ — формулы превращаются в «среднее невязок со сглаживанием» $\frac{\sum r_i}{n_j + \lambda}$;
- **L1 на веса** ($\alpha\sum|w_j|$) даёт мягкий порог: $w_j^* = -\operatorname{sign}(G_j)\frac{(|G_j| - \alpha)_+}{H_j + \lambda}$ — ровно псевдокод
  из семинара по LightGBM (`reg_alpha`, `reg_lambda`);
- `min_child_weight` (`min_sum_hessian_in_leaf`) ограничивает снизу $H_j$ — для log loss это «эффективное число
  неуверенных объектов» в листе.

**Пользовательский лосс** требует $g$ и $h$. Качество обучения зависит от того, насколько хорошо лосс
приближается своим разложением Тейлора второго порядка: у MSE приближение **точное** (функция
квадратичная); у log loss — хорошее при малых шагах; у MAE $h = 0$ почти всюду, и шаг Ньютона не определён,
поэтому для MAE/квантильных лоссов библиотеки пересчитывают значения в листьях отдельно (медиана /
квантиль остатков). Если кастомный лосс «расходится», причина обычно в плохом квадратичном приближении
(семинар по LightGBM) — помогают меньший `learning_rate` и ограничение снизу на $h$.

</details>

<details>
<summary><b>5. Shrinkage и стохастический бустинг как регуляризация</b></summary>

**Темп обучения** $\eta$: $a_T = a_{T-1} + \eta b_T$, $0 < \eta \le 1$. Почему помогает:
- шаг в направлении антиградиента не приводит сразу в минимум, а мы ещё и шагаем лишь «по ответам
  алгоритма», который приближает антиградиент с ошибкой — незачем шагать далеко;
- маленький $\eta$ = много маленьких шагов = эффект, похожий на L1-регуляризацию пути (связь с
  forward stagewise / LARS, Hastie et al.);
- чем меньше $\eta$, тем больше нужно деревьев; число деревьев **надо контролировать** (early stopping) —
  в отличие от леса, бустинг **переобучается** с ростом $T$ (он уменьшает смещение, подгоняясь всё точнее).

**Stochastic gradient boosting** (Friedman, 1999): каждое дерево учится на случайной доле объектов
(`subsample`, bag fraction, обычно без возвращения) — аналог мини-батчей: быстрее и часто лучше
(«регуляризация»). Аналогично — **подвыборка признаков** (`colsample_bytree`, `colsample_bynode`).

</details>

<details>
<summary><b>6. AdaBoost как частный случай</b></summary>

Экспоненциальная потеря $L(y, z) = e^{-yz}$, $y \in \lbrace -1, +1\rbrace$, базовые классификаторы $b(x) \in \lbrace -1, +1\rbrace$.
FSAM-шаг:

$$
\sum_i e^{-y_i(a_{T-1}(x_i) + \gamma b(x_i))} = \sum_i w_i e^{-\gamma y_i b(x_i)}, \qquad w_i = e^{-y_i a_{T-1}(x_i)} .
$$

Разделим на правильно ($y_i b(x_i) = 1$) и неправильно классифицированные:

$$
= e^{-\gamma}\sum_{\text{верно}} w_i + e^{\gamma}\sum_{\text{ошибка}} w_i
= (e^{\gamma} - e^{-\gamma})\sum_i w_i[y_i \ne b(x_i)] + e^{-\gamma}\sum_i w_i .
$$

- При фиксированном $\gamma > 0$ оптимальный $b$ минимизирует **взвешенную ошибку** $\text{err} = \frac{\sum w_i[y_i \ne b(x_i)]}{\sum w_i}$.
- Оптимальный вес: производная по $\gamma$ равна нулю ⇒ $\gamma^* = \frac12\ln\frac{1 - \text{err}}{\text{err}}$.
- Новые веса $w_i \leftarrow w_i e^{-\gamma y_i b(x_i)}$: **ошибочные объекты получают больший вес**.

Это ровно алгоритм AdaBoost (Freund & Schapire, 1997), выведенный Фридманом, Хасти и Тибширани (2000)
как FSAM с экспоненциальной потерей. Экспоненциальная потеря сильно штрафует большие отрицательные
отступы → AdaBoost чувствителен к выбросам и шуму в разметке; log loss растёт лишь линейно.

</details>

<details>
<summary><b>7. Ускорения: гистограммы, GOSS, EFB, leaf-wise</b></summary>

**Гистограммы.** Каждый признак заранее разбивается на $\le 255$ бинов (квантизация). В вершине для каждого
бина копим $\sum g$ и $\sum h$ за $O(n)$, перебор порогов — $O(\#\text{бинов})$ вместо $O(n\log n)$. Трюк
**вычитания гистограмм**: гистограмма одного потомка = гистограмма родителя − гистограмма другого потомка,
поэтому считаем явно только для меньшего потомка.

**GOSS** (Gradient-based One-Side Sampling, LightGBM). Объекты с большим $|g_i|$ «плохо выучены» и
информативнее. Берём **все** объекты из top-$a$ доли по $|g|$ и случайную $b$-долю из остальных, а у
сэмплированных умножаем градиенты на $\frac{1 - a}{b}$ — так оценка суммы градиентов по «малым» объектам
остаётся несмещённой. Пример со слайда: «выбираем top-2, из остальных сэмплируем, но берём с весом».

**EFB** (Exclusive Feature Bundling). Разреженные признаки, которые почти никогда не бывают ненулевыми
одновременно, склеиваются в один «бандл» (значения сдвигаются в разные диапазоны бинов). Оптимальное
разбиение на бандлы — задача раскраски графа (NP-трудна) → жадный алгоритм: граф признаков с весами рёбер
= числом конфликтов, сортировка по степени, добавление в бандл при малом числе конфликтов.

**Leaf-wise (best-first)** рост (LightGBM): расщепляем лист с максимальным Gain во всём дереве, а не все
листья уровня. При том же числе листьев даёт меньший loss, но может строить глубокие несбалансированные
деревья — ограничивают `num_leaves` (а не `max_depth`).

**Sparsity-aware split finding** (XGBoost): при поиске сплита пропуски (и нули) временно исключаются, затем
пробуются оба направления «по умолчанию» и выбирается лучшее.

</details>

<details>
<summary><b>8. CatBoost: сдвиг предсказаний и ordered boosting</b></summary>

**Проблема утечки таргета (prediction shift).** В классическом бустинге градиенты $g_i$ для объекта $x_i$
вычисляются моделью $a_{T-1}$, которая **сама обучалась на $y_i$**. Значения в листьях — суммы
антиградиентов, в том числе самого объекта. Поэтому распределение $a_{T-1}(x_i)$ на обучении отличается
от распределения на тесте → смещение. Особенно остро — с **target encoding** категориальных признаков:
среднее таргета по категории, посчитанное с участием самого объекта, — прямая утечка (на редких категориях
признак почти равен таргету).

**Ordered Target Statistics.** Берём случайную перестановку $\sigma$ объектов. Для объекта $i$ категориальный
признак кодируем только по объектам, **стоящим раньше** в перестановке:

$$
\text{TS}_i = \frac{\sum_{j:\,\sigma(j) < \sigma(i),\; x_j^k = x_i^k} y_j + a\cdot P}{\sum_{j:\,\sigma(j) < \sigma(i),\; x_j^k = x_i^k} 1 + a},
$$

$P$ — априорное значение (prior, например средний таргет), $a > 0$ — его вес (сглаживание). Для первых
объектов перестановки оценка опирается почти только на prior — шумно, поэтому используют несколько
перестановок. Аналогия — online-обучение: значение для объекта считается только по «прошлому».

**Ordered boosting.** Та же идея для градиентов: для объекта $i$ остаток считается моделью, обученной
только на объектах раньше него в перестановке. Наивно нужно $\ell$ моделей; CatBoost хранит
приближения для логарифмического набора префиксов. Медленно на CPU, полезно на малых данных
(рекомендация семинара: < 10 тыс. объектов).

**Oblivious (симметричные) деревья.** На каждом уровне — **один и тот же** предикат для всех вершин уровня.
Дерево глубины $d$ — это $d$ бинарных признаков, а номер листа — $d$-битное число: прогноз считается
битовыми операциями без ветвлений (очень быстрый инференс), а сильное ограничение структуры работает как
регуляризация.

**Комбинации категорий.** Жадно добавляются признаки-пересечения категориальных признаков, уже
использованных в дереве (например, «город × категория товара»), со своими TS.

</details>

<details>
<summary><b>9. DART</b></summary>

**Dropouts meet Multiple Additive Regression Trees** (Rashmi & Gilad-Bachrach, 2015): при построении
нового дерева случайно «выключаем» подмножество уже построенных деревьев $D$ и учим новое дерево на
антиградиент ансамбля **без** них. Затем нормируем: новое дерево и выключенные деревья масштабируются
(например, новое на $\frac{1}{|D| + 1}$, выключенные — на $\frac{|D|}{|D| + 1}$), чтобы не «перескочить» цель. Борется с
тем, что первые деревья делают основную работу, а поздние лишь мелко поправляют («over-specialization»).

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему «градиентный»?</summary>

каждое дерево — шаг по антиградиенту функции потерь в пространстве ответов.

</details>

<details>
<summary>На что учится дерево в бинарной классификации с log loss?</summary>

на $y-\sigma(z)$; веса листьев — шаг Ньютона $\sum(y-p)/\sum p(1-p)$.

</details>

<details>
<summary>Зачем CatBoost'у ordered TS?</summary>

обычный target encoding «видит» таргет объекта → утечка, сдвиг предсказаний.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- XGBoost: Тейлор 2-го порядка + регуляризация $\gamma J+\frac\lambda2\sum w_j^2$ → критерий сплита из самой функции потерь.
- AdaBoost = FSAM с экспоненциальной потерей: $\gamma=\frac12\ln\frac{1-\text{err}}{\text{err}}$, веса ошибочных объектов растут.
- GOSS: все объекты с большими $|g|$ + выборка малых с весом $\frac{1-a}{b}$ (несмещённая оценка суммы градиентов).
- Oblivious-деревья: номер листа — $d$-битное число, быстрый инференс и регуляризация.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

X, y = make_friedman1(n_samples=2000, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

# Градиентный бустинг с нуля (MSE): каждое дерево учится на невязку
f0, lr, trees = y_tr.mean(), 0.1, []
pred_tr = np.full(len(y_tr), f0)
for _ in range(200):
    tree = DecisionTreeRegressor(max_depth=3).fit(X_tr, y_tr - pred_tr)
    pred_tr += lr * tree.predict(X_tr)
    trees.append(tree)
pred_te = f0 + lr * sum(t.predict(X_te) for t in trees)
skl = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, y_tr)
print(f"MSE с нуля: {mean_squared_error(y_te, pred_te):.3f}   sklearn: {mean_squared_error(y_te, skl.predict(X_te)):.3f}")
```

Результат запуска:

```text
MSE с нуля: 1.800   sklearn: 1.801
```

```python
# Переобучение по числу деревьев и early stopping (HistGradientBoosting ~ LightGBM внутри sklearn)
from sklearn.ensemble import HistGradientBoostingRegressor
test_curve = [mean_squared_error(y_te, p) for p in
              GradientBoostingRegressor(n_estimators=1500, learning_rate=0.3, max_depth=4, random_state=0)
              .fit(X_tr, y_tr).staged_predict(X_te)]
print(f"lr=0.3: лучший тест на {int(np.argmin(test_curve)) + 1} деревьях (MSE={min(test_curve):.3f}), "
      f"на 1500 — MSE={test_curve[-1]:.3f}")
hgb = HistGradientBoostingRegressor(max_iter=5000, learning_rate=0.05, early_stopping=True,
                                    validation_fraction=0.15, n_iter_no_change=50, random_state=0).fit(X_tr, y_tr)
print(f"HistGB + early stopping: остановился на {hgb.n_iter_} итерациях, test MSE={mean_squared_error(y_te, hgb.predict(X_te)):.3f}")
```

Результат запуска:

```text
lr=0.3: лучший тест на 77 деревьях (MSE=1.928), на 1500 — MSE=2.036
HistGB + early stopping: остановился на 222 итерациях, test MSE=1.778
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Градиентный бустинг](https://education.yandex.ru/handbook/ml/article/gradientnyj-busting)
- Документация: [XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/index.html) (туториалы, model.html — вывод формул) · [LightGBM](https://lightgbm.readthedocs.io/en/latest/) (Features, Parameters Tuning) · [CatBoost](https://catboost.ai/docs/en/) (Algorithm Details, Training Parameters)
- [LightGBM and XGBoost explained (mlexplained.com)](https://mlexplained.com/2018/01/05/lightgbm-and-xgboost-explained/)
- Friedman, [Stochastic Gradient Boosting](http://statweb.stanford.edu/~jhf/ftp/stobst.pdf)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Градиентный бустинг.pdf` — лекция (39 слайдов): FSAM, алгоритм, shrinkage, stochastic GB, TreeBoost, второй порядок, сравнение XGBoost/LightGBM/CatBoost, гистограммы, EFB, GOSS, ordered boosting, DART, итоговые рекомендации.
- Colab «Семинар 1. Библиотека LightGBM» (UCI Adult): Training API, callbacks/early stopping, кастомные метрики (F1), CV, веса объектов, интерпретация, гиперпараметры, начальные приближения, кастомные лоссы, `linear_tree`, CEGB, monotone/interaction constraints.
- Colab «Семинар 2. Библиотека CatBoost»: Pool, overfitting detector, логирование метрик, кастомные метрики/лоссы, веса, интерпретация (json-деревья, Interaction), Ordered vs Plain, `has_time`, сэмплинг (Bernoulli, MVS), квантизация, CTR-схемы (Counter, Borders, Buckets).
- `task_for_students.ipynb` — ДЗ «MLP против бустинга на табличных данных».

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Friedman J. *Greedy Function Approximation: A Gradient Boosting Machine* (2001); *Stochastic Gradient Boosting* (2002)
- Friedman J., Hastie T., Tibshirani R. *Additive Logistic Regression: a Statistical View of Boosting* (2000)
- Chen T., Guestrin C. *XGBoost: A Scalable Tree Boosting System* (KDD 2016)
- Ke G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (NeurIPS 2017)
- Prokhorenkova L. et al. *CatBoost: unbiased boosting with categorical features* (NeurIPS 2018)
- Rashmi K., Gilad-Bachrach R. *DART: Dropouts meet Multiple Additive Regression Trees* (2015)
- Shi Y. et al. *Gradient Boosting With Piece-Wise Linear Regression Trees* (2019, GBDT-PL / `linear_tree`)
- Учебник ML Яндекса: [Градиентный бустинг](https://education.yandex.ru/handbook/ml/article/gradientnyj-busting)

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): Градиентный бустинг
- **Семинары:** [Семинар 1: LightGBM](../../materials/seminars/09_Семинар_1_LightGBM.ipynb) ([Colab](https://colab.research.google.com/drive/18icSb8-xns30CsaVH6NKXF6m_c1-jHHf)) · [Семинар 2: CatBoost](../../materials/seminars/09_Семинар_2_CatBoost.ipynb) ([Colab](https://colab.research.google.com/drive/1IzHnPFzBCBpbj72VL59hkz7JmFbYyi_E))
- **Домашние задания:** [DL, ДЗ 1: MLP против бустинга (задание)](../../materials/homework/09-15_DL_ДЗ_1_MLP_против_бустинга.ipynb)

</details>

---

[← 08. Ансамбли: бэггинг, случайный лес, стекинг](../08_bagging_random_forest/README.md) · [Оглавление курса](../../README.md) · [10. Обнаружение аномалий →](../10_anomaly_detection/README.md)
