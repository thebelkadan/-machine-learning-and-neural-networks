# 09. Градиентный бустинг: теория, XGBoost, LightGBM, CatBoost

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 08. Бэггинг и RF](../08_bagging_random_forest/README.md) · [10. Аномалии →](../10_anomaly_detection/README.md)

---

## 1. Идея

**Бустинг** — последовательный ансамбль: каждый следующий базовый алгоритм строится так, чтобы **исправлять
ошибки** уже построенной композиции. «Шагаем последовательно к таргету».

$$
a_T(x) = a_0(x) + \eta\sum_{t=1}^{T} b_t(x) .
$$

Если бэггинг/лес усредняет **сложные** модели и снижает дисперсию, то бустинг складывает **простые**
(неглубокие деревья, high bias) и снижает в первую очередь **смещение**.

---

## 2. Алгоритм градиентного бустинга

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

[Подробный вывод и связь с AdaBoost](deep.md#2-идея-градиентный-спуск-в-пространстве-ответов)

---

## 3. Регуляризация бустинга

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

---

## 4. Продвинутые методы: второй порядок (XGBoost)

Раскладываем функцию ошибки в ряд Тейлора до второго порядка ($g_i$, $h_i$ — первая и вторая производные
по ответу) и добавляем регуляризацию дерева $\gamma J + \frac\lambda2\sum w_j^2$. Для дерева с фиксированной структурой:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad
\text{Gain} = \frac12\Big[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Big] - \gamma ,
$$

$G_j = \sum_{i \in R_j} g_i$, $H_j = \sum_{i \in R_j} h_i$. **Критерий расщепления берётся из самой функции ошибки**,
а не из Джини/энтропии; можно строить дерево и рекурсивно обрезать ветви с Gain < 0
([вывод](deep.md#4-xgboost-вторые-производные-и-регуляризация)).

---

## 5. Современные реализации

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

### Ключевые идеи реализаций

- **Гистограммный подход**: каждый вещественный признак заранее разбивается на бины (≤ 255) → число
  проверяемых порогов ~ число бинов, а не число уникальных значений.
- **EFB** (Exclusive Feature Bundling): признаки с большим числом нулей, которые одновременно не ненулевые,
  жадно склеиваются в бандлы (оптимум — NP-полная задача).
- **GOSS**: объекты с большими градиентами берём все, из остальных сэмплируем, но с повышенным весом.
- **Игнорирование нулей/NaN**: убираем их, выбираем сплит, добавляем в «выгодное» поддерево.
- **CatBoost и проблема смещения**: значение в листе — сумма антиградиентов, и мы оцениваем значение
  в точке, **зная метку в ней** → утечка. Решение — **ordered boosting** (динамический бустинг): случайная
  перестановка объектов, для каждого объекта используем только информацию «до него»; то же для
  **smoothed target encoding** категорий ([подробно](deep.md#8-catboost-сдвиг-предсказаний-и-ordered-boosting)).
- **DART**: при построении нового дерева «выключаем» случайное подмножество старых (dropout для деревьев).
- **GBDT-PL** (`linear_tree=True` в LightGBM): линейные регрессии в листьях вместо констант — бустинг
  получает способность к экстраполяции.

---

## 6. Как настраивать (итог лекции + семинары)

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

---

## 7. Код

### sklearn

```python
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8)
hgb = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05, max_leaf_nodes=31,
                                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
                                     categorical_features="from_dtype")       # нативные категории и NaN
```

### LightGBM (Training API, семинар 1)

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

### CatBoost (семинар 2)

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

### Бустинг с нуля (регрессия, MSE)

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

---

## 8. Бустинг vs нейросети на таблицах

ДЗ 1 курса DL (`task_for_students.ipynb`, Covertype: 581 тыс. участков леса, 7 классов): CatBoost «по умолчанию»
(1000 деревьев глубины 6) даёт **0.890** точности на тесте; MLP нужно много настраивать, чтобы его догнать.
На таблицах бустинг обычно сильнее: ему всё равно, в каких шкалах признаки, он устойчив к неинформативным
признакам и почти не требует настройки. Подробнее про MLP — темы [14](../14_neural_networks_backprop/README.md)–[15](../15_nn_training/README.md).

---

## Типичные ошибки

- Обучать без раннего останова (бустинг переобучается с числом деревьев, в отличие от леса).
- Сравнивать модели, подобранные с разным числом деревьев/темпом, без контроля на валидации.
- Подавать категории в LightGBM/CatBoost как числа (Label Encoding), не указав, что это категории.
- Делать target encoding вручную без out-of-fold и отдавать его бустингу (утечка) — CatBoost решает это сам.
- Задавать веса объектов с суммой, сильно отличающейся от $N$, или нулевые веса.
- Интерпретировать `predict_proba` как откалиброванные вероятности при сильных весах классов (`is_unbalance`).

---

## Что почитать / посмотреть

- 📘 Учебник ML Яндекса: [Градиентный бустинг](https://education.yandex.ru/handbook/ml/article/gradientnyj-busting)
- 📄 Документация: [XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/index.html) (туториалы, model.html — вывод формул) · [LightGBM](https://lightgbm.readthedocs.io/en/latest/) (Features, Parameters Tuning) · [CatBoost](https://catboost.ai/docs/en/) (Algorithm Details, Training Parameters)
- 📄 [LightGBM and XGBoost explained (mlexplained.com)](https://mlexplained.com/2018/01/05/lightgbm-and-xgboost-explained/)
- 📄 Friedman, [Stochastic Gradient Boosting](http://statweb.stanford.edu/~jhf/ftp/stobst.pdf)

## Источники курса

- `Градиентный бустинг.pdf` — лекция (39 слайдов): FSAM, алгоритм, shrinkage, stochastic GB, TreeBoost, второй порядок, сравнение XGBoost/LightGBM/CatBoost, гистограммы, EFB, GOSS, ordered boosting, DART, итоговые рекомендации.
- Colab «Семинар 1. Библиотека LightGBM» (UCI Adult): Training API, callbacks/early stopping, кастомные метрики (F1), CV, веса объектов, интерпретация, гиперпараметры, начальные приближения, кастомные лоссы, `linear_tree`, CEGB, monotone/interaction constraints.
- Colab «Семинар 2. Библиотека CatBoost»: Pool, overfitting detector, логирование метрик, кастомные метрики/лоссы, веса, интерпретация (json-деревья, Interaction), Ordered vs Plain, `has_time`, сэмплинг (Bernoulli, MVS), квантизация, CTR-схемы (Counter, Borders, Buckets).
- `task_for_students.ipynb` — ДЗ «MLP против бустинга на табличных данных».


<!-- materials:start -->
**Файлы курса в репозитории:**

- Градиентный бустинг (лекция курса, файл в репозиторий не включён)
- [DL, ДЗ 1: MLP против бустинга (задание)](../../materials/homework/09-15_DL_ДЗ_1_MLP_против_бустинга.ipynb)
- [Семинар 1: LightGBM](../../materials/seminars/09_Семинар_1_LightGBM.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/18icSb8-xns30CsaVH6NKXF6m_c1-jHHf)
- [Семинар 2: CatBoost](../../materials/seminars/09_Семинар_2_CatBoost.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1IzHnPFzBCBpbj72VL59hkz7JmFbYyi_E)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [10. Обнаружение аномалий →](../10_anomaly_detection/README.md)
