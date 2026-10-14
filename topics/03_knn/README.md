# 03. Метрические алгоритмы: ближайший центроид, kNN, Надарая–Ватсон

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 02. Метрики классификации](../02_classification_metrics/README.md) · [04. Линейные модели →](../04_linear_models/README.md)

---

## 1. Идея

Чтобы определить свойства нового объекта, сравниваем его с теми, что уже видели.
**Метрические алгоритмы** находят в выборке наиболее близкие объекты и переносят их свойства на новый.
Основа — **гипотеза компактности**: близкие объекты, как правило, принадлежат одному классу / имеют близкие ответы.

Два основных подхода:
1. **Ближайший центроид** (nearest centroid) — сравниваем со «средним объектом» каждого класса;
2. **k ближайших соседей** (kNN, k-nearest neighbors) — сравниваем с самими объектами выборки.

Метрические алгоритмы — **непараметрические**: не предполагают форму зависимости, «модель» — это сама выборка
(lazy learning: обучение = запоминание).

---

## 2. Расстояние

**Расстояние (метрика)** $\rho(x, y)$ измеряет близость объектов в признаковом пространстве и удовлетворяет:
неотрицательности, тождеству неразличимых ($\rho = 0 \iff x = y$), симметричности, неравенству треугольника.

| Расстояние | Формула | Когда |
|---|---|---|
| **Минковского** $L_p$ | $\big(\sum_j \lvert x^j - y^j\rvert^p\big)^{1/p}$ | общее семейство |
| **Манхэттенское** $L_1$ | $\sum_j \lvert x^j - y^j\rvert$ | выбросы, высокая размерность; «движение по городским кварталам» |
| **Евклидово** $L_2$ | $\sqrt{\sum_j (x^j - y^j)^2}$ | по умолчанию для числовых признаков |
| **Чебышёва** $L_\infty$ | $\max_j \lvert x^j - y^j\rvert$ | максимальное отличие по одной координате |
| **Косинусное** | $1 - \frac{\langle x, y\rangle}{\lVert x\rVert\,\lVert y\rVert}$ | важны пропорции, а не абсолютные значения: тексты, эмбеддинги, предпочтения пользователей |
| **Хэмминга** | число несовпадающих позиций | категориальные признаки, строки одной длины |
| Махаланобиса | $\sqrt{(x-y)^\top\Sigma^{-1}(x-y)}$ | коррелированные признаки разного масштаба |
| Жаккара | $1 - \frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}$ | множества (корзины, теги) |
| Левенштейна | число правок строки | опечатки, строки разной длины |
| DTW | расстояние с выравниванием по времени | временные ряды |
| Canberra | $\sum_j \frac{\lvert x^j - y^j\rvert}{\lvert x^j\rvert + \lvert y^j\rvert}$ | признаки около нуля |

> ⚠️ Расстояние Минковского **чувствительно к масштабу признаков**: признак «доход в рублях» задавит
> «возраст в годах». **Масштабирование обязательно** (`StandardScaler`, `MinMaxScaler`).

Метрика подбирается эмпирически, но вдумчиво — по типу данных (см. таблицу) и на кросс-валидации.

---

## 3. Метод ближайшего центроида

1. Для каждого класса считаем **центроид** — среднее его объектов: $\mu_k = \frac{1}{n_k}\sum_{i:\,y_i = k} x_i$.
2. Новый объект относим к классу, чей центроид ближе: $a(x) = \arg\min_k \rho(x, \mu_k)$.

| Плюсы | Минусы |
|---|---|
| храним только $K$ центроидов, их можно адаптивно обновлять | годится только для (почти) линейно разделимых классов |
| быстрое предсказание — $K$ расстояний | плохо работает со сложной структурой и разным разбросом классов |
| крошечная модель | |

С евклидовой метрикой это **линейный классификатор** ([вывод](deep.md#2-ближайший-центроид--это-линейный-классификатор)).

```python
from sklearn.neighbors import NearestCentroid
NearestCentroid(metric="euclidean").fit(X_train, y_train).predict(X_test)
```

---

## 4. kNN

### 4.1. Алгоритм

Для нового объекта $u$:
1. Считаем расстояния до всех объектов выборки и упорядочиваем: $\rho(u, x_u^{(1)}) \le \rho(u, x_u^{(2)}) \le \dots$
2. Окрестность $N_k(u)$ — $k$ ближайших соседей (окрестность фиксированного радиуса подбирать неудобно — плотность данных разная).
3. Предсказание:
   - **классификация** — голосование большинства (majority vote): $a(u) = \arg\max_y \sum_{i=1}^{k}[y_u^{(i)} = y]$;
   - **регрессия** — среднее ответов соседей: $a(u) = \frac{1}{k}\sum_{i=1}^{k} y_u^{(i)}$ (можно медиану — устойчивее к выбросам).

### 4.2. Гиперпараметр $k$

$k$ отвечает за **локальность** алгоритма. Аналитически не выбирается — только **кросс-валидацией**.

| Маленький $k$ | Большой $k$ |
|---|---|
| чувствительность к шуму | устойчивость к шуму |
| сложная граница, подстраивается под каждый объект | гладкая граница |
| риск **переобучения** | потеря локальных паттернов, риск **недообучения** |

$k = 1$ даёт нулевую ошибку на обучении (каждый объект — сам себе сосед) — это ничего не говорит о качестве.
В бинарной классификации $k$ берут **нечётным**, чтобы не было ничьих.

### 4.3. Взвешенный kNN

Чем ближе сосед, тем больше его вес $w_i$:

$$
a(u) = \arg\max_y \sum_{i=1}^{k} w_i\,[y_u^{(i)} = y] \quad\text{(классификация)},\qquad
a(u) = \frac{\sum_{i=1}^{k} w_i\, y_u^{(i)}}{\sum_{i=1}^{k} w_i} \quad\text{(регрессия)}.
$$

Схемы весов:
- **по рангу**: $w_i = q^i$, $0 < q < 1$, или $w_i = \frac{k + 1 - i}{k}$;
- **обратное расстояние** (`weights="distance"` в sklearn): $w_i = \frac{1}{\rho(u, x_u^{(i)}) + \varepsilon}$;
- **ядро** $K$ от расстояния, делённого на ширину окна $h$: $w_i = K\big(\rho(u, x_u^{(i)}) / h\big)$ —
  гауссовское, Епанечникова, треугольное, прямоугольное, квартическое, косинусное
  ([таблица ядер](deep.md#5-регрессия-надараяватсона-как-локальная-константа)).

### 4.4. Регрессия Надарая–Ватсона

Обобщение взвешенного kNN: используем **все** точки выборки с ядерными весами:

$$
a(u) = \frac{\sum_{i=1}^{\ell} y_i\, K\big(\frac{\rho(u, x_i)}{h}\big)}{\sum_{i=1}^{\ell} K\big(\frac{\rho(u, x_i)}{h}\big)} .
$$

- **Ширина окна** $h$ (bandwidth) — аналог $k$: маленькое $h$ → переобучение, большое → недообучение.
- **Не экстраполирует**: прогноз — взвешенное среднее имеющихся $y_i$, вне диапазона данных он «застывает».

### 4.5. Особенности kNN

| Плюсы | Минусы |
|---|---|
| прост в понимании и реализации | вычислительно дорогой: $O(\ell d)$ на каждый запрос |
| непараметрический — нет предположений о форме зависимости | требователен к памяти: хранит всю выборку |
| улавливает локальные паттерны, сложные границы | чувствителен к выбору метрики и масштабированию |
| легко добавлять новые данные (без переобучения) | нет универсального $k$ — каждый раз подбирать |
| работает и для классификации, и для регрессии, и для поиска похожих | страдает от проклятия размерности |

### 4.6. Проклятие размерности

В пространстве с большим числом признаков данные сильно разрежены, а **все точки становятся примерно
равноудалены** — «ближайший сосед» теряет смысл. Теоретически эффективность kNN падает.
Но реальные данные обычно лежат на **низкоразмерном многообразии**, и kNN работает с его внутренней
размерностью ([детали и числа](deep.md#6-проклятие-размерности)). Помогают: отбор признаков,
снижение размерности (PCA), обученные эмбеддинги.

### 4.7. Быстрый поиск соседей (ANN)

В продакшене используют **приближённый поиск ближайших соседей** (ANN, approximate nearest neighbors) —
почти без потери качества, но на порядки быстрее:
- **KD-деревья**, Ball tree — точный поиск, эффективны при малой размерности;
- **LSH** (locality-sensitive hashing) — хеши, совпадающие у близких объектов;
- **HNSW** (hierarchical navigable small worlds) — графовый поиск, **SoTA**;
- библиотеки: **Annoy** (Spotify), **FAISS** (Meta), **hnswlib**.

Подробно HNSW и поиск по эмбеддингам — в теме [19](../19_contrastive_embeddings/README.md).

---

## 5. Код

### sklearn

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())   # масштабирование внутри пайплайна!
grid = GridSearchCV(pipe, {
    "kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
    "kneighborsclassifier__weights": ["uniform", "distance"],
    "kneighborsclassifier__metric": ["euclidean", "manhattan", "chebyshev"],
}, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)

# algorithm: 'brute' | 'kd_tree' | 'ball_tree' | 'auto'
KNeighborsRegressor(n_neighbors=10, weights="distance", algorithm="kd_tree")
```

### kNN с нуля (по мотивам ДЗ «Метрические и линейные алгоритмы»)

```python
import numpy as np
from collections import Counter

class KNN:
    KERNELS = {
        "uniform":      lambda d: 0.5 * (np.abs(d) <= 1),
        "triangular":   lambda d: (1 - np.abs(d)) * (np.abs(d) <= 1),
        "epanechnikov": lambda d: 0.75 * (1 - d ** 2) * (np.abs(d) <= 1),
        "gaussian":     lambda d: np.exp(-0.5 * d ** 2) / np.sqrt(2 * np.pi),
    }

    def __init__(self, k=5, weights="uniform", kernel="gaussian", bandwidth=1.0, task="clf", eps=1e-6):
        self.k, self.weights, self.kernel, self.h, self.task, self.eps = k, weights, kernel, bandwidth, task, eps

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y)   # «обучение» = запоминание
        return self

    def _neighbors(self, x):
        d = np.linalg.norm(self.X - x, axis=1)
        idx = np.argsort(d, kind="stable")[: self.k]
        return idx, d[idx]

    def _weights(self, d):
        if self.weights == "uniform":
            return np.ones_like(d)
        if self.weights == "distance":
            return 1.0 / (d + self.eps)
        return self.KERNELS[self.kernel](d / self.h)              # ядерные веса

    def predict(self, X):
        out = []
        for x in np.asarray(X, float):
            idx, d = self._neighbors(x)
            w, t = self._weights(d), self.y[idx]
            if self.task == "reg":
                out.append(np.sum(w * t) / (np.sum(w) + self.eps))
            else:
                votes = Counter()
                for wi, ti in zip(w, t):
                    votes[ti] += wi
                out.append(votes.most_common(1)[0][0])
        return np.array(out)
```

### Подбор $k$: кривая валидации

```python
from sklearn.model_selection import validation_curve
ks = range(1, 51, 2)
train_sc, val_sc = validation_curve(make_pipeline(StandardScaler(), KNeighborsClassifier()), X, y,
                                    param_name="kneighborsclassifier__n_neighbors",
                                    param_range=ks, cv=5)
# train падает с ростом k (при k=1 — 100%), val сначала растёт, затем падает
```

---

## 6. Типичные ошибки

- Не масштабировать признаки (или масштабировать **до** разбиения — утечка; scaler должен быть в пайплайне).
- Выбирать $k$ по ошибке на обучении (при $k = 1$ она всегда 0).
- Использовать евклидово расстояние на сырых категориальных кодах (`LabelEncoder`) — порядок категорий бессмысленный.
- Использовать kNN на сотнях «сырых» признаков без снижения размерности.
- Ждать от kNN экстраполяции (прогноз всегда внутри диапазона обучающих $y$).

---

## Что почитать / посмотреть

- 📘 Учебник ML Яндекса: [Метрические методы](https://education.yandex.ru/handbook/ml/article/metricheskiye-metody)
- 📄 [sklearn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- 📄 [Annoy](https://github.com/spotify/annoy) · [FAISS](https://github.com/facebookresearch/faiss) · [hnswlib](https://github.com/nmslib/hnswlib)

## Источники курса

- `Лекция 3+4_072523.pdf` — «Метрические и линейные модели» (MACHINE LEARNING, неделя 3), с. 1–29.
- Colab «Домашнее задание по теме „Метрические и линейные алгоритмы“» (ч. 1: BaseKNN, ядра, KNNRegressor, сравнение с sklearn на California Housing).
- `ДЗ_метрические_алгоритмы_*.ipynb` — решённое ДЗ.


<!-- materials:start -->
**Файлы курса в репозитории:**

- Лекции 3–4 (ML): метрические и линейные методы (лекция курса, файл в репозиторий не включён)
- [ДЗ: метрические алгоритмы (решение)](../../materials/homework/03_ДЗ_метрические_алгоритмы.ipynb)
- [ДЗ: метрические и линейные алгоритмы (задание)](../../materials/homework/03-04_ДЗ_Метрические_и_линейные_задание.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1TNDdFci_nIOayOFFzSxRHW0VP3Zpie1u)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [04. Линейные модели →](../04_linear_models/README.md)
