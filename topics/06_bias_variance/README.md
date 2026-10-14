# 06. Сложность моделей, bias–variance, переобучение

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Ожидаемая ошибка = **смещение²** (модель в среднем «не туда») + **дисперсия** (модель «скачет»
от выборки к выборке) + **шум** (неустранимо). Сложная модель — низкое смещение, высокая дисперсия
(переобучение); простая — наоборот (недообучение). Регуляризация, больше данных, ансамбли, early stopping,
стрижка деревьев снижают дисперсию.

| Термин | Определение |
|---|---|
| Bias (смещение) | $f(x)-\mathbb{E}_D\hat f_D(x)$ — систематическая ошибка модели |
| Variance (разброс) | $\mathbb{E}_D(\hat f_D-\mathbb{E}_D\hat f_D)^2$ — чувствительность к обучающей выборке |
| Noise | $\sigma^2$ — ошибка даже у истинной $f$ |
| Overfitting / underfitting | высокая дисперсия / высокое смещение |
| Double descent | после порога интерполяции ($p\approx n$) тестовая ошибка снова падает |
| Pre-pruning / post-pruning | ограничения при росте дерева / стрижка выращенного ($R_\alpha(T)=R(T)+\alpha\lvert T\rvert$) |

$$
\mathbb{E}\big(y-\hat f_D(x)\big)^2=\underbrace{\big(f(x)-\bar f(x)\big)^2}_{\text{bias}^2}+\underbrace{\mathbb{E}_D\big(\hat f_D(x)-\bar f(x)\big)^2}_{\text{variance}}+\underbrace{\sigma^2}_{\text{noise}},\qquad \bar f=\mathbb{E}_D\hat f_D
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>1. Зачем это всё</b></summary>

Работа с ML-алгоритмами: (1) **выбор модели** — алгоритм и гиперпараметры, минимизирующие ошибку;
(2) **оценка** на тестовых данных.

**Почему бы не взять модель, идеально проходящую через обучающие точки (интерполяцию)?**
Потому что задача — оценить **истинную зависимость** и минимизировать **тестовую** ошибку, а не обучающую.
Интерполяция минимизирует обучающую ошибку, но подгоняется под шум — тестовая ошибка при этом не минимальна.

</details>

<details>
<summary><b>2. Статистические основы: смещение и дисперсия оценки</b></summary>

По ограниченным случайным данным оцениваем параметр $\theta$. Оценка $\hat\theta$ — случайная величина
(зависит от выборки). Сравниваем оценки по среднему квадрату отклонения от истины:

$$
\text{MSE}(\hat\theta) = \mathbb{E}(\hat\theta - \theta)^2 = \underbrace{(\mathbb{E}\hat\theta - \theta)^2}_{\text{смещение}^2\ (\text{bias}^2)} + \underbrace{\mathbb{E}(\hat\theta - \mathbb{E}\hat\theta)^2}_{\text{дисперсия (variance)}} .
$$

Пример из лекции: одна оценка **несмещённая, но с большим разбросом**, другая **смещённая, но
плотнее сконцентрирована** вокруг истины. По MSE вторая может быть лучше — это и есть идея регуляризации
(пример с вычислениями — см. раздел «1. MSE оценки = смещение² + дисперсия»).

</details>

<details>
<summary><b>3. Разложение bias–variance–noise</b></summary>

Пусть $y = f(x) + \varepsilon$, $\mathbb{E}\varepsilon = 0$, $\operatorname{Var}\varepsilon = \sigma^2$; $f(x) = \mathbb{E}[y \mid x]$ — **истинная регрессия**
(она минимизирует ожидаемую ошибку при квадратичной потере). Модель $\hat f_D$ обучается на случайной
выборке $D$. Тогда ожидаемая ошибка в точке $x$:

$$
\mathbb{E}\big(y - \hat f_D(x)\big)^2 = \underbrace{\big(f(x) - \mathbb{E}_D\hat f_D(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\big(\hat f_D(x) - \mathbb{E}_D\hat f_D(x)\big)^2}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}} .
$$

| Компонента | Смысл | Тип ошибки |
|---|---|---|
| **Смещение** (bias) | насколько **в среднем по обучающим выборкам** прогноз отличается от истины | устранимая (reducible) |
| **Дисперсия** (variance) | насколько прогноз «скачет» вокруг своего среднего при смене обучающей выборки | устранимая |
| **Шум** (noise) | ошибка, которая есть даже у истинной функции регрессии | **неустранимая** (irreducible) |

Доказательство разложения (см. раздел «2. Разложение ошибки: bias–variance–noise»)

Четыре комбинации («мишени»): low bias + low variance — идеал; high bias + low variance — стабильно мажем;
low bias + high variance — в среднем попадаем, но разброс огромный; high bias + high variance — хуже всего.

</details>

<details>
<summary><b>4. Bias–variance tradeoff, переобучение и недообучение</b></summary>

Модели различаются **сложностью** (гибкостью). Зависимость смещения и разброса от сложности часто **обратная**:
- усложнение модели → смещение ↓, разброс ↑;
- упрощение → разброс ↓, смещение ↑.

| | **Недообучение** (underfitting) | **Хорошая модель** | **Переобучение** (overfitting) |
|---|---|---|---|
| Что происходит | модель слишком проста, не улавливает закономерности | улавливает зависимость, игнорирует шум | подстраивается под шум, теряет обобщающую способность |
| Компонента | высокое **смещение** | баланс | высокая **дисперсия** |
| Train error | высокая | низкая | очень низкая |
| Test error | высокая | минимальная | высокая |
| Что делать | усложнить модель, добавить признаки, ослабить регуляризацию | — | упростить модель, регуляризация, больше данных |

Тестовая ошибка как функция сложности — **U-образная кривая**; оптимальная сложность зависит от данных.

Пример из лекции — деревья разной глубины на трёх подвыборках: глубина 1 — все три дерева похожи, но
грубые (bias); глубина 7 — каждое дерево повторяет свою подвыборку (variance). Причём **дисперсия
немонотонна** по глубине.

**Double descent: tradeoff не абсолютен**

По современным исследованиям (Belkin, Hsu et al., 2019) рост разброса при снижении смещения — **не абсолютная истина**:
- мало параметров → недообучение; больше → «sweet spot»;
- когда параметров ровно столько, чтобы **запомнить** выборку, тестовая ошибка резко растёт (пик);
- **парадокс:** если добавлять параметры дальше, тестовая ошибка **снова падает**, иногда ниже классического оптимума.

Объяснение: избыточные параметры позволяют найти «самую простую» (с малой нормой) интерполяцию;
пик — точка, где модель смогла всё запомнить «самым напряжённым» способом
(разбор и код — см. раздел «5. Double descent: почему классическая U-кривая не всегда верна»).
Это объясняет, почему огромные нейросети обобщают.

</details>

<details>
<summary><b>5. Как бороться с переобучением на практике</b></summary>

В реальности у нас нет многих обучающих выборок и неизвестна истинная зависимость, поэтому
нарисовать bias–variance-график и найти теоретический минимум нельзя. Переобучение ловят **контролем
качества** (тема [05](../05_validation/README.md)) и лечат так:

| Метод | Суть | Что уменьшает |
|---|---|---|
| **Выборка специальной структуры** | баланс классов, репрезентативные примеры вблизи границы классов | variance |
| **Больше данных** | «данные первичны, алгоритмы вторичны»: запомнить много примеров труднее; минус — дорого | variance |
| **Аугментация** | CV: повороты, сдвиги, шум; NLP: синонимы, back-translation; таблицы: аддитивный шум | variance |
| **Улучшение качества данных** | удаление выбросов и дубликатов, исправление меток, заполнение пропусков | шум в разметке |
| **Чужие данные / модели** | transfer learning (ResNet на ImageNet), multi-task learning | variance |
| **Снижение размерности / отбор признаков** | PCA, t-SNE, UMAP, жадный отбор — меньше «мёртвых» признаков | variance |
| **Регуляризация** | снижение сложности: Ridge, LASSO, Dropout, pruning деревьев | variance (ценой bias) |
| **Организация контроля, early stopping** | train/val/test, CV; остановка, когда валидационный loss не падает N эпох/итераций | variance |
| **Выбор модели / архитектуры** | менее гибкая модель (ограничить глубину дерева, число слоёв); ансамбли (бэггинг, бустинг, стекинг), batch normalization | variance / bias |

</details>

<details>
<summary><b>6. Борьба с переобучением в решающих деревьях</b></summary>

Неограниченное дерево запоминает обучающую выборку (по листу на объект). Два подхода:

**Pre-pruning — сдерживать рост при обучении**

Гиперпараметры-ограничения, задаются до обучения:
- `max_depth` — глубина дерева;
- `min_samples_split`, `min_samples_leaf` — минимум объектов для разбиения / в листе;
- `max_leaf_nodes` — максимум листьев; `min_impurity_decrease` — минимальное улучшение критерия.

Плюсы: просто и дёшево. Минусы: повышенный риск недообучения (полезное разбиение может идти после
бесполезного — «эффект горизонта»).

**Post-pruning — стричь выращенное дерево**

**Cost-complexity pruning** (CART): строим полное дерево, затем для параметра $\alpha$ ищем поддерево,
минимизирующее

$$
R_\alpha(T) = R(T) + \alpha\,|T| ,
$$

$R(T)$ — ошибка на обучении, $|T|$ — число листьев. Большое $\alpha$ → маленькое дерево. $\alpha$ подбирают
на кросс-валидации (алгоритм «слабейшего звена» — см. раздел «6. Стрижка дерева: cost-complexity pruning»).

Плюсы: контролируемый баланс ошибка/сложность, часто лучше pre-pruning. Минусы: надо сначала
вырастить большое дерево (дорого); при неудачном $\alpha$ легко «отпилить» редкие, но важные ветви.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier(random_state=0)
path = tree.cost_complexity_pruning_path(X_train, y_train)          # все «критические» alpha
grid = GridSearchCV(tree, {"ccp_alpha": path.ccp_alphas[:-1]}, cv=5).fit(X_train, y_train)
print(grid.best_params_, grid.best_estimator_.get_n_leaves())
```

</details>

<details>
<summary><b>7. Как диагностировать</b></summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, learning_curve

# 1) Кривая валидации: train/CV от сложности
depths = range(1, 21)
tr, cv = validation_curve(DecisionTreeClassifier(random_state=0), X, y,
                          param_name="max_depth", param_range=depths, cv=5)
plt.plot(depths, tr.mean(1), label="train"); plt.plot(depths, cv.mean(1), label="CV"); plt.legend()
# слева обе низкие (недообучение), справа train→1, CV падает (переобучение)

# 2) Кривая обучения: train/CV от объёма данных
sizes, tr, cv = learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1, 8))
# большой устойчивый зазор → variance → поможет больше данных; кривые сошлись низко → bias
```

</details>

<details>
<summary><b>Итоги лекции</b></summary>

- Предсказания в ML строятся на статистическом обучении; ошибка раскладывается на **смещение, разброс и шум**.
- Цель — минимизировать смещение и разброс, которые часто обратно зависимы — **bias–variance tradeoff**.
- Бывают исключения (double descent).
- **Переобучение** — слишком высокий разброс; **недообучение** — слишком высокое смещение.
- В деревьях — pre-pruning и post-pruning.

</details>

**Математика и выводы**

<details>
<summary><b>1. MSE оценки = смещение² + дисперсия</b></summary>

Пусть $\hat\theta$ — оценка параметра $\theta$ по случайной выборке. Добавим и вычтем $\mathbb{E}\hat\theta$:

$$
\mathbb{E}(\hat\theta - \theta)^2 = \mathbb{E}\big(\hat\theta - \mathbb{E}\hat\theta + \mathbb{E}\hat\theta - \theta\big)^2
= \underbrace{\mathbb{E}(\hat\theta - \mathbb{E}\hat\theta)^2}_{\operatorname{Var}\hat\theta} + \underbrace{(\mathbb{E}\hat\theta - \theta)^2}_{\operatorname{bias}^2} ,
$$

перекрёстный член $2(\mathbb{E}\hat\theta - \theta)\,\mathbb{E}(\hat\theta - \mathbb{E}\hat\theta) = 0$.

**Смещённая оценка бывает лучше несмещённой**

Оцениваем среднее $\mu$ по $n$ наблюдениям с дисперсией $\sigma^2$. Рассмотрим «сжатую» оценку $c\,\bar x$:

$$
\text{MSE}(c) = \underbrace{(1-c)^2\mu^2}_{\text{bias}^2} + \underbrace{c^2\frac{\sigma^2}{n}}_{\text{variance}},
\qquad c^* = \frac{\mu^2}{\mu^2 + \sigma^2/n} < 1 .
$$

При $c = 1$ (несмещённая $\bar x$) MSE $= \sigma^2/n$; при $c^*$ — строго меньше. Это ровно ситуация со слайда
лекции «первая оценка несмещённая, но с высоким разбросом; вторая — смещённая, но точнее».
Регуляризация (Ridge, Lasso, стрижка деревьев) — это сознательное введение смещения ради уменьшения
дисперсии. (В многомерном случае — знаменитый парадокс Стейна: при $d \ge 3$ $\bar x$ недопустима по MSE.)

</details>

<details>
<summary><b>2. Разложение ошибки: bias–variance–noise</b></summary>

Модель данных: $y = f(x) + \varepsilon$, $\mathbb{E}\varepsilon = 0$, $\operatorname{Var}\varepsilon = \sigma^2$. Обучающая выборка $D$
случайна, по ней строится $\hat f_D$. Ожидаемая ошибка в точке $x$ (по выборкам $D$ и по шуму нового $y$):

$$
\mathbb{E}_{D,\varepsilon}\big(y - \hat f_D(x)\big)^2 = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{\big(f(x) - \bar f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\big(\hat f_D(x) - \bar f(x)\big)^2}_{\text{variance}},
\qquad \bar f(x) = \mathbb{E}_D\hat f_D(x) .
$$

**Доказательство.** Запишем $y - \hat f_D = \varepsilon + (f - \bar f) + (\bar f - \hat f_D)$. Возводим в квадрат и берём
математическое ожидание:
- $\mathbb{E}\varepsilon^2 = \sigma^2$;
- $(f - \bar f)^2$ — неслучайна;
- $\mathbb{E}_D(\bar f - \hat f_D)^2$ — дисперсия;
- перекрёстные члены: $\mathbb{E}[\varepsilon(\cdot)] = 0$, так как шум нового наблюдения не зависит от $D$ и имеет нулевое среднее;
  $2(f - \bar f)\,\mathbb{E}_D(\bar f - \hat f_D) = 0$ по определению $\bar f$. ∎

- **noise** — неустранимая ошибка (irreducible): её имеет даже истинная $f$;
- **bias² + variance** — устранимая (reducible) ошибка.

Почему истинная регрессия $f(x) = \mathbb{E}[y \mid x]$: при квадратичной потере она минимизирует
Expected Prediction Error (оптимальная константа для условного распределения — среднее, см. тему
[01](../01_intro_regression_metrics/README.md)).

**Как оценить разложение численно**

Нужны многие обучающие выборки — в симуляции генерируем их сами (так построен слайд с деревьями разной глубины):

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
x_test = np.linspace(0, 1, 200)[:, None]
sigma = 0.3

for depth in (1, 3, 7, None):
    preds = []
    for _ in range(200):                                    # 200 независимых обучающих выборок
        x = rng.uniform(0, 1, (100, 1))
        y = f(x[:, 0]) + rng.normal(0, sigma, 100)
        preds.append(DecisionTreeRegressor(max_depth=depth).fit(x, y).predict(x_test))
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(0) - f(x_test[:, 0])) ** 2)
    var = np.mean(preds.var(0))
    print(f"depth={depth}: bias²={bias2:.3f}  variance={var:.3f}  noise={sigma**2:.3f}  total≈{bias2 + var + sigma**2:.3f}")
```

**Для классификации**

Для 0–1 loss разложение не аддитивно: дисперсия может как увеличивать, так и **уменьшать** ошибку
(если среднее предсказание неверно, разброс иногда «случайно» исправляет ответ). Есть обобщения
(Domingos, 2000), но интуиция «сложнее модель → меньше смещение, больше разброс» сохраняется.

</details>

<details>
<summary><b>3. Ridge: смещение и дисперсия в явном виде</b></summary>

Для $y = Xw + \varepsilon$ и $\hat w_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$, обозначим $A_\lambda = X^\top X + \lambda I$:

$$
\mathbb{E}\hat w_\lambda - w = -\lambda A_\lambda^{-1}w, \qquad
\operatorname{Var}\hat w_\lambda = \sigma^2 A_\lambda^{-1}X^\top X A_\lambda^{-1} .
$$

В базисе собственных векторов $X^\top X$ (собственные значения $s_j$) по каждому направлению:

$$
\text{bias}_j^2 = \Big(\frac{\lambda}{s_j + \lambda}\Big)^2 w_j^2, \qquad \text{var}_j = \frac{\sigma^2 s_j}{(s_j + \lambda)^2} .
$$

При $\lambda = 0$: смещение 0, дисперсия $\sigma^2/s_j$ — огромна для малых $s_j$ (мультиколлинеарность).
Производная суммарной MSE по $\lambda$ в нуле равна $-2\sigma^2\sum_j s_j^{-2} < 0$, поэтому
**всегда существует $\lambda > 0$, при котором MSE Ridge строго меньше MSE OLS** (теорема Хоэрла–Кеннарда, 1970).

</details>

<details>
<summary><b>4. Усреднение моделей снижает дисперсию</b></summary>

Пусть $B$ моделей с одинаковой дисперсией $\sigma^2$ и попарной корреляцией $\rho$. Дисперсия среднего:

$$
\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^{B}\hat f_b\Big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2 .
$$

Смещение среднего равно смещению одной модели. Поэтому ансамбли из **сложных** (низкое смещение)
и **слабо коррелированных** моделей выигрывают: второе слагаемое → 0 с ростом $B$, а первое
уменьшают декорреляцией (бутстрэп, случайные подпространства) — это бэггинг и случайный лес,
тема [08](../08_bagging_random_forest/README.md).

</details>

<details>
<summary><b>5. Double descent: почему классическая U-кривая не всегда верна</b></summary>

**Наблюдение** (Belkin, Hsu, Ma, Mandal, 2019): если увеличивать число параметров $p$ дальше порога
интерполяции ($p \approx n$ — модель может идеально запомнить обучающую выборку), тестовая ошибка сначала
**взлетает**, а потом **снова падает**, иногда ниже классического «sweet spot».

**Почему пик при $p \approx n$.** Для линейной регрессии на случайных признаках при $p = n$ матрица $X$ квадратная
и почти наверняка плохо обусловлена: минимальное сингулярное число близко к 0, и интерполирующее
решение $w = X^{-1}y$ имеет огромную норму — подгонка шума делается «самым напряжённым» способом.

**Почему спад при $p > n$.** Интерполирующих решений становится бесконечно много; алгоритм выбирает
решение с **минимальной нормой** $w = X^+y = X^\top(XX^\top)^{-1}y$. С ростом $p$ у матрицы $XX^\top$ растут
малые собственные значения, норма минимального решения падает, подгонка распределяется по многим
«гладким» направлениям — фактически появляется неявная регуляризация.

**Неявная регуляризация градиентного спуска.** GD для $\|Xw - y\|^2$ из $w_0 = 0$ сходится именно к решению с
минимальной нормой. *Доказательство:* каждый шаг $w \leftarrow w - \eta X^\top(Xw - y)$ добавляет вектор из
пространства строк $X$, значит все итерации лежат в $\operatorname{row}(X)$; единственная интерполяция
в $\operatorname{row}(X)$ — это $X^+y$, решение с минимальной нормой. ∎

```python
import numpy as np
rng = np.random.default_rng(0)
n, d_true = 40, 100
X_all = rng.normal(size=(n + 1000, d_true)); w_true = rng.normal(size=d_true) / np.sqrt(d_true)
y_all = X_all @ w_true + 0.5 * rng.normal(size=n + 1000)
Xtr, ytr, Xte, yte = X_all[:n], y_all[:n], X_all[n:], y_all[n:]
for p in (5, 20, 35, 40, 45, 60, 100):
    w = np.linalg.pinv(Xtr[:, :p]) @ ytr                # МНК / min-norm решение по первым p признакам
    print(p, round(np.mean((Xte[:, :p] @ w - yte) ** 2), 2))   # пик около p = n = 40
```

</details>

<details>
<summary><b>6. Стрижка дерева: cost-complexity pruning</b></summary>

Для поддерева $T$ полного дерева $T_0$ определим функцию стоимости

$$
R_\alpha(T) = R(T) + \alpha\,|T| ,
$$

где $R(T)$ — ошибка дерева на обучении (сумма по листьям: MSE для регрессии, ошибка/неопределённость
для классификации), $|T|$ — число листьев, $\alpha \ge 0$ — штраф за сложность. Для каждого $\alpha$ ищем
поддерево $T_\alpha = \arg\min_{T \subseteq T_0} R_\alpha(T)$.

**Алгоритм «слабейшего звена».** Для внутренней вершины $t$ со своим поддеревом $T_t$ свернуть $T_t$ в лист
выгодно, когда $R(t) + \alpha \le R(T_t) + \alpha|T_t|$, т.е. при

$$
\alpha \ge g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1} .
$$

$g(t)$ — прирост ошибки на один удалённый лист. Последовательно сворачиваем вершину с минимальным $g(t)$,
получая **вложенную** последовательность поддеревьев $T_0 \supset T_1 \supset \dots \supset \lbrace\text{корень}\rbrace$ и
соответствующие пороги $\alpha_1 \le \alpha_2 \le \dots$ (Breiman et al., 1984). Оптимальное $\alpha$ выбирают на кросс-валидации.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
path = DecisionTreeRegressor(random_state=0).cost_complexity_pruning_path(X_train, y_train)
scores = [cross_val_score(DecisionTreeRegressor(ccp_alpha=a, random_state=0), X_train, y_train, cv=5).mean()
          for a in path.ccp_alphas]
best_alpha = path.ccp_alphas[int(np.argmax(scores))]
```

</details>

<details>
<summary><b>7. Ранняя остановка ≈ L2-регуляризация</b></summary>

Для GD на $\frac12\|Xw - y\|^2$ из нуля с шагом $\eta$ после $t$ шагов компонента решения вдоль
сингулярного направления с числом $s_j$ равна $\big(1 - (1 - \eta s_j^2)^t\big)\frac{u_j^\top y}{s_j}$, а у Ridge —
$\frac{s_j^2}{s_j^2 + \lambda}\cdot\frac{u_j^\top y}{s_j}$. Оба «фильтра» близки к 1 для больших $s_j$ и подавляют малые
$s_j$; соответствие $\lambda \approx \frac{1}{\eta t}$. Поэтому число итераций (эпох, деревьев в бустинге) — такой же
регуляризационный гиперпараметр, как $\lambda$, и его подбирают по валидации (early stopping).

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Как отличить переобучение от недообучения?</summary>

большой разрыв train/val — дисперсия; обе ошибки высокие — смещение.

</details>

<details>
<summary>Что поможет при высокой дисперсии?</summary>

больше данных, регуляризация, упрощение модели, бэггинг, early stopping. При высоком смещении — усложнить модель, добавить признаки, бустинг.

</details>

<details>
<summary>Почему бэггинг снижает дисперсию?</summary>

$\operatorname{Var}(\text{среднего})=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$.

</details>

<details>
<summary>Можно ли уменьшить шум $\sigma^2$ моделью?</summary>

нет; только лучшими признаками/данными.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- Смещённая оценка может иметь меньшую MSE: $c\bar x$ с $c^*=\frac{\mu^2}{\mu^2+\sigma^2/n}<1$; для Ridge всегда есть $\lambda>0$ лучше OLS (Хоэрл–Кеннард).
- Double descent: при $p>n$ МНК выбирает решение минимальной нормы $X^+y$; GD из нуля сходится именно к нему (неявная регуляризация).
- Ранняя остановка GD ≈ Ridge с $\lambda\approx 1/(\eta t)$.
- Cost-complexity pruning: сворачиваем узел с минимальным $g(t)=\frac{R(t)-R(T_t)}{\lvert T_t\rvert-1}$.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# Bias и variance деревьев разной глубины: 200 обучающих выборок из одного распределения
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
x_test = np.linspace(0, 1, 200)[:, None]
for depth in (1, 3, 7, None):
    preds = []
    for _ in range(200):
        x = rng.uniform(0, 1, (100, 1))
        preds.append(DecisionTreeRegressor(max_depth=depth).fit(x, f(x[:, 0]) + rng.normal(0, 0.3, 100)).predict(x_test))
    preds = np.array(preds)
    print(f"max_depth={str(depth):4}: bias²={np.mean((preds.mean(0) - f(x_test[:, 0]))**2):.3f}  "
          f"variance={preds.var(0).mean():.3f}  (noise=0.090)")
```

Результат запуска:

```text
max_depth=1   : bias²=0.076  variance=0.030  (noise=0.090)
max_depth=3   : bias²=0.004  variance=0.034  (noise=0.090)
max_depth=7   : bias²=0.000  variance=0.072  (noise=0.090)
max_depth=None: bias²=0.001  variance=0.091  (noise=0.090)
```

```python
# Кривая валидации: train растёт с глубиной, CV — сначала растёт, потом падает
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
depths = [1, 2, 3, 5, 8, 12, None]
tr, cv = validation_curve(DecisionTreeClassifier(random_state=0), X, y, param_name="max_depth",
                          param_range=depths, cv=5)
for d, a, b in zip(depths, tr.mean(1), cv.mean(1)):
    print(f"max_depth={str(d):4}  train={a:.3f}  CV={b:.3f}")
```

Результат запуска:

```text
max_depth=1     train=0.924  CV=0.900
max_depth=2     train=0.952  CV=0.928
max_depth=3     train=0.974  CV=0.917
max_depth=5     train=0.991  CV=0.916
max_depth=8     train=1.000  CV=0.916
max_depth=12    train=1.000  CV=0.917
max_depth=None  train=1.000  CV=0.917
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Bias-variance decomposition](https://education.yandex.ru/handbook/ml/article/bias-variance-decomposition)
- [Scott Fortmann-Roe — Understanding the Bias-Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html)
- ISLR (James, Witten, Hastie, Tibshirani), гл. 2.2
- Belkin et al., 2019 — [Reconciling modern ML practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)
- [sklearn: Post pruning decision trees with cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Сложность алгоритмов_final.pdf` — «Сложность алгоритмов и переобучение» (MACHINE LEARNING, неделя 6), все 39 страниц.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- James, Witten, Hastie, Tibshirani. *ISLR*, разд. 2.2 (bias-variance tradeoff)
- Hastie, Tibshirani, Friedman. *ESL*, разд. 7.3, 9.2.2 (cost-complexity pruning)
- Belkin M., Hsu D., Ma S., Mandal S. *Reconciling modern machine-learning practice and the classical bias–variance trade-off* (PNAS 2019)
- Nakkiran P. et al. *Deep Double Descent* (2019)
- Hoerl A., Kennard R. *Ridge Regression: Biased Estimation for Nonorthogonal Problems* (1970)
- Учебник ML Яндекса: [Bias-variance decomposition](https://education.yandex.ru/handbook/ml/article/bias-variance-decomposition)

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): Сложность алгоритмов (bias–variance)

</details>

---

[← 05. Контроль качества и выбор модели](../05_validation/README.md) · [Оглавление курса](../../README.md) · [07. Решающие деревья →](../07_decision_trees/README.md)
