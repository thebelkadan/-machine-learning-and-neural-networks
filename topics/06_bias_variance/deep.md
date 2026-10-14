# 06. Bias–variance и переобучение — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

---

## 1. MSE оценки = смещение² + дисперсия

Пусть $\hat\theta$ — оценка параметра $\theta$ по случайной выборке. Добавим и вычтем $\mathbb{E}\hat\theta$:

$$
\mathbb{E}(\hat\theta - \theta)^2 = \mathbb{E}\big(\hat\theta - \mathbb{E}\hat\theta + \mathbb{E}\hat\theta - \theta\big)^2
= \underbrace{\mathbb{E}(\hat\theta - \mathbb{E}\hat\theta)^2}_{\operatorname{Var}\hat\theta} + \underbrace{(\mathbb{E}\hat\theta - \theta)^2}_{\operatorname{bias}^2} ,
$$

перекрёстный член $2(\mathbb{E}\hat\theta - \theta)\,\mathbb{E}(\hat\theta - \mathbb{E}\hat\theta) = 0$.

### Смещённая оценка бывает лучше несмещённой

Оцениваем среднее $\mu$ по $n$ наблюдениям с дисперсией $\sigma^2$. Рассмотрим «сжатую» оценку $c\,\bar x$:

$$
\text{MSE}(c) = \underbrace{(1-c)^2\mu^2}_{\text{bias}^2} + \underbrace{c^2\frac{\sigma^2}{n}}_{\text{variance}},
\qquad c^* = \frac{\mu^2}{\mu^2 + \sigma^2/n} < 1 .
$$

При $c = 1$ (несмещённая $\bar x$) MSE $= \sigma^2/n$; при $c^*$ — строго меньше. Это ровно ситуация со слайда
лекции «первая оценка несмещённая, но с высоким разбросом; вторая — смещённая, но точнее».
Регуляризация (Ridge, Lasso, стрижка деревьев) — это сознательное введение смещения ради уменьшения
дисперсии. (В многомерном случае — знаменитый парадокс Стейна: при $d \ge 3$ $\bar x$ недопустима по MSE.)

---

## 2. Разложение ошибки: bias–variance–noise

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
[01](../01_intro_regression_metrics/deep.md#21-mse--среднее)).

### Как оценить разложение численно

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

### Для классификации

Для 0–1 loss разложение не аддитивно: дисперсия может как увеличивать, так и **уменьшать** ошибку
(если среднее предсказание неверно, разброс иногда «случайно» исправляет ответ). Есть обобщения
(Domingos, 2000), но интуиция «сложнее модель → меньше смещение, больше разброс» сохраняется.

---

## 3. Ridge: смещение и дисперсия в явном виде

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

---

## 4. Усреднение моделей снижает дисперсию

Пусть $B$ моделей с одинаковой дисперсией $\sigma^2$ и попарной корреляцией $\rho$. Дисперсия среднего:

$$
\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^{B}\hat f_b\Big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2 .
$$

Смещение среднего равно смещению одной модели. Поэтому ансамбли из **сложных** (низкое смещение)
и **слабо коррелированных** моделей выигрывают: второе слагаемое → 0 с ростом $B$, а первое
уменьшают декорреляцией (бутстрэп, случайные подпространства) — это бэггинг и случайный лес,
тема [08](../08_bagging_random_forest/deep.md).

---

## 5. Double descent: почему классическая U-кривая не всегда верна

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

---

## 6. Стрижка дерева: cost-complexity pruning

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

---

## 7. Ранняя остановка ≈ L2-регуляризация

Для GD на $\frac12\|Xw - y\|^2$ из нуля с шагом $\eta$ после $t$ шагов компонента решения вдоль
сингулярного направления с числом $s_j$ равна $\big(1 - (1 - \eta s_j^2)^t\big)\frac{u_j^\top y}{s_j}$, а у Ridge —
$\frac{s_j^2}{s_j^2 + \lambda}\cdot\frac{u_j^\top y}{s_j}$. Оба «фильтра» близки к 1 для больших $s_j$ и подавляют малые
$s_j$; соответствие $\lambda \approx \frac{1}{\eta t}$. Поэтому число итераций (эпох, деревьев в бустинге) — такой же
регуляризационный гиперпараметр, как $\lambda$, и его подбирают по валидации (early stopping).

---

## Ссылки

- James, Witten, Hastie, Tibshirani. *ISLR*, разд. 2.2 (bias-variance tradeoff)
- Hastie, Tibshirani, Friedman. *ESL*, разд. 7.3, 9.2.2 (cost-complexity pruning)
- Belkin M., Hsu D., Ma S., Mandal S. *Reconciling modern machine-learning practice and the classical bias–variance trade-off* (PNAS 2019)
- Nakkiran P. et al. *Deep Double Descent* (2019)
- Hoerl A., Kennard R. *Ridge Regression: Biased Estimation for Nonorthogonal Problems* (1970)
- Учебник ML Яндекса: [Bias-variance decomposition](https://education.yandex.ru/handbook/ml/article/bias-variance-decomposition)

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb)
