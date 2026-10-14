# 02. Метрики классификации

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Для «чёткого» ответа (метки) всё качество описывает **матрица ошибок** TN/FP/FN/TP; из неё —
accuracy, precision, recall, F1. Для «нечёткого» ответа (оценки/вероятности) — **ROC-AUC** (качество
порядка) и **log loss** (качество вероятностей). Метка = оценка > **порог**, порог — гиперпараметр.

| Термин | Определение |
|---|---|
| FP / ошибка I рода | предсказали «1», на самом деле «0» (ложная тревога) |
| FN / ошибка II рода | предсказали «0», на самом деле «1» (пропуск) |
| Precision | доля верных среди предсказанных «1» |
| Recall = TPR | доля найденных среди всех настоящих «1» |
| FPR | доля «0», ошибочно названных «1» |
| ROC / AUC | кривая TPR(FPR) по всем порогам / площадь под ней = $P(a(x_+) > a(x_-))$ |
| OvR / OvO / Softmax | $K$ моделей / $K(K-1)/2$ моделей / одна модель с $K$ выходами |
| macro / micro / weighted | среднее по классам / по суммарным TP,FP,FN / с весами-долями |

**Ключевые формулы**

$$
\text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN},\quad
F_\beta=(1+\beta^2)\frac{P\cdot R}{\beta^2P+R},\quad F_1=\frac{2TP}{2TP+FP+FN}
$$

$$
\text{LogLoss}=-\frac1\ell\sum_{i}\big(y_i\ln p_i+(1-y_i)\ln(1-p_i)\big),\qquad
\text{AUC}=\frac{1}{\ell_+\ell_-}\sum_{y_i=1}\sum_{y_j=0}[a(x_i)>a(x_j)]
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>1. Задача классификации</b></summary>

**Классификация** — предсказание класса из **конечного** множества меток. Модель учится находить
границы между классами. Отличие от регрессии — в целевой переменной: конечное множество вместо $\mathbb{R}$.

| Вид | Описание | Как решать |
|---|---|---|
| **Бинарная** | 2 класса: «болен / здоров», «кошка / собака» | один классификатор |
| **Многоклассовая** (multiclass) | $K > 2$ взаимоисключающих класса | One-vs-Rest, One-vs-One, Softmax |
| **Многометочная** (multilabel) | у объекта может быть несколько меток: теги товара, `[1 0 1]` | $K$ независимых бинарных задач (наличие / отсутствие метки) |

**Сведение многоклассовой задачи к бинарным**

| Подход | Сколько моделей | Как предсказываем |
|---|---|---|
| **One-vs-All / One-vs-Rest** | $K$: «класс $k$ против всех остальных» | класс с самой уверенной оценкой |
| **One-vs-One** | $C_K^2 = K(K-1)/2$: по модели на пару классов | голосование: побеждает класс, выбранный чаще всего |
| **Softmax** | 1 модель с $K$ выходами | вектор вероятностей через softmax, берём $\arg\max$ |

При $K = 4$: OvR — 4 модели, OvO — 6 моделей. Softmax-подход — например, `LogisticRegression` в sklearn
(multinomial), любая нейросеть-классификатор.

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

OneVsRestClassifier(LinearSVC())      # K моделей
OneVsOneClassifier(LinearSVC())       # K(K-1)/2 моделей
LogisticRegression()                  # multinomial (softmax) для K>2 по умолчанию
```

</details>

<details>
<summary><b>2. «Чёткая» классификация: модель выдаёт метку</b></summary>

**2.1. Матрица ошибок (Confusion Matrix)**

Ориентация как в **sklearn**: строки — истина, столбцы — предсказание.

| | $\hat y = 0$ | $\hat y = 1$ |
|---|---|---|
| $y = 0$ | **TN** = 13 599 | **FP** = 2 600 — *ошибка I рода* |
| $y = 1$ | **FN** = 898 — *ошибка II рода* | **TP** = 903 |

```python
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y, a).ravel()
```

Аналогия с экзаменом из лекции (нулевая гипотеза «студент не знает»):
- **Ошибка I рода (FP)** — не учил, но сдал (экзаменатор решил, что знает);
- **Ошибка II рода (FN)** — учил, но не сдал (экзаменатор решил, что не знает).

**2.2. Основные метрики**

| Метрика | Формула | Вопрос, на который отвечает | Пример (матрица выше) |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | какая доля ответов верна? | 0.806 |
| **Precision** (PPV) | $\frac{TP}{TP + FP}$ | если модель сказала «1», как часто она права? | 0.258 |
| **Recall** = **TPR** = Sensitivity | $\frac{TP}{TP + FN}$ | какую долю положительных нашли? | 0.501 |
| **FPR** | $\frac{FP}{FP + TN}$ | какую долю отрицательных ошибочно пометили как «1»? | 0.161 |
| **Specificity** = TNR | $\frac{TN}{TN + FP} = 1 - \text{FPR}$ | какую долю отрицательных распознали? | 0.839 |
| **F1** | $\frac{2PR}{P + R} = \frac{2TP}{2TP + FP + FN}$ | баланс P и R | 0.340 |
| **$F_\beta$** | $(1+\beta^2)\frac{P R}{\beta^2 P + R}$ | $\beta > 1$ — важнее Recall, $\beta < 1$ — важнее Precision | $F_2$ = 0.422 |

> **Важно.** **Ловушка accuracy.** Здесь положительных всего 10%. Константа «всегда 0» даёт accuracy
> **0.900** — больше, чем у модели (0.806)! На несбалансированных данных accuracy почти бесполезна:
> смотрите Precision / Recall / F1, balanced accuracy, PR-AUC.

**Почему гармоническое среднее в F1?** Оно штрафует дисбаланс между P и R: при $P = 1$, $R = 0.01$
арифметическое среднее 0.5, а F1 ≈ 0.02. F1 высока, только если **обе** метрики высоки.

**2.3. Precision–Recall tradeoff**

Модель обычно выдаёт оценку, а метку получаем сравнением с **порогом** (threshold) — гиперпараметром.

- **Низкий порог** → почти всех больных находим (высокий Recall), но часто путаем здоровых с больными (низкий Precision).
- **Высокий порог** → находим только «очевидных» больных (высокий Precision), упускаем остальных (низкий Recall).

Порог выбирают под задачу: по максимуму F1 на валидации, под требуемый Recall (медицина, антифрод),
или по стоимостям ошибок: $t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}$ (вывод (см. раздел «5. Оптимальный порог при разных стоимостях ошибок»)).

```python
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_val, proba_val)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_thr = thr[f1[:-1].argmax()]
```

</details>

<details>
<summary><b>3. «Нечёткая» классификация: модель выдаёт оценку / вероятность</b></summary>

**3.1. ROC-кривая и AUC-ROC**

**ROC-кривая** (receiver operating characteristic) — график TPR (ось Y) от FPR (ось X) при **всех**
порогах. Каждый порог — точка на кривой: при пороге $+\infty$ получаем $(0, 0)$, при $-\infty$ — $(1, 1)$.

**AUC-ROC** (area under curve) — площадь под ROC-кривой:
- **1** — идеальный классификатор (все положительные выше всех отрицательных);
- **0.5** — случайный;
- **0** — идеально «наоборот» (инвертируйте ответы).

**Смысл AUC:** доля правильно упорядоченных пар (положительный, отрицательный) — вероятность того,
что случайный положительный объект получит оценку выше случайного отрицательного
(доказательство (см. раздел «1. AUC-ROC = вероятность правильно упорядоченной пары»)).

Функционал зависит **не от значений оценок, а от их порядка**.

| Плюсы AUC-ROC | Минусы AUC-ROC |
|---|---|
| полезен там, где важен порядок (ранжирование, скоринг) | «завышает» качество на сильном дисбалансе (много TN) |
| не зависит от пропорций классов | оценивает не конкретный классификатор (порог), а «регрессию»-скоринг |
| опирается на порядок, а не на значения | ничего не говорит о калибровке вероятностей |
| можно оценивать отдельные признаки (AUC признака как скора) | сложно объяснить заказчику |

**Gini** $= 2\,\text{AUC} - 1$ — то же самое в шкале $[-1, 1]$, принято в кредитном скоринге.

**3.2. PR-кривая и Average Precision**

PR-кривая — Precision от Recall при всех порогах. Площадь под ней — **PR-AUC / Average Precision (AP)**.
Для редкого положительного класса PR-кривая информативнее ROC: случайный классификатор
даёт AP ≈ доле положительных (а не 0.5), а Precision сразу «чувствует» рост FP.

**3.3. Log Loss (бинарная кросс-энтропия)**

Модель при обучении минимизирует не метрику, а **функцию потерь**. Для вероятностной классификации
основной loss — **Log Loss** (logistic loss, cross-entropy):

$$
\text{LogLoss} = -\frac{1}{\ell}\sum_{i=1}^{\ell}\Big( y_i \ln p_i + (1 - y_i)\ln(1 - p_i) \Big),
$$

где $p_i$ — предсказанная вероятность класса 1. Раздельная запись: штраф $-\ln p_i$ для $y_i = 1$ и
$-\ln(1 - p_i)$ для $y_i = 0$.

- **Нельзя уверенно ошибаться**: $p = 0.001$ при $y = 1$ даёт штраф $\approx 6.9$, при $p \to 0$ — бесконечность.
- Оптимальная константа — **доля класса 1**: $p^* = \bar y$ (вывод (см. раздел «3. Log Loss: вывод из правдоподобия и оптимальная константа»)).
- Log loss = MLE для распределения Бернулли и **собственное правило**: минимум ожидаемого
  штрафа — при истинной вероятности. Поэтому модели на log loss выдают осмысленные вероятности.
- Многоклассовый вариант: $-\frac{1}{\ell}\sum_i\sum_k y_{ik}\ln p_{ik}$ (softmax + cross-entropy).

**Выводы лекции:** Log Loss ~ **вероятность**; ROC-AUC ~ **порядок**.

</details>

<details>
<summary><b>4. Метрики многоклассовой и многометочной классификации</b></summary>

Метрики P, R, F1 считаются **для каждого класса** (один против всех), затем усредняются:

| Усреднение | Как | Особенность |
|---|---|---|
| `macro` | среднее по классам | все классы равноправны — чувствительно к редким классам |
| `weighted` | среднее с весами = доли классов | близко к accuracy на дисбалансе |
| `micro` | суммируем TP, FP, FN по всем классам, потом формула | для multiclass micro-F1 = accuracy (доказательство (см. раздел «6. Многоклассовые усреднения»)) |

Для **multilabel**: Hamming loss (доля неверно предсказанных пар «объект–метка»),
subset accuracy (все метки объекта угаданы), те же P/R/F1 с усреднениями `micro / macro / samples`.

Дополнительно: **balanced accuracy** = $\frac12(\text{TPR} + \text{TNR})$ (среднее Recall по классам),
**MCC** (корреляция Мэтьюса, учитывает все 4 клетки матрицы), **Cohen's κ**.

```python
from sklearn.metrics import (classification_report, f1_score, balanced_accuracy_score,
                             matthews_corrcoef, hamming_loss, roc_auc_score, log_loss)

print(classification_report(y_true, y_pred, digits=3))       # P/R/F1 по классам + усреднения
f1_score(y_true, y_pred, average="macro")
roc_auc_score(y_true, proba, multi_class="ovr", average="macro")   # proba: (n, K)
log_loss(y_true, proba)
```

</details>

<details>
<summary><b>5. Как выбрать метрику</b></summary>

| Ситуация | Что смотреть |
|---|---|
| Классы сбалансированы, ошибки равноценны | Accuracy, F1 |
| Сильный дисбаланс, важен редкий класс | Precision, Recall, F1, PR-AUC, balanced accuracy, MCC |
| Пропуск положительного очень дорог (болезнь, мошенничество) | Recall (при ограничении на Precision), $F_2$ |
| Ложная тревога дорога (спам-фильтр, блокировка карты) | Precision, $F_{0.5}$ |
| Нужно ранжирование / скоринг, порог выберут потом | ROC-AUC (Gini), PR-AUC |
| Нужны честные вероятности (расчёт ожидаемой прибыли, ставки) | Log Loss, Brier score + калибровка (тема [04](../04_linear_models/README.md)) |
| Многоклассовая с дисбалансом | macro-F1 |

</details>

<details>
<summary><b>6. Код: полный пример</b></summary>

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, log_loss, confusion_matrix)

X, y = make_classification(n_samples=5000, weights=[0.9, 0.1], random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

proba = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
pred = (proba > 0.5).astype(int)

print(confusion_matrix(y_te, pred))
print(f"acc={accuracy_score(y_te, pred):.3f}  P={precision_score(y_te, pred):.3f}  "
      f"R={recall_score(y_te, pred):.3f}  F1={f1_score(y_te, pred):.3f}")
print(f"ROC-AUC={roc_auc_score(y_te, proba):.3f}  PR-AUC={average_precision_score(y_te, proba):.3f}  "
      f"LogLoss={log_loss(y_te, proba):.3f}")
print("бейзлайн-константа accuracy:", 1 - y_te.mean())
```

</details>

<details>
<summary><b>7. Типичные ошибки</b></summary>

- Отчитываться accuracy на несбалансированных данных.
- Считать ROC-AUC по **меткам** (`predict`) вместо оценок (`predict_proba` / `decision_function`).
- Оставлять порог 0.5 по умолчанию, не подбирая его под задачу.
- Подбирать порог на тесте (это утечка — подбирать на валидации).
- Интерпретировать `predict_proba` как вероятность без проверки калибровки (у SVM, RF, бустинга она часто кривая).
- Путать ориентацию матрицы ошибок (в sklearn строки — истина).

</details>

**Математика и выводы**

<details>
<summary><b>1. AUC-ROC = вероятность правильно упорядоченной пары</b></summary>

Пусть классификатор выдаёт оценку $a(x) \in \mathbb{R}$ (не обязательно вероятность).
ROC-кривая — множество точек $(\text{FPR}(t), \text{TPR}(t))$ при пороге $t$ от $+\infty$ до $-\infty$:

$$
\text{TPR}(t) = P\big(a(x) > t \mid y = 1\big), \qquad \text{FPR}(t) = P\big(a(x) > t \mid y = 0\big).
$$

**Утверждение.** Площадь под ROC-кривой равна

$$
\text{AUC} = P\big(a(x_+) > a(x_-)\big) + \tfrac12 P\big(a(x_+) = a(x_-)\big),
$$

где $x_+$ — случайный положительный, $x_-$ — случайный отрицательный объект.

**Доказательство (непрерывный случай, без совпадений).** Обозначим через $F_0, F_1$ функции
распределения оценок в классах 0 и 1, а через $f_0$ — плотность в классе 0. Тогда
$\text{TPR}(t) = 1 - F_1(t)$, $\text{FPR}(t) = 1 - F_0(t)$. Площадь:

$$
\text{AUC} = \int_0^1 \text{TPR}\, d\,\text{FPR} = \int_{+\infty}^{-\infty} \big(1 - F_1(t)\big)\, d\big(1 - F_0(t)\big)
= \int_{-\infty}^{+\infty} \big(1 - F_1(t)\big) f_0(t)\, dt .
$$

Но $1 - F_1(t) = P(a(x_+) > t)$, и интеграл по $f_0(t)\,dt$ — это усреднение по $t = a(x_-)$:

$$
\text{AUC} = \mathbb{E}_{x_-}\, P\big(a(x_+) > a(x_-)\big) = P\big(a(x_+) > a(x_-)\big). \;\blacksquare
$$

**Эмпирическая формула** (на выборке с $\ell_+$ положительными и $\ell_-$ отрицательными):

$$
\text{AUC} = \frac{1}{\ell_+ \ell_-}\sum_{i:\,y_i = 1}\;\sum_{j:\,y_j = 0} \Big( [a(x_i) > a(x_j)] + \tfrac12 [a(x_i) = a(x_j)] \Big).
$$

Это доля **правильно упорядоченных пар** «положительный выше отрицательного» — то, что на слайде
лекции названо «число правильно отсортированных пар». С точностью до нормировки это
статистика Манна–Уитни $U$, поэтому AUC можно посчитать за $O(\ell \log \ell)$ через ранги:

$$
\text{AUC} = \frac{\sum_{i:\,y_i=1} \operatorname{rank}(a(x_i)) - \frac{\ell_+(\ell_+ + 1)}{2}}{\ell_+ \ell_-} .
$$

```python
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def auc_pairs(y, s):                        # O(l+ * l-)
    pos, neg = s[y == 1], s[y == 0]
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0) + 0.5 * (diff == 0)).mean()

def auc_ranks(y, s):                        # O(l log l), ранги со средними для совпадений
    r = rankdata(s)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 1000); s = y + rng.normal(0, 1.5, 1000)
print(auc_pairs(y, s), auc_ranks(y, s), roc_auc_score(y, s))   # совпадают
```

**Следствия**

1. **AUC зависит только от порядка оценок**: инвариантен к любому строго монотонному
   преобразованию $a \mapsto g(a)$. Поэтому AUC не проверяет калибровку вероятностей.
2. **AUC не зависит от соотношения классов**: TPR и FPR считаются внутри своих классов
   (условные вероятности), доля классов в формулу не входит.
3. Случайные оценки: $P(a(x_+) > a(x_-)) = 1/2$ → AUC = 0.5. Если AUC < 0.5, инвертированный
   классификатор $-a(x)$ даёт $1 - \text{AUC} > 0.5$.
4. **Коэффициент Джини** (кредитный скоринг): $\text{Gini} = 2\,\text{AUC} - 1 \in [-1, 1]$.
5. AUC можно обобщить на регрессию — доля пар $(i, j)$ с $y_i > y_j$, упорядоченных моделью
   правильно (это связано с τ Кендалла).

</details>

<details>
<summary><b>2. Почему Precision зависит от баланса классов, а ROC — нет</b></summary>

Пусть $\pi = P(y = 1)$ — доля положительных. По формуле Байеса:

$$
\text{Precision} = P(y = 1 \mid \hat y = 1) = \frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi + \text{FPR}\cdot(1-\pi)} .
$$

При фиксированных TPR и FPR (т.е. той же ROC-точке) Precision **падает**, когда $\pi$ уменьшается.
Пример: TPR = 0.9, FPR = 0.05. При $\pi = 0.5$ Precision ≈ 0.95; при $\pi = 0.01$ —
$\frac{0.009}{0.009 + 0.0495} \approx 0.15$. Поэтому:
- ROC-AUC «завышает» впечатление о качестве на сильно несбалансированных данных
  (большое число TN маскирует FP);
- PR-кривая и **Average Precision** честнее показывают качество для редкого класса;
  базовая линия PR-кривой — горизонталь на уровне $\pi$ (а не 0.5).

</details>

<details>
<summary><b>3. Log Loss: вывод из правдоподобия и оптимальная константа</b></summary>

**3.1. Вывод**

Модель выдаёт $p_i = P(y_i = 1 \mid x_i)$. Распределение Бернулли: $P(y_i \mid x_i) = p_i^{y_i}(1 - p_i)^{1 - y_i}$.
Правдоподобие независимой выборки и его логарифм:

$$
\mathcal{L} = \prod_{i=1}^{\ell} p_i^{y_i}(1-p_i)^{1-y_i}, \qquad
\ln \mathcal{L} = \sum_{i=1}^{\ell}\Big( y_i \ln p_i + (1-y_i)\ln(1-p_i) \Big).
$$

Максимизация правдоподобия ⇔ минимизация **log loss** (бинарной кросс-энтропии):

$$
\text{LogLoss} = -\frac{1}{\ell}\sum_{i=1}^{\ell}\Big( y_i \ln p_i + (1-y_i)\ln(1-p_i) \Big).
$$

Раздельная запись (как на слайде): потеря объекта равна $-\ln p_i$, если $y_i = 1$, и
$-\ln(1 - p_i)$, если $y_i = 0$. Уверенная ошибка ($p_i \to 0$ при $y_i = 1$) даёт штраф $\to \infty$.

**3.2. Оптимальная константа**

Пусть $p_i \equiv p$. Тогда $Q(p) = -\big(\bar y \ln p + (1 - \bar y)\ln(1-p)\big)$, где $\bar y$ — доля единиц:

$$
Q'(p) = -\frac{\bar y}{p} + \frac{1-\bar y}{1-p} = 0 \;\Rightarrow\; p^* = \bar y .
$$

Значение в оптимуме — энтропия $H(\bar y) = -\bar y\ln\bar y - (1-\bar y)\ln(1-\bar y)$ — это
**бейзлайн для log loss**: модель с log loss выше $H(\bar y)$ хуже константы.

**3.3. Log loss — строго собственное правило (proper scoring rule)**

Пусть истинная вероятность $q = P(y = 1 \mid x)$. Ожидаемый штраф за прогноз $p$:

$$
\mathbb{E}_y\big[\ell(y,p)\big] = -q\ln p - (1-q)\ln(1-p) = H(q) + \mathrm{KL}\big(q \,\|\, p\big),
$$

и минимум достигается **ровно при $p = q$** (KL ≥ 0, равенство только при $p = q$). То есть log loss
поощряет **честные (калиброванные) вероятности**. То же верно для **Brier score**
$\frac{1}{\ell}\sum (p_i - y_i)^2$: $\mathbb{E}(p - y)^2 = (p - q)^2 + q(1-q)$.
Accuracy, F1, AUC собственными правилами не являются.

**3.4. Градиент по логиту**

Если $p = \sigma(z) = \frac{1}{1 + e^{-z}}$, то $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ и

$$
\frac{\partial\, \text{LogLoss}_i}{\partial z_i} = -\frac{y_i}{p_i}p_i(1-p_i) + \frac{1-y_i}{1-p_i}p_i(1-p_i) = p_i - y_i .
$$

Простой градиент «предсказание минус ответ» — причина, почему пара sigmoid + log loss так удобна
(подробно — тема [04](../04_linear_models/README.md)).

**3.5. Многоклассовый случай**

Для $K$ классов с вероятностями $p_{ik} = \operatorname{softmax}(z_i)_k = \frac{e^{z_{ik}}}{\sum_j e^{z_{ij}}}$ и one-hot ответом $y_{ik}$:

$$
\text{CE} = -\frac{1}{\ell}\sum_{i=1}^{\ell}\sum_{k=1}^{K} y_{ik}\ln p_{ik}, \qquad \frac{\partial\, \text{CE}_i}{\partial z_{ik}} = p_{ik} - y_{ik} .
$$

</details>

<details>
<summary><b>4. F-мера: откуда гармоническое среднее и $\beta$</b></summary>

$$
F_\beta = (1 + \beta^2)\frac{P \cdot R}{\beta^2 P + R} = \frac{(1+\beta^2)\,TP}{(1+\beta^2)\,TP + \beta^2 FN + FP}.
$$

- $F_\beta$ — взвешенное гармоническое среднее: $\frac{1}{F_\beta} = \frac{1}{1+\beta^2}\cdot\frac{1}{P} + \frac{\beta^2}{1+\beta^2}\cdot\frac{1}{R}$.
- Гармоническое среднее близко к **минимуму** из двух чисел: при $P = 1, R = 0.01$ арифметическое
  среднее 0.505, гармоническое ≈ 0.02. Поэтому F1 нельзя «накрутить», максимизируя одну метрику.
- Смысл $\beta$ (по ван Рейсбергену): $\beta$ — во сколько раз Recall важнее Precision.
  Формально $\beta^2 = $ отношение весов при FN и FP в знаменателе.
- $F_1 = \frac{2TP}{2TP + FP + FN}$ не зависит от TN — удобна, когда негативов очень много.

**Оптимальный порог для F1 не равен 0.5.** Для откалиброванных вероятностей оптимальный порог
$t^* = F_1^*/2$, где $F_1^*$ — достижимая максимальная F1 (Lipton et al., 2014). Так как обычно
$F_1^* < 1$, оптимальный порог < 0.5 — порог надо подбирать на валидации.

</details>

<details>
<summary><b>5. Оптимальный порог при разных стоимостях ошибок</b></summary>

Пусть ложноположительная ошибка стоит $c_{FP}$, ложноотрицательная — $c_{FN}$, правильные ответы бесплатны.
Для объекта с откалиброванной вероятностью $p$:

- ответ «1» → ожидаемые потери $(1-p)\,c_{FP}$;
- ответ «0» → ожидаемые потери $p\,c_{FN}$.

Выгодно отвечать «1», если $(1-p)c_{FP} < p\,c_{FN}$, т.е.

$$
p > t^* = \frac{c_{FP}}{c_{FP} + c_{FN}} .
$$

При равных стоимостях $t^* = 0.5$. Если пропустить болезнь в 9 раз дороже ложной тревоги
($c_{FN} = 9c_{FP}$), то $t^* = 0.1$. **Порог — бизнес-гиперпараметр**, а не константа 0.5.

</details>

<details>
<summary><b>6. Многоклассовые усреднения</b></summary>

Для класса $k$ считаем $TP_k, FP_k, FN_k$ по схеме «один против всех».

$$
\text{macro-}F_1 = \frac{1}{K}\sum_{k} F_1^{(k)}, \qquad
\text{weighted-}F_1 = \sum_k \frac{n_k}{\ell} F_1^{(k)}, \qquad
\text{micro-}F_1 = \frac{2\sum_k TP_k}{2\sum_k TP_k + \sum_k FP_k + \sum_k FN_k}.
$$

**Утверждение.** В однометочной многоклассовой задаче micro-Precision = micro-Recall = micro-F1 = Accuracy.

*Доказательство.* Каждый неверный ответ «класс $j$ вместо истинного $k$» — одновременно один $FP_j$ и
один $FN_k$. Значит $\sum_k FP_k = \sum_k FN_k = $ число ошибок $E$, а $\sum_k TP_k = \ell - E$. Тогда
micro-P $= \frac{\ell - E}{\ell - E + E} = \frac{\ell - E}{\ell} = $ Accuracy, так же micro-R; F1 от двух
равных чисел равна им. ∎

Поэтому для дисбаланса классов смотрят **macro**-усреднение: оно даёт редким классам тот же вес,
что и частым.

</details>

<details>
<summary><b>7. Сводка других метрик</b></summary>

| Метрика | Формула | Когда |
|---|---|---|
| Balanced accuracy | $\frac12(\text{TPR} + \text{TNR})$ | дисбаланс; константа даёт 0.5 |
| MCC (Matthews) | $\frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | корреляция Пирсона между $y$ и $\hat y$; использует все 4 клетки; $[-1, 1]$ |
| Cohen's κ | $\frac{p_o - p_e}{1 - p_e}$ | согласие сверх случайного ($p_e$ — accuracy случайного угадывания с теми же частотами) |
| Average Precision | $\sum_n (R_n - R_{n-1}) P_n$ | площадь под PR-кривой (ступенчатая) |
| Hamming loss | доля неверных пар (объект, метка) | multilabel |
| Subset accuracy | доля объектов, где **все** метки угаданы | multilabel, строгая |

</details>

<details>
<summary><b>8. One-vs-Rest и One-vs-One: число моделей</b></summary>

- **OvR**: $K$ моделей, каждая обучается на всех $\ell$ объектах. Предсказание — $\arg\max_k a_k(x)$.
  Проблема: оценки разных моделей не обязаны быть сравнимы по шкале; дисбаланс «1 против $K-1$».
- **OvO**: $C_K^2 = \frac{K(K-1)}{2}$ моделей, каждая — на объектах только двух классов
  (≈ $2\ell/K$ объектов). Предсказание — голосование. При $K = 4$ — 6 моделей (как на слайде).
  Выгодно для алгоритмов, чья сложность растёт сверхлинейно по $\ell$ (ядровой SVM).
- **Softmax**: одна модель с $K$ выходами и кросс-энтропией (multinomial logistic regression, нейросети).

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Accuracy при 1% положительных: константа «0» даёт 99%.</summary>

Смотрите P/R/F1, PR-AUC, balanced accuracy.

</details>

<details>
<summary>AUC инвариантен к монотонным преобразованиям оценок</summary>

ничего не говорит о калибровке; на сильном дисбалансе «завышает» качество.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- Log loss = MLE для Бернулли; оптимальная константа $p^*=\bar y$; это *собственное* правило — минимум при истинной вероятности.
- Градиент log loss по логиту: $\sigma(z)-y$.
- Precision $=\frac{\text{TPR}\,\pi}{\text{TPR}\,\pi+\text{FPR}(1-\pi)}$ зависит от доли класса $\pi$, ROC — нет.
- Gini $=2\,\text{AUC}-1$. MCC использует все 4 клетки матрицы ошибок.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, log_loss)

X, y = make_classification(n_samples=5000, weights=[0.9, 0.1], random_state=0)   # 10% положительных
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
proba = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

for thr in (0.5, 0.3):
    pred = (proba > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"thr={thr}: TN={tn} FP={fp} FN={fn} TP={tp} | acc={accuracy_score(y_te, pred):.3f} "
          f"P={precision_score(y_te, pred):.3f} R={recall_score(y_te, pred):.3f} F1={f1_score(y_te, pred):.3f}")
print(f"ROC-AUC={roc_auc_score(y_te, proba):.3f}  PR-AUC={average_precision_score(y_te, proba):.3f}  "
      f"LogLoss={log_loss(y_te, proba):.3f}  | accuracy константы '0' = {1 - y_te.mean():.3f}")
```

Результат запуска:

```text
thr=0.5: TN=1116 FP=4 FN=21 TP=109 | acc=0.980 P=0.965 R=0.838 F1=0.897
thr=0.3: TN=1100 FP=20 FN=20 TP=110 | acc=0.968 P=0.846 R=0.846 F1=0.846
ROC-AUC=0.972  PR-AUC=0.929  LogLoss=0.081  | accuracy константы '0' = 0.896
```

```python
# AUC = доля правильно упорядоченных пар (положительный выше отрицательного)
pos, neg = proba[y_te == 1], proba[y_te == 0]
print(f"AUC через пары: {(pos[:, None] > neg[None, :]).mean():.4f}   sklearn: {roc_auc_score(y_te, proba):.4f}")
```

Результат запуска:

```text
AUC через пары: 0.9723   sklearn: 0.9723
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Метрики классификации и регрессии](https://education.yandex.ru/handbook/ml/article/metriki-klassifikacii-i-regressii)
- [sklearn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- Google ML Crash Course: [ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Лекция 2_072913.pdf` — «Метрики и функционалы потерь в задаче классификации» (MACHINE LEARNING, неделя 2), с. 1–42.
  Вторая половина лекции (кросс-валидация, подбор гиперпараметров) — в теме [05](../05_validation/README.md).

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Fawcett T. *An introduction to ROC analysis* (2006)
- Lipton Z. et al. *Thresholding Classifiers to Maximize F1 Score* (2014)
- Saito T., Rehmsmeier M. *The Precision-Recall Plot Is More Informative than the ROC Plot…* (2015)
- Учебник ML Яндекса: [Метрики классификации и регрессии](https://education.yandex.ru/handbook/ml/article/metriki-klassifikacii-i-regressii)

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): Лекция 2 (ML): метрики классификации

</details>

---

[← 01. Введение в ML и метрики регрессии](../01_intro_regression_metrics/README.md) · [Оглавление курса](../../README.md) · [03. Метрические алгоритмы и kNN →](../03_knn/README.md)
