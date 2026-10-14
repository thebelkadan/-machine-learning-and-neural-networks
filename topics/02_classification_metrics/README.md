# 02. Классификация: метрики и функции потерь

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 01. Метрики регрессии](../01_intro_regression_metrics/README.md) · [03. kNN →](../03_knn/README.md)

---

## 1. Задача классификации

**Классификация** — предсказание класса из **конечного** множества меток. Модель учится находить
границы между классами. Отличие от регрессии — в целевой переменной: конечное множество вместо $\mathbb{R}$.

| Вид | Описание | Как решать |
|---|---|---|
| **Бинарная** | 2 класса: «болен / здоров», «кошка / собака» | один классификатор |
| **Многоклассовая** (multiclass) | $K > 2$ взаимоисключающих класса | One-vs-Rest, One-vs-One, Softmax |
| **Многометочная** (multilabel) | у объекта может быть несколько меток: теги товара, `[1 0 1]` | $K$ независимых бинарных задач (наличие / отсутствие метки) |

### Сведение многоклассовой задачи к бинарным

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

---

## 2. «Чёткая» классификация: модель выдаёт метку

### 2.1. Матрица ошибок (Confusion Matrix)

Ориентация как в **sklearn**: строки — истина, столбцы — предсказание.

|  | $\hat y = 0$ | $\hat y = 1$ |
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

### 2.2. Основные метрики

| Метрика | Формула | Вопрос, на который отвечает | Пример (матрица выше) |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | какая доля ответов верна? | 0.806 |
| **Precision** (PPV) | $\frac{TP}{TP + FP}$ | если модель сказала «1», как часто она права? | 0.258 |
| **Recall** = **TPR** = Sensitivity | $\frac{TP}{TP + FN}$ | какую долю положительных нашли? | 0.501 |
| **FPR** | $\frac{FP}{FP + TN}$ | какую долю отрицательных ошибочно пометили как «1»? | 0.161 |
| **Specificity** = TNR | $\frac{TN}{TN + FP} = 1 - \text{FPR}$ | какую долю отрицательных распознали? | 0.839 |
| **F1** | $\frac{2PR}{P + R} = \frac{2TP}{2TP + FP + FN}$ | баланс P и R | 0.340 |
| **$F_\beta$** | $(1+\beta^2)\frac{P R}{\beta^2 P + R}$ | $\beta > 1$ — важнее Recall, $\beta < 1$ — важнее Precision | $F_2$ = 0.422 |

> ⚠️ **Ловушка accuracy.** Здесь положительных всего 10%. Константа «всегда 0» даёт accuracy
> **0.900** — больше, чем у модели (0.806)! На несбалансированных данных accuracy почти бесполезна:
> смотрите Precision / Recall / F1, balanced accuracy, PR-AUC.

**Почему гармоническое среднее в F1?** Оно штрафует дисбаланс между P и R: при $P = 1$, $R = 0.01$
арифметическое среднее 0.5, а F1 ≈ 0.02. F1 высока, только если **обе** метрики высоки.

### 2.3. Precision–Recall tradeoff

Модель обычно выдаёт оценку, а метку получаем сравнением с **порогом** (threshold) — гиперпараметром.

- **Низкий порог** → почти всех больных находим (высокий Recall), но часто путаем здоровых с больными (низкий Precision).
- **Высокий порог** → находим только «очевидных» больных (высокий Precision), упускаем остальных (низкий Recall).

Порог выбирают под задачу: по максимуму F1 на валидации, под требуемый Recall (медицина, антифрод),
или по стоимостям ошибок: $t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}$ ([вывод](deep.md#5-оптимальный-порог-при-разных-стоимостях-ошибок)).

```python
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_val, proba_val)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_thr = thr[f1[:-1].argmax()]
```

---

## 3. «Нечёткая» классификация: модель выдаёт оценку / вероятность

### 3.1. ROC-кривая и AUC-ROC

**ROC-кривая** (receiver operating characteristic) — график TPR (ось Y) от FPR (ось X) при **всех**
порогах. Каждый порог — точка на кривой: при пороге $+\infty$ получаем $(0, 0)$, при $-\infty$ — $(1, 1)$.

**AUC-ROC** (area under curve) — площадь под ROC-кривой:
- **1** — идеальный классификатор (все положительные выше всех отрицательных);
- **0.5** — случайный;
- **0** — идеально «наоборот» (инвертируйте ответы).

**Смысл AUC:** доля правильно упорядоченных пар (положительный, отрицательный) — вероятность того,
что случайный положительный объект получит оценку выше случайного отрицательного
([доказательство](deep.md#1-auc-roc--вероятность-правильно-упорядоченной-пары)).

Функционал зависит **не от значений оценок, а от их порядка**.

| Плюсы AUC-ROC | Минусы AUC-ROC |
|---|---|
| полезен там, где важен порядок (ранжирование, скоринг) | «завышает» качество на сильном дисбалансе (много TN) |
| не зависит от пропорций классов | оценивает не конкретный классификатор (порог), а «регрессию»-скоринг |
| опирается на порядок, а не на значения | ничего не говорит о калибровке вероятностей |
| можно оценивать отдельные признаки (AUC признака как скора) | сложно объяснить заказчику |

**Gini** $= 2\,\text{AUC} - 1$ — то же самое в шкале $[-1, 1]$, принято в кредитном скоринге.

### 3.2. PR-кривая и Average Precision

PR-кривая — Precision от Recall при всех порогах. Площадь под ней — **PR-AUC / Average Precision (AP)**.
Для редкого положительного класса PR-кривая информативнее ROC: случайный классификатор
даёт AP ≈ доле положительных (а не 0.5), а Precision сразу «чувствует» рост FP.

### 3.3. Log Loss (бинарная кросс-энтропия)

Модель при обучении минимизирует не метрику, а **функцию потерь**. Для вероятностной классификации
основной loss — **Log Loss** (logistic loss, cross-entropy):

$$
\text{LogLoss} = -\frac{1}{\ell}\sum_{i=1}^{\ell}\Big( y_i \ln p_i + (1 - y_i)\ln(1 - p_i) \Big),
$$

где $p_i$ — предсказанная вероятность класса 1. Раздельная запись: штраф $-\ln p_i$ для $y_i = 1$ и
$-\ln(1 - p_i)$ для $y_i = 0$.

- **Нельзя уверенно ошибаться**: $p = 0.001$ при $y = 1$ даёт штраф $\approx 6.9$, при $p \to 0$ — бесконечность.
- Оптимальная константа — **доля класса 1**: $p^* = \bar y$ ([вывод](deep.md#32-оптимальная-константа)).
- Log loss = MLE для распределения Бернулли и **собственное правило**: минимум ожидаемого
  штрафа — при истинной вероятности. Поэтому модели на log loss выдают осмысленные вероятности.
- Многоклассовый вариант: $-\frac{1}{\ell}\sum_i\sum_k y_{ik}\ln p_{ik}$ (softmax + cross-entropy).

**Выводы лекции:** Log Loss ~ **вероятность**; ROC-AUC ~ **порядок**.

---

## 4. Метрики многоклассовой и многометочной классификации

Метрики P, R, F1 считаются **для каждого класса** (один против всех), затем усредняются:

| Усреднение | Как | Особенность |
|---|---|---|
| `macro` | среднее по классам | все классы равноправны — чувствительно к редким классам |
| `weighted` | среднее с весами = доли классов | близко к accuracy на дисбалансе |
| `micro` | суммируем TP, FP, FN по всем классам, потом формула | для multiclass micro-F1 = accuracy ([доказательство](deep.md#6-многоклассовые-усреднения)) |

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

---

## 5. Как выбрать метрику

| Ситуация | Что смотреть |
|---|---|
| Классы сбалансированы, ошибки равноценны | Accuracy, F1 |
| Сильный дисбаланс, важен редкий класс | Precision, Recall, F1, PR-AUC, balanced accuracy, MCC |
| Пропуск положительного очень дорог (болезнь, мошенничество) | Recall (при ограничении на Precision), $F_2$ |
| Ложная тревога дорога (спам-фильтр, блокировка карты) | Precision, $F_{0.5}$ |
| Нужно ранжирование / скоринг, порог выберут потом | ROC-AUC (Gini), PR-AUC |
| Нужны честные вероятности (расчёт ожидаемой прибыли, ставки) | Log Loss, Brier score + калибровка (тема [04](../04_linear_models/README.md)) |
| Многоклассовая с дисбалансом | macro-F1 |

---

## 6. Код: полный пример

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

---

## 7. Типичные ошибки

- Отчитываться accuracy на несбалансированных данных.
- Считать ROC-AUC по **меткам** (`predict`) вместо оценок (`predict_proba` / `decision_function`).
- Оставлять порог 0.5 по умолчанию, не подбирая его под задачу.
- Подбирать порог на тесте (это утечка — подбирать на валидации).
- Интерпретировать `predict_proba` как вероятность без проверки калибровки (у SVM, RF, бустинга она часто кривая).
- Путать ориентацию матрицы ошибок (в sklearn строки — истина).

---

## Что почитать / посмотреть

- 📘 Учебник ML Яндекса: [Метрики классификации и регрессии](https://education.yandex.ru/handbook/ml/article/metriki-klassifikacii-i-regressii)
- 📄 [sklearn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- 📄 Google ML Crash Course: [ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

## Источники курса

- `Лекция 2_072913.pdf` — «Метрики и функционалы потерь в задаче классификации» (MACHINE LEARNING, неделя 2), с. 1–42.
  Вторая половина лекции (кросс-валидация, подбор гиперпараметров) — в теме [05](../05_validation/README.md).


<!-- materials:start -->
**Файлы курса в репозитории:**

- Лекция 2 (ML): метрики классификации (лекция курса, файл в репозиторий не включён)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [03. Метрические алгоритмы, kNN →](../03_knn/README.md)
