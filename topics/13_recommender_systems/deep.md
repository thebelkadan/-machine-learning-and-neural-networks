# 13. Рекомендательные системы — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

Обозначения: $R \in \mathbb{R}^{m\times n}$ — матрица взаимодействий ($m$ пользователей, $n$ объектов), $\Omega$ — множество
известных пар $(u, i)$, $r_{ui}$ — оценка, $\hat r_{ui}$ — прогноз.

---

## 1. Соседский метод (user-based kNN)

Сходство пользователей — косинус по **общим** известным координатам:

$$
\cos(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui}r_{vi}}{\sqrt{\sum_{i\in I_{uv}} r_{ui}^2}\sqrt{\sum_{i\in I_{uv}} r_{vi}^2}} .
$$

Пример из лекции: $u_5 = [8, 8, \text{nan}, 3, \text{nan}]$, $u_3 = [9, 8, 4, \text{nan}, 5]$, общие — первые два объекта:
$\cos = \frac{8\cdot9 + 8\cdot8}{\sqrt{8^2 + 8^2}\sqrt{9^2 + 8^2}} = \frac{136}{11.31\cdot12.04} \approx 0.998$.

Прогноз — среднее по $k$ соседям $N(u)$, оценившим объект; лучше — взвешенное и **с учётом смещения пользователей**
(разные люди по-разному «щедры» на оценки):

$$
\hat r_{ui} = \bar r_u + \frac{\sum_{v \in N(u)}\operatorname{sim}(u, v)(r_{vi} - \bar r_v)}{\sum_{v\in N(u)}|\operatorname{sim}(u, v)|} .
$$

Косинус «по общим координатам» опасен: два пользователя с одной общей оценкой имеют косинус 1. Поэтому
используют **центрированный** косинус (= корреляция Пирсона) и штраф за малое пересечение
(shrinkage: $\operatorname{sim}\cdot\frac{|I_{uv}|}{|I_{uv}| + \beta}$). Item-based вариант симметричен (сходство столбцов) и обычно стабильнее:
объектов меньше, и их «профили» меняются медленнее пользовательских.

---

## 2. Базовые смещения

$\hat r_{ui} = \mu + b_u + b_i$; обучение — регуляризованный МНК:

$$
\sum_{(u,i)\in\Omega}(r_{ui} - \mu - b_u - b_i)^2 + \lambda\Big(\sum_u b_u^2 + \sum_i b_i^2\Big) \to \min .
$$

При фиксированных $b_u$ оптимум по $b_i$ (производная = 0): $b_i = \frac{\sum_{u \in U_i}(r_{ui} - \mu - b_u)}{|U_i| + \lambda}$ — среднее отклонение,
**стянутое к нулю** для объектов с малым числом оценок. Это сильный бейзлайн: в ДЗ нейросеть NCF дала RMSE на уровне
«среднего рейтинга фильма» (1.92).

---

## 3. Матричная факторизация

$$
\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i, \qquad
\sum_{(u,i)\in\Omega}\big(r_{ui} - \hat r_{ui}\big)^2 + \lambda\big(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2\big) \to \min .
$$

$p_u, q_i \in \mathbb{R}^r$ — **эмбеддинги** (латентные факторы); скалярное произведение — «насколько вкусы
пользователя совпадают со свойствами объекта».

**SGD** (Funk SVD): для известной пары с ошибкой $e_{ui} = r_{ui} - \hat r_{ui}$:

$$
b_u \leftarrow b_u + \eta(e_{ui} - \lambda b_u),\quad
p_u \leftarrow p_u + \eta(e_{ui}q_i - \lambda p_u),\quad
q_i \leftarrow q_i + \eta(e_{ui}p_u - \lambda q_i) .
$$

**Почему не обычный SVD.** Теорема Эккарта–Янга: лучшее приближение ранга $r$ по норме Фробениуса — усечённый
SVD, $\min_{\operatorname{rank}B = r}\|A - B\|_F = \sqrt{\sigma_{r+1}^2 + \dots}$. Но она требует **полностью известной** матрицы; у нас известно < 1%
элементов. Заполнить пропуски нулями — значит учить модель, что «не смотрел» = «оценка 0». Поэтому оптимизируют
ошибку только на $\Omega$ (с регуляризацией), а это уже невыпуклая задача без формулы.

### ALS

Задача невыпукла по $(P, Q)$ вместе, но **при фиксированном $Q$ — это гребневая регрессия по каждому $p_u$**:

$$
\nabla_{p_u}\Big[\sum_{i \in I_u}(r_{ui} - p_u^\top q_i)^2 + \lambda\|p_u\|^2\Big] = -2\sum_{i\in I_u}(r_{ui} - p_u^\top q_i)q_i + 2\lambda p_u = 0
\;\Rightarrow\;
p_u = \Big(\sum_{i\in I_u}q_iq_i^\top + \lambda I\Big)^{-1}\sum_{i\in I_u}r_{ui}q_i .
$$

Аналогично $q_i = \big(\sum_{u\in U_i}p_up_u^\top + \lambda I\big)^{-1}\sum_{u\in U_i}r_{ui}p_u$. Чередуем шаги — каждый не увеличивает функционал
(точный минимум по блоку). Шаги по разным пользователям **независимы** → идеально параллелятся (Spark ALS).
(В формуле лекции без $\lambda$ матрица может быть вырожденной — регуляризация нужна.)

### iALS (неявный фидбек)

Предпочтение $x_{ui} = [r_{ui} > 0]$, уверенность $c_{ui} = 1 + \alpha r_{ui}$; сумма — по **всем** парам (пропуск = слабый негатив):

$$
\sum_{u, i}c_{ui}(x_{ui} - p_u^\top q_i)^2 + \lambda\big(\|P\|_F^2 + \|Q\|_F^2\big), \qquad
p_u = (Q^\top C^uQ + \lambda I)^{-1}Q^\top C^u x_u .
$$

Трюк Ху–Корена–Волинского: $Q^\top C^uQ = Q^\top Q + Q^\top(C^u - I)Q$; $Q^\top Q$ общий для всех пользователей, а $C^u - I$ ненулевая
лишь на объектах пользователя ⇒ шаг стоит $O(r^2|I_u| + r^3)$, а не $O(r^2 n)$.

---

## 4. SLIM и EASE

**SLIM:** $\tilde r_{uj} = \sum_i r_{ui}w_{ij}$, т.е. $\tilde R = RW$ — каждый столбец (объект) — неотрицательная разреженная
комбинация других объектов:

$$
\frac12\|R - RW\|_F^2 + \frac{\beta}{2}\|W\|_F^2 + \lambda\|W\|_1 \to \min_W, \qquad w_{jj} = 0,\; w_{ij} \ge 0 .
$$

Распадается на $n$ независимых задач Elastic Net (по столбцам). Ограничение $w_{jj} = 0$ обязательно: иначе тривиальное
решение $W = I$ (объект «объясняет сам себя»). Почему $W$ не делают симметричной: «насколько $i$ полезен для
предсказания $j$» ≠ «насколько $j$ для $i$» (популярный объект хорошо предсказывает многие, но не наоборот).

**EASE** (Steck, 2019) — без L1 и неотрицательности; есть **формула**:

$$
\|R - RW\|_F^2 + \lambda\|W\|_F^2 \to \min_W \quad\text{s.t.}\quad \operatorname{diag}(W) = 0 .
$$

*Вывод.* Лагранжиан $\|R - RW\|_F^2 + \lambda\|W\|_F^2 + 2\gamma^\top\operatorname{diag}(W)$. Производная по $W$:
$-2R^\top(R - RW) + 2\lambda W + 2\operatorname{diag}(\gamma) = 0$ ⇒ $(R^\top R + \lambda I)W = R^\top R - \operatorname{diag}(\gamma)$. Обозначим
$P = (R^\top R + \lambda I)^{-1}$; так как $P(R^\top R + \lambda I - \lambda I) = I - \lambda P$:

$$
W = P\,R^\top R - P\operatorname{diag}(\gamma) = I - \lambda P - P\operatorname{diag}(\gamma) = I - P\operatorname{diag}(\tilde\gamma),\quad \tilde\gamma = \lambda\mathbf{1} + \gamma .
$$

Условие $w_{jj} = 0$: $1 - P_{jj}\tilde\gamma_j = 0$ ⇒ $\tilde\gamma_j = 1/P_{jj}$. Итого

$$
W_{ij} = \begin{cases} 0, & i = j,\\ -\dfrac{P_{ij}}{P_{jj}}, & i \ne j .\end{cases}
$$

Одна обратная матрица $n\times n$ ($O(n^3)$) — отсюда ограничение размера каталога (оставляют популярные объекты).
Связь со статистикой: $-P_{ij}/P_{jj}$ — коэффициенты регрессии объекта $j$ на все остальные (через матрицу
точности), т.е. EASE — это $n$ Ridge-регрессий, решённых разом. Рекомендации: $\hat r_u = r_uW$, исключая уже
просмотренное.

---

## 5. Функции потерь для ранжирования

**BPR** (Rendle et al., 2009). Для пользователя $u$, позитивного $i$ и негативного (ненаблюдаемого) $j$ моделируем
$P(i \succ_u j) = \sigma(\hat x_{ui} - \hat x_{uj})$. MAP-оценка с гауссовским априорным распределением параметров:

$$
\max_\Theta\sum_{(u,i,j)}\ln\sigma(\hat x_{ui} - \hat x_{uj}) - \lambda\|\Theta\|^2 ,
$$

т.е. $L_{\text{BPR}} = -\ln\sigma(\text{sim}(u, i) - \text{sim}(u, j))$. Это гладкая оценка снизу доли правильно упорядоченных пар,
т.е. **AUC** пользователя (тема [02](../02_classification_metrics/deep.md#1-auc-roc--вероятность-правильно-упорядоченной-пары)).

**Попарные потери** от отступа $M = a(x_j) - a(x_i)$ для пары $i \prec j$: $(1 - M)_+$ (RankSVM), $e^{-M}$ (RankBoost),
$\ln(1 + e^{-M})$ (RankNet) — гладкие мажоранты индикатора $[M < 0]$ (как суррогаты в классификации).

**RankNet → LambdaRank.** Градиент RankNet по скору документа $i$ в паре $(i, j)$ с $y_i > y_j$:
$\lambda_{ij} = -\frac{\partial L}{\partial s_i} = \sigma(s_j - s_i)$ — все пары равноценны. LambdaRank домножает его на изменение метрики
при перестановке документов:

$$
\lambda_{ij} = \sigma(s_j - s_i)\,\big|\Delta\text{NDCG}_{ij}\big| ,
$$

так что ошибки в верхушке списка (где NDCG чувствителен) весят больше. LambdaMART — градиентный бустинг с этими
«лямбда-градиентами» (`LGBMRanker(objective="lambdarank")`). Функции, для которой это точный градиент, может не
существовать — но эмпирически это прямо оптимизирует NDCG.

**Triplet loss:** $\max(0, \text{margin} + \text{sim}(u, j) - \text{sim}(u, i))$ (на слайде — с обратным знаком sim/расстояния).
**InfoNCE / sampled softmax:** $-\ln\frac{e^{s(u,i)/\tau}}{e^{s(u,i)/\tau} + \sum_{j\in\text{neg}}e^{s(u,j)/\tau}}$ (тема [19](../19_contrastive_embeddings/deep.md)).
При in-batch негативах популярные объекты попадают в негативы чаще — их скоры несправедливо занижаются;
**logQ-коррекция**: $s(u, j) \leftarrow s(u, j) - \ln Q(j)$, где $Q(j)$ — вероятность объекта оказаться негативом.

**DSSM:** $P(D \mid Q) = \frac{\exp(\gamma R(Q, D))}{\sum_{D'}\exp(\gamma R(Q, D'))}$, $R$ — косинус эмбеддингов башен; знаменатель по всем документам
оценивают по сэмплу негативов; минимизируют $-\ln\prod P(D^+ \mid Q)$.

---

## 6. Метрики ранжирования

Для пользователя: ранжированный список, $\text{rel}_k \in \lbrace 0, 1\rbrace$ (или градации), $R$ — число релевантных.

$$
P@K = \frac{1}{K}\sum_{k=1}^{K}\text{rel}_k,\qquad R@K = \frac{1}{R}\sum_{k=1}^{K}\text{rel}_k,\qquad
\text{AP@K} = \frac{1}{\min(R, K)}\sum_{k=1}^{K}P@k\cdot\text{rel}_k,
$$

$$
\text{DCG@K} = \sum_{k=1}^{K}\frac{2^{\text{rel}_k} - 1}{\log_2(k + 1)},\qquad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}},\qquad
\text{MRR} = \frac{1}{|U|}\sum_u\frac{1}{\text{rank}_u},\qquad \text{HR@K} = \frac{1}{|U|}\sum_u[\text{попадание в топ-}K] .
$$

(Для бинарной релевантности $2^{\text{rel}} - 1 = \text{rel}$.) MAP — среднее AP по пользователям.

**Пример.** Релевантны позиции 1 и 3 из топ-5, всего релевантных $R = 2$:
$P@5 = 0.4$; $R@5 = 1$; $\text{AP} = \frac12(1 + \frac23) \approx 0.833$; $\text{DCG} = 1 + \frac{1}{\log_2 4} = 1.5$, $\text{IDCG} = 1 + \frac{1}{\log_2 3} \approx 1.631$,
$\text{NDCG} \approx 0.920$; $\text{MRR} = 1$.

**Почему логарифм в знаменателе.** Модель внимания: вероятность, что пользователь дойдёт до позиции $k$, падает с
номером, но медленно; $\frac{1}{\log_2(k+1)}$ — «мягкий» дисконт (у MRR — жёсткий $\frac1k$). Нормировка на IDCG делает метрику
сравнимой между пользователями с разным числом релевантных.

**Beyond accuracy:**
$\text{ILD}(L) = \frac{2}{|L|(|L| - 1)}\sum_{i<j\in L}d(i, j)$ (разнообразие списка),
$\text{CovCatalog} = \frac{|\bigcup_u \hat\pi_K(u)|}{|I|}$, $\text{Novelty}(u) = \frac{1}{K}\sum_{i\in\hat\pi_K(u)} -\log_2 p(i)$ ($p(i)$ — популярность),
**MMR**-переранжирование: $\lambda\,\text{rel}(u, i) - (1 - \lambda)\max_{j\in S}\text{sim}(i, j)$ — жадно добавляем объект, балансируя релевантность
и непохожесть на уже выбранные $S$.

---

## 7. Смещения в данных и их коррекция

**MNAR и exposure bias.** Наблюдаем $Y_{ui} = R_{ui}\cdot O_{ui}$, где $O_{ui}$ — был ли объект показан. Отсутствие взаимодействия —
это либо «не понравилось», либо «не видел». Вероятность показа $p_{ui} = P(O_{ui} = 1)$ зависит от популярности и
прошлой рекомендательной системы.

**IPS-оценка несмещённа.** Идеальная потеря — по всем парам $L = \frac{1}{|U||I|}\sum_{u,i}\ell(R_{ui}, \hat R_{ui})$. Оценка по
наблюдаемым с весами $1/p_{ui}$:

$$
\hat L_{\text{IPS}} = \frac{1}{|U||I|}\sum_{u,i}\frac{O_{ui}\,\ell(R_{ui}, \hat R_{ui})}{p_{ui}},\qquad
\mathbb{E}_O\hat L_{\text{IPS}} = \frac{1}{|U||I|}\sum_{u,i}\frac{\mathbb{E}[O_{ui}]}{p_{ui}}\ell(\cdot) = L .
$$

Цена — дисперсия при малых $p_{ui}$ (клиппинг $\max(p_{ui}, p_{\min})$, SNIPS — нормировка на $\sum 1/p_{ui}$).

**Сэмплированные метрики несостоятельны** (Krichene & Rendle, 2020). В протоколе «1 позитив + 99 случайных
негативов» HR@10 зависит от того, обгоняет ли позитив случайные объекты, — это почти AUC, а не качество верхушки
полного каталога. Можно построить модели $A$, $B$ с $\text{HR@10}_{\text{full}}(A) > \text{HR@10}_{\text{full}}(B)$, но обратным порядком по
сэмплированной метрике **даже в ожидании**. Вывод: считать метрики по полному каталогу.

---

## Ссылки

- Koren Y., Bell R., Volinsky C. *Matrix Factorization Techniques for Recommender Systems* (2009)
- Hu Y., Koren Y., Volinsky C. *Collaborative Filtering for Implicit Feedback Datasets* (iALS, 2008)
- Ning X., Karypis G. *SLIM: Sparse Linear Methods for Top-N Recommender Systems* (2011)
- Steck H. *Embarrassingly Shallow Autoencoders for Sparse Data* (EASE, 2019)
- Rendle S. et al. *BPR: Bayesian Personalized Ranking from Implicit Feedback* (2009)
- Burges C. *From RankNet to LambdaRank to LambdaMART: An Overview* (2010)
- He X. et al. *Neural Collaborative Filtering* (2017); Rendle S. et al. *Neural Collaborative Filtering vs. Matrix Factorization Revisited* (2020)
- Dacrema M.F. et al. *Are We Really Making Much Progress?* (2019)
- Krichene W., Rendle S. *On Sampled Metrics for Item Recommendation* (2020)
- Schnabel T. et al. *Recommendations as Treatments: Debiasing Learning and Evaluation* (IPS, 2016)
- Учебник ML Яндекса: [Рекомендации на основе матричных разложений](https://education.yandex.ru/handbook/ml/article/rekomendacii-na-osnove-matrichnyh-razlozhenij)

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb)
