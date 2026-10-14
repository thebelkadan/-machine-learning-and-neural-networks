# 13. Рекомендательные системы

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** По матрице взаимодействий «пользователь × объект» (явные оценки или неявные клики/просмотры)
предсказываем, что пользователю понравится, и выдаём **ранжированный топ-$K$**. Подходы: популярное (сильный бейзлайн),
**коллаборативная фильтрация** (kNN, матричные разложения ALS/iALS, линейные SLIM/EASE, нейросети NCF),
**контентные** и **гибридные** модели; в проде — двухэтапно: кандидаты + ранжирование бустингом.
Оценка — метрики ранжирования на **временном** сплите.

| Термин | Определение |
|---|---|
| Explicit / implicit feedback | оценки, лайки / просмотры, клики, покупки |
| Холодный старт | нет истории у нового пользователя или объекта |
| MF / ALS / iALS | $\hat r_{ui}=p_u^\top q_i$; попеременный МНК; версия с весами уверенности для неявного фидбека |
| EASE | $W=-P_{ij}/P_{jj}$, $P=(R^\top R+\lambda I)^{-1}$, $\text{diag}(W)=0$ |
| Pointwise / pairwise / listwise | лосс по объекту / по паре (BPR, RankNet) / по списку (LambdaMART) |
| P@K, R@K, MAP, NDCG, MRR, HR | метрики качества топа |
| Coverage, diversity, novelty | метрики «за пределами точности» |
| Popularity / exposure bias | популярное доминирует в данных / «не видел» ≠ «не понравилось» |

$$
\text{DCG@K}=\sum_{k=1}^{K}\frac{2^{\text{rel}_k}-1}{\log_2(k+1)},\quad \text{NDCG}=\frac{\text{DCG}}{\text{IDCG}},\qquad
p_u=\Big(\sum_{i\in I_u}q_iq_i^\top+\lambda I\Big)^{-1}\sum_{i\in I_u}r_{ui}q_i,\qquad
L_{\text{BPR}}=-\ln\sigma(\hat x_{ui}-\hat x_{uj})
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>1. Постановка</b></summary>

**Рекомендательная система** — система фильтрации информации, которая выдаёт пользователю (user) $u \in U$
персонализированные рекомендации объектов (items) $i \in I$ сервиса на основе хранимых данных о пользователях и объектах.
Фильмы, товары, песни, статьи, вакансии, люди.

| Зачем пользователю | Зачем сервису |
|---|---|
| контент под потребности | рост целевых метрик: заказы, подписки, просмотры, длина сессии |
| экономия времени на поиск | удовлетворённость пользователей и производителей контента |
| открытие нового контента | экономия на коммуникациях, дополнительный канал продаж |

**Подзадачи:** User2Item (основная), User2Bundle (набор товаров), Users2Item (рекомендация группе), Item2Item
(«с этим товаром обычно берут»), User2User («возможно, вы знакомы»), Item2User (сегмент товара).

**Данные:**
- **взаимодействия** пользователь–объект: **явная** обратная связь (оценки, лайки) и **неявная** (просмотры, клики, покупки,
  время прослушивания) — неявной на порядки больше, но она неоднозначна (клик ≠ понравилось, нет клика ≠ не понравилось);
- **о пользователях**: пол, возраст, дата регистрации, любимые жанры, устройство в момент запроса;
- **об объектах**: жанр, длительность, автор, число лайков и прослушиваний.

**Матрица взаимодействий** $R_{m\times n}$ — сильно разреженная (обычно < 1% заполнено).

| Вид системы | Какие данные | Примеры |
|---|---|---|
| **Коллаборативная фильтрация** | только взаимодействия | метрические (kNN), матричные разложения (SVD, ALS), линейные (SLIM, EASE), нейросетевые (NCF) |
| **Контентные** | признаки пользователей и объектов | «похожие по описанию», DSSM |
| **Гибридные** | всё вместе | LightFM, двухэтапные пайплайны с бустингом |

</details>

<details>
<summary><b>2. Неперсонализированные рекомендации</b></summary>

- **Популярное за период** $\Delta T$: $\text{Popularity}(i) = \frac{\text{interactions}_i^{\Delta T}}{\sum_j \text{interactions}_j^{\Delta T}}$;
- **новое**; популярное в категории; «смотрят сейчас».

> Не стоит недооценивать «самое популярное»: в Netflix есть разделы «Популярное», «Новинки» без персонализации,
> и они релевантны многим. Если пользователь регулярно покупает одно и то же (продукты), «рекомендуй то, что он уже
> покупал» — модель, которую очень сложно побить. Чтобы топ не заняли 20 видов хлеба, товары не сортируют по
> популярности, а **сэмплируют с вероятностью, пропорциональной популярности**.

</details>

<details>
<summary><b>3. Коллаборативная фильтрация: соседский метод</b></summary>

Пример из лекции (оценки фильмов «Бег», «Идиот», «Мимино», «Покровские ворота», «Римские каникулы»):
1. **Метрика близости:** векторы оценок пользователей, косинус **по известным координатам**:
   $\cos(u_5, u_3) = \frac{8\cdot9 + 8\cdot8}{\sqrt{8^2 + 8^2}\sqrt{9^2 + 8^2}} = 0.998$.
2. **$k$ ближайших соседей** пользователя.
3. **Оценка** объекта — среднее оценок соседей: $\hat r_{ui} = \frac{1}{|N(u)|}\sum_{j\in N(u)}r_{ji}$; «Мимино» для $u_5$: $\frac{6 + 4}{2} = 5$.
4. Сортируем объекты по оценке и возвращаем топ-$n$.

| Достоинства | Недостатки |
|---|---|
| простота реализации, интерпретируемость | нужны оценки пользователей |
| низкая вычислительная сложность | нельзя рекомендовать **новому пользователю** |
| легко добавлять пользователей и объекты | нельзя рекомендовать **новый объект** (**холодный старт**) |

Item-based вариант (сходство объектов по столбцам) обычно стабильнее. Улучшения — центрирование оценок,
взвешивание по сходству, штраф за малое пересечение (формулы — см. раздел «1. Соседский метод (user-based kNN)»).

</details>

<details>
<summary><b>4. Матричные разложения</b></summary>

**Базовая модель смещений:** $\hat r_{ui} = \mu + b_u + b_i$ ($\mu$ — средняя оценка, $b_u$ — «щедрость» пользователя, $b_i$ — качество объекта),
подбираем $b_u$, $b_i$ по МНК.

**Латентные факторы:** $\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i$, где $p_u, q_i \in \mathbb{R}^r$ — **эмбеддинги** пользователя и объекта:

$$
\sum_{(u,i)}\big(r_{ui} - \mu - b_u - b_i - p_u^\top q_i\big)^2 + \lambda(\dots) \to \min .
$$

- **SVD и теорема Эккарта–Янга:** лучшее приближение ранга $r$ — усечённый SVD (пример из лекции: матрица $20\times10$
  приближается произведением $20\times2$ на $2\times10$ с ошибкой 3.4%). Но SVD требует полной матрицы — а у нас
  пропуски, поэтому оптимизируют ошибку **только на известных** парах.
- **ALS** (Alternating Least Squares): при фиксированной $Q$ каждый $p_u$ находится **по формуле** (гребневая регрессия), и наоборот;
  чередуем. Отлично параллелится (вывод — см. раздел «3. Матричная факторизация»).
- **iALS** — для неявного фидбека: $x_{ui} = [r_{ui} > 0]$, уверенность $c_{ui} = 1 + \alpha|r_{ui}|$, сумма по **всем** парам с весами $c_{ui}$.
- **SGD** (Funk SVD) — по одной известной паре за шаг.

**Линейные item-item модели: SLIM и EASE**

**SLIM:** $\tilde r_{uj} = \sum_i r_{ui}w_{ij}$ — каждый объект — неотрицательная разреженная комбинация других:
$\frac12\|R - RW\|_F^2 + \frac\beta2\|W\|_F^2 + \lambda\|W\|_1 \to \min$, $w_{jj} = 0$, $w_{ij} \ge 0$; распадается на $n$ независимых задач.

**EASE** — без L1 и неотрицательности, есть **аналитическое решение**: $P = (R^\top R + \lambda I)^{-1}$, $W_{ij} = -P_{ij}/P_{jj}$, $W_{jj} = 0$
(вывод — см. раздел «4. SLIM и EASE»).

| Плюсы EASE | Минусы EASE |
|---|---|
| формула, один гиперпараметр, устойчив к нему | $O(n^3)$ — обращение матрицы $n\times n$ (убирают непопулярные объекты; модификации SANSA, ELSA) |
| $W$ обучается один раз и применяется к любому пользователю, при новых действиях пользователя $W$ не пересчитывается | не учитывает последовательность взаимодействий (SLIST) |
| хорошо работает с редкими объектами; поиск похожих объектов через $W$ | |
| онлайн: latency 50 мс при 200 RPS на каталоге ~10к | |

**Бенчмарки из лекции:** на MovieLens-1M (HR@10) SLIM 0.716 > NeuMF 0.712 > UserKNN 0.671 > TopPopular 0.453; на
MSD (Recall@20) EASE 0.333 > iALS 0.309 > RecVAE 0.276 > Popularity 0.043. Вывод статьи *«Are We Really Making Much
Progress?»*: хорошо настроенные простые модели часто не хуже нейросетевых.

</details>

<details>
<summary><b>5. Нейросетевая коллаборативная фильтрация (NCF / NeuMF)</b></summary>

Пользователь и объект — one-hot векторы → **эмбеддинг-слои** → конкатенация → полносвязные слои → прогноз $\tilde r_{ui}$.
Варианты: **GMF** (поэлементное произведение эмбеддингов с весами — обобщение скалярного произведения MF),
MLP-ветка, NeuMF = GMF + MLP.
- легко добавить признаки пользователей/объектов (частично решает холодный старт);
- можно встроить CNN для картинок и учить end-to-end;
- эмбеддинги — веса первых слоёв.

ДЗ (Keras, 3 и 5 слоёв, RMSE): сеть дала RMSE ≈ 1.91 — **на уровне предсказания средним рейтингом фильма** (1.92);
средний рейтинг пользователя — 2.12, глобальное среднее — 2.05. Нейросеть без настройки не гарантирует выигрыша.

```python
import torch, torch.nn as nn

class NCF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb, self.item_emb = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, u, i):
        return self.mlp(torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)).squeeze(-1)
```

**Поиск по эмбеддингам** в проде — приближённый поиск соседей: Annoy, nmslib, hnswlib, FAISS, ElasticSearch ANN
(бенчмарк [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)); тема [19](../19_contrastive_embeddings/README.md).

</details>

<details>
<summary><b>6. Контент, контекст, DSSM и двухбашенные модели</b></summary>

- **Контент** — редко меняющиеся свойства: описание и категория товара, пол пользователя, режиссёр фильма.
- **Контекст** — внешние/быстро меняющиеся условия: время, геолокация, устройство, баланс, недавний поисковый запрос.
- **Контентный метрический метод:** рекомендуем объекты, похожие по признаковому описанию на понравившиеся.
- **DSSM** (Deep Structured Semantic Model): две «башни» превращают запрос и документ (пользователя и объект) в векторы,
  релевантность — косинус; $P(D \mid Q) = \operatorname{softmax}(\gamma R(Q, D))$ по документам (на практике по сэмплу негативов). Объекты
  можно эмбеддить заранее и искать ANN-поиском → **кандидатогенерация** (YouTube DNN и др.).

</details>

<details>
<summary><b>7. Ранжирование и функции потерь</b></summary>

Задача ранжирования: по отношению порядка $i \prec j$ найти функцию $a(x)$ с $a(x_i) < a(x_j)$.

| Подход | Единица обучения | Примеры |
|---|---|---|
| **Point-wise** | отдельный объект (регрессия/классификация) | MSE, BCE на кликах |
| **Pair-wise** | пара объектов | RankSVM $(1 - M)_+$, RankBoost $e^{-M}$, RankNet $\ln(1 + e^{-M})$, **BPR** |
| **List-wise** | весь список | LambdaRank, LambdaMART, ListNet |

| Функция потерь | Формула |
|---|---|
| Кросс-энтропия | $-\sum_{u,i}\big(r_{ui}\ln\hat p_{ui} + (1 - r_{ui})\ln(1 - \hat p_{ui})\big)$, $\hat p_{ui} = \sigma(y_u^\top y_i)$ |
| Triplet | $\max(0,\; \text{margin} + \text{dist}(u, i^+) - \text{dist}(u, i^-))$ |
| BPR | $-\ln\sigma\big(\text{sim}(u, i^+) - \text{sim}(u, i^-)\big)$ |
| InfoNCE | softmax-кросс-энтропия позитива против набора негативов |

Вывод BPR, лямбды LambdaRank, logQ-коррекция (см. раздел «5. Функции потерь для ранжирования»)

**Стратегии negative sampling:** случайные, популярностные (пропорционально популярности), in-batch, «трудные» негативы.

</details>

<details>
<summary><b>8. Метрики качества</b></summary>

| Метрика | Формула (для пользователя, топ-$K$) | Когда |
|---|---|---|
| **Precision@K** | доля релевантных в топ-$K$: $\frac{1}{K}\sum_{k\le K}\text{rel}_k$ | важно, что именно в топе |
| **Recall@K** | доля найденных релевантных: $\frac{1}{R}\sum_{k\le K}\text{rel}_k$ | покрытие интересов |
| **AP / MAP** | среднее $P@k$ по позициям релевантных; MAP — среднее по пользователям | порядок + несколько релевантных |
| **NDCG@K** | $\frac{\text{DCG}}{\text{IDCG}}$, $\text{DCG} = \sum\frac{2^{\text{rel}_k} - 1}{\log_2(k + 1)}$ | градации релевантности, важна позиция |
| **MRR** | среднее $1/\text{rank}$ первого релевантного | один целевой объект (поиск, Q&A) |
| **HR@K** | доля пользователей с хотя бы одним попаданием | leave-one-out протокол |
| **Coverage** | доля каталога, рекомендованная хоть кому-то; доля пользователей с полным списком | монетизация «длинного хвоста» |
| **Diversity (ILD)** | средняя попарная непохожесть объектов списка | борьба с «пузырём фильтров» |
| **Novelty / Serendipity** | непопулярность / неожиданная, но релевантная рекомендация | UX, удержание |
| **Fairness** | равенство качества/экспозиции для групп пользователей и поставщиков | нормативные требования |

Пример расчёта всех метрик — см. раздел «6. Метрики ранжирования». **Дилемма точность–разнообразие:** чем лучше
система знает вкусы, тем однороднее рекомендации; помогает переранжирование **MMR**.

</details>

<details>
<summary><b>9. Валидация</b></summary>

| Сплит | Утечка | Реалистичность | Вердикт |
|---|---|---|---|
| Random split | высокая | низкая | не использовать |
| **Global temporal split** (общий порог $T^*$) | нет | высокая | промышленный стандарт |
| Per-user temporal (последние $N$ взаимодействий каждого) | низкая | средняя | осторожно |
| Leave-one-out случайный | высокая | низкая | устарел |
| **Leave-last-out** (последнее по времени) | нет | высокая | академический стандарт |
| User split | — | для холодного старта | по задаче |

Взаимодействия — **временные ряды**: при random split модель «знает будущее» (посмотрел «Дюну 2» — значит, видел и первую часть).
Подбор гиперпараметров — на val, тест — один раз. K-fold — только во временной версии (expanding window).

**Подводные камни офлайн-оценки:**
- **Popularity bias:** большинство тестовых взаимодействий — с популярными объектами, TopPop выглядит сильным → оценивать
  по стратам (head / mid-tail / long-tail).
- **Exposure bias / MNAR:** «не взаимодействовал» ≠ «не понравилось» (мог не видеть) → IPS-взвешивание (почему несмещённо — см. раздел «7. Смещения в данных и их коррекция»).
- **Сэмплированные метрики** (1 позитив + 99 негативов) могут **переставить** модели местами — считать по полному каталогу.
- **Утечки в признаках:** популярность объекта и статистики пользователя — только по train-периоду.
- **Офлайн ≠ онлайн:** финальное решение — A/B-тест на бизнес-метриках (CTR, конверсия, retention); остерегаться novelty effect.

Чек-лист: временной сплит; фильтр пользователей с < $K_{\min}$ взаимодействий в train; признаки без утечки; тюнинг только на val;
метрики по стратам; отдельно cold-start; сравнение с TopPop; полное ранжирование.

</details>

<details>
<summary><b>10. Двухэтапная архитектура (семинар на данных МТС Kion)</b></summary>

1. **Кандидатогенерация** — быстрые модели отбирают сотни кандидатов из каталога: EASE, ALS, популярное, ANN по эмбеддингам.
2. **Ранжирование** — «тяжёлая» модель (градиентный бустинг) переупорядочивает кандидатов с богатыми признаками:
   скор кандидатогенератора, признаки пользователя и объекта, контекст.

Семинар (Kion, взаимодействия 13.03–22.08.2021; глобальный временной сплит — последняя неделя в тест):
EASE (`reg_weight=2000`, каталог 4215 объектов) → топ-40 кандидатов → обучение LightGBM **классификатора**
(«посмотрит ли пользователь кандидата») и **LGBMRanker** (lambdarank) на признаках из соревнования ODS → сравнение
по Recall@10/NDCG@10 и SHAP-анализ. Hit Rate одной EASE ≈ 0.25.

```python
import numpy as np
import scipy.sparse as sp

def fit_ease(X, reg_weight=2000.0):
    """X — user×item матрица (scipy.sparse). Возвращает item×item матрицу весов EASE."""
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += reg_weight
    P = np.linalg.inv(G)
    W = -P / np.diag(P)                 # W_ij = -P_ij / P_jj
    W[np.diag_indices_from(W)] = 0.0
    return W

def recommend(user_row, W, k=10):
    scores = user_row @ W               # user_row: вектор взаимодействий пользователя
    scores[user_row > 0] = -np.inf      # не рекомендуем уже просмотренное
    return np.argsort(-scores)[:k]
```

```python
from lightgbm import LGBMRanker
ranker = LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.05)
ranker.fit(X_cand, y_cand, group=group_sizes,          # group: число кандидатов на каждого пользователя
           eval_set=[(X_val, y_val)], eval_group=[val_groups], eval_at=[10])
```

**Данные МТС** (в папке курса): `interactions_mts.csv` (user_id, item_id, last_watch_dt, total_dur, watched_pct),
`items.csv` (жанры, страны, режиссёры, описание), `users.csv` (возраст, доход, пол, наличие детей) — датасет
[MTS Kion / RecTools](https://github.com/MobileTeleSystems/RecTools/tree/main/datasets/KION).

</details>

<details>
<summary><b>11. Практические советы (из лекции)</b></summary>

- **Всегда спрашивайте «зачем»**: «повысить продажи» шкафов → покупки редкие, угадать конкретный шкаф трудно (холодный
  старт). Но цель — чтобы пользователь купил **хоть что-то**: рекомендуем **категорию** + скидку на неё.
- Популярное (в т.ч. по категориям) — сильный бейзлайн; «купить снова» — для регулярных покупок.
- Разнообразие — сэмплирование пропорционально популярности, MMR.

**Инструменты:** [implicit](https://github.com/benfred/implicit) (ALS, BPR), [LightFM](https://github.com/lyst/lightfm),
[RecTools](https://github.com/MobileTeleSystems/RecTools), [RePlay](https://github.com/sberbank-ai-lab/RePlay),
[Surprise](https://github.com/NicolasHug/Surprise), [RecBole](https://github.com/RUCAIBox/RecBole/),
[recommenders](https://github.com/recommenders-team/recommenders), [TorchRec](https://github.com/pytorch/torchrec),
[DeepCTR](https://github.com/shenweichen/DeepCTR), [TensorFlow Recommenders](https://github.com/tensorflow/recommenders).

</details>

**Математика и выводы**

<details>
<summary><b>Введение</b></summary>

Обозначения: $R \in \mathbb{R}^{m\times n}$ — матрица взаимодействий ($m$ пользователей, $n$ объектов), $\Omega$ — множество
известных пар $(u, i)$, $r_{ui}$ — оценка, $\hat r_{ui}$ — прогноз.

</details>

<details>
<summary><b>1. Соседский метод (user-based kNN)</b></summary>

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

</details>

<details>
<summary><b>2. Базовые смещения</b></summary>

$\hat r_{ui} = \mu + b_u + b_i$; обучение — регуляризованный МНК:

$$
\sum_{(u,i)\in\Omega}(r_{ui} - \mu - b_u - b_i)^2 + \lambda\Big(\sum_u b_u^2 + \sum_i b_i^2\Big) \to \min .
$$

При фиксированных $b_u$ оптимум по $b_i$ (производная = 0): $b_i = \frac{\sum_{u \in U_i}(r_{ui} - \mu - b_u)}{|U_i| + \lambda}$ — среднее отклонение,
**стянутое к нулю** для объектов с малым числом оценок. Это сильный бейзлайн: в ДЗ нейросеть NCF дала RMSE на уровне
«среднего рейтинга фильма» (1.92).

</details>

<details>
<summary><b>3. Матричная факторизация</b></summary>

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

**ALS**

Задача невыпукла по $(P, Q)$ вместе, но **при фиксированном $Q$ — это гребневая регрессия по каждому $p_u$**:

$$
\nabla_{p_u}\Big[\sum_{i \in I_u}(r_{ui} - p_u^\top q_i)^2 + \lambda\|p_u\|^2\Big] = -2\sum_{i\in I_u}(r_{ui} - p_u^\top q_i)q_i + 2\lambda p_u = 0
\;\Rightarrow\;
p_u = \Big(\sum_{i\in I_u}q_iq_i^\top + \lambda I\Big)^{-1}\sum_{i\in I_u}r_{ui}q_i .
$$

Аналогично $q_i = \big(\sum_{u\in U_i}p_up_u^\top + \lambda I\big)^{-1}\sum_{u\in U_i}r_{ui}p_u$. Чередуем шаги — каждый не увеличивает функционал
(точный минимум по блоку). Шаги по разным пользователям **независимы** → идеально параллелятся (Spark ALS).
(В формуле лекции без $\lambda$ матрица может быть вырожденной — регуляризация нужна.)

**iALS (неявный фидбек)**

Предпочтение $x_{ui} = [r_{ui} > 0]$, уверенность $c_{ui} = 1 + \alpha r_{ui}$; сумма — по **всем** парам (пропуск = слабый негатив):

$$
\sum_{u, i}c_{ui}(x_{ui} - p_u^\top q_i)^2 + \lambda\big(\|P\|_F^2 + \|Q\|_F^2\big), \qquad
p_u = (Q^\top C^uQ + \lambda I)^{-1}Q^\top C^u x_u .
$$

Трюк Ху–Корена–Волинского: $Q^\top C^uQ = Q^\top Q + Q^\top(C^u - I)Q$; $Q^\top Q$ общий для всех пользователей, а $C^u - I$ ненулевая
лишь на объектах пользователя ⇒ шаг стоит $O(r^2|I_u| + r^3)$, а не $O(r^2 n)$.

</details>

<details>
<summary><b>4. SLIM и EASE</b></summary>

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

</details>

<details>
<summary><b>5. Функции потерь для ранжирования</b></summary>

**BPR** (Rendle et al., 2009). Для пользователя $u$, позитивного $i$ и негативного (ненаблюдаемого) $j$ моделируем
$P(i \succ_u j) = \sigma(\hat x_{ui} - \hat x_{uj})$. MAP-оценка с гауссовским априорным распределением параметров:

$$
\max_\Theta\sum_{(u,i,j)}\ln\sigma(\hat x_{ui} - \hat x_{uj}) - \lambda\|\Theta\|^2 ,
$$

т.е. $L_{\text{BPR}} = -\ln\sigma(\text{sim}(u, i) - \text{sim}(u, j))$. Это гладкая оценка снизу доли правильно упорядоченных пар,
т.е. **AUC** пользователя (тема [02](../02_classification_metrics/README.md)).

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
**InfoNCE / sampled softmax:** $-\ln\frac{e^{s(u,i)/\tau}}{e^{s(u,i)/\tau} + \sum_{j\in\text{neg}}e^{s(u,j)/\tau}}$ (тема [19](../19_contrastive_embeddings/README.md)).
При in-batch негативах популярные объекты попадают в негативы чаще — их скоры несправедливо занижаются;
**logQ-коррекция**: $s(u, j) \leftarrow s(u, j) - \ln Q(j)$, где $Q(j)$ — вероятность объекта оказаться негативом.

**DSSM:** $P(D \mid Q) = \frac{\exp(\gamma R(Q, D))}{\sum_{D'}\exp(\gamma R(Q, D'))}$, $R$ — косинус эмбеддингов башен; знаменатель по всем документам
оценивают по сэмплу негативов; минимизируют $-\ln\prod P(D^+ \mid Q)$.

</details>

<details>
<summary><b>6. Метрики ранжирования</b></summary>

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

</details>

<details>
<summary><b>7. Смещения в данных и их коррекция</b></summary>

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

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему нельзя random split?</summary>

взаимодействия — временной ряд; модель «увидит будущее»; нужен temporal / leave-last-out.

</details>

<details>
<summary>Почему SVD «как есть» плох для рекомендаций?</summary>

требует полной матрицы; пропуски ≠ нули; оптимизируют ошибку только на известных парах (ALS/SGD).

</details>

<details>
<summary>Как бороться с холодным стартом?</summary>

популярное, контентные признаки, гибридные модели, онбординг-опросы.

</details>

<details>
<summary>Почему TopPop так хорош в офлайне?</summary>

popularity bias тестовых данных; оценивать по стратам и в A/B.

</details>

<details>
<summary>Почему «1 позитив + 99 негативов» — плохой протокол?</summary>

сэмплированные метрики могут переставить модели (Krichene & Rendle).

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- ALS: при фиксированной $Q$ задача по $p_u$ — Ridge-регрессия с формулой; iALS: $p_u=(Q^\top C^uQ+\lambda I)^{-1}Q^\top C^ux_u$, $Q^\top C^uQ=Q^\top Q+Q^\top(C^u-I)Q$.
- EASE выводится через множители Лагранжа для $\text{diag}(W)=0$; это $n$ Ridge-регрессий разом.
- LambdaRank: градиент RankNet × $|\Delta\text{NDCG}|$ перестановки → оптимизирует верх списка.
- IPS: $\mathbb{E}\big[\sum O_{ui}\ell/p_{ui}\big]=\sum\ell$ — несмещённая оценка при MNAR.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
import numpy as np

# Синтетика: 300 пользователей, 120 объектов; у пользователей и объектов скрытые «жанры»
rng = np.random.default_rng(0)
n_users, n_items, n_genres = 300, 120, 4
user_g = rng.integers(0, n_genres, n_users); item_g = rng.integers(0, n_genres, n_items)
pop = rng.zipf(1.6, n_items).clip(1, 50).astype(float)               # длинный хвост популярности
prob = np.where(user_g[:, None] == item_g[None, :], 0.25, 0.02) * (pop / pop.mean())[None, :].clip(0.2, 3)
R = (rng.random((n_users, n_items)) < prob.clip(0, 0.9)).astype(float)

# «Временной» сплит: по 20% взаимодействий каждого пользователя — в тест
train, test = R.copy(), np.zeros_like(R)
for u in range(n_users):
    items = np.flatnonzero(R[u]); rng.shuffle(items)
    hold = items[: max(1, len(items) // 5)]; train[u, hold] = 0; test[u, hold] = 1

def ndcg_at_k(scores, k=10):
    res = []
    for u in range(n_users):
        s = scores[u].copy(); s[train[u] > 0] = -np.inf           # не рекомендуем уже просмотренное
        top = np.argsort(-s)[:k]; rel = test[u, top]
        dcg = (rel / np.log2(np.arange(2, k + 2))).sum()
        idcg = (1 / np.log2(np.arange(2, min(k, int(test[u].sum())) + 2))).sum()
        res.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(res)

popular = np.tile(train.sum(0), (n_users, 1))                         # бейзлайн: популярное
G = train.T @ train + 50 * np.eye(n_items); P = np.linalg.inv(G)      # EASE
W = -P / np.diag(P); np.fill_diagonal(W, 0)
U, S, Vt = np.linalg.svd(train, full_matrices=False)                  # «PureSVD» ранга 8
svd = (U[:, :8] * S[:8]) @ Vt[:8]
print(f"NDCG@10  Popular={ndcg_at_k(popular):.3f}  PureSVD={ndcg_at_k(svd):.3f}  EASE={ndcg_at_k(train @ W):.3f}")
```

Результат запуска:

```text
NDCG@10  Popular=0.186  PureSVD=0.449  EASE=0.504
```

```python
# Метрики для одного списка: релевантны позиции 1 и 3 из топ-5, всего релевантных 2
rel = np.array([1, 0, 1, 0, 0]); R_total = 2
prec_k = np.cumsum(rel) / np.arange(1, 6)
print("P@5 =", rel.mean(), " R@5 =", rel.sum() / R_total, " AP =", round((prec_k * rel).sum() / R_total, 3),
      " NDCG@5 =", round((rel / np.log2(np.arange(2, 7))).sum() / (1 + 1 / np.log2(3)), 3), " MRR =", 1 / (np.argmax(rel) + 1))
```

Результат запуска:

```text
P@5 = 0.4  R@5 = 1.0  AP = 0.833  NDCG@5 = 0.92  MRR = 1.0
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Рекомендации на основе матричных разложений](https://education.yandex.ru/handbook/ml/article/rekomendacii-na-osnove-matrichnyh-razlozhenij)
- [Курс RecSys ВШЭ](https://github.com/anamarina/RecSys_course) · [Курс МТС/ODS «Ваша первая рекомендательная система»](https://ods.ai/tracks/mts-recsys-df2020) · [Слайды MADE RecSys (Н. Анохин)](https://github.com/aamamaev/MADE_recsys/tree/master/slides)
- К.В. Воронцов, ШАД 2019, лекции 17–18 ([плейлист](https://www.youtube.com/playlist?list=PLJOzdkh8T5krxc4HsHbB8g8f0hu7973fK))
- Статьи: [Are We Really Making Much Progress?](https://arxiv.org/pdf/1907.06902) · [EASE](https://arxiv.org/pdf/1905.03375) · [NCF](https://arxiv.org/pdf/1708.05031) · [Revisiting iALS](https://arxiv.org/pdf/2110.14037) · [On Sampled Metrics (Krichene, Rendle)](https://dl.acm.org/doi/10.1145/3535335) · [Data splitting strategies (Campos et al.)](https://arxiv.org/pdf/2007.13237)
- [Конференция RecSys](https://recsys.acm.org/) · Telegram-канал «ML — это ОК»

</details>

<details>
<summary><b>Источники курса</b></summary>

- `recsys-hse.ipynb` — лекция-ноутбук «Рекомендательные системы» (А. Мамаев): виды систем, kNN, смещения, MF, SVD, ALS, iALS, SLIM, EASE, NeuMF, DSSM, ранжирование, функции потерь, метрики, валидация, инструменты.
- `seminar_recsys_hse.ipynb` — ДЗ: NCF на Keras (3 и 5 слоёв, GMF), сравнение с предсказанием средним.
- `recsys-sirius.ipynb` — лекция «Метрики качества, валидация и функции потерь в задачах ранжирования».
- `recsys-sirius-seminar.ipynb` — семинар на данных МТС Kion: popularity bias, временной сплит, EASE, LightGBM classifier/ranker, SHAP.
- Данные: `interactions_mts.csv`, `items.csv`, `users.csv`.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

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

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Семинары:** [Рекомендательные системы (HSE)](../../materials/seminars/13_recsys_hse.ipynb) · [Семинар по рекомендательным системам (HSE)](../../materials/seminars/13_seminar_recsys_hse.ipynb) · [Рекомендательные системы (Сириус)](../../materials/seminars/13_recsys_sirius.ipynb) · [Семинар по рекомендациям, данные МТС (Сириус)](../../materials/seminars/13_recsys_sirius_seminar.ipynb)

</details>

---

[← 12. Генерация и отбор признаков](../12_feature_engineering_selection/README.md) · [Оглавление курса](../../README.md) · [14. Нейронные сети и backpropagation →](../14_neural_networks_backprop/README.md)
