# 19. Contrastive learning, эмбеддинги, векторный поиск, интерпретация

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Когда классы заранее неизвестны (лица, товары, места — **open set**), учат не классификатор, а **эмбеддер**: свои объекты близко,
чужие далеко (contrastive / **triplet** loss с правильным **выбором троек**, ArcFace, без разметки — **InfoNCE**/SimCLR). Поиск — ближайшие
соседи (до ~1 млн перебор, дальше IVF / **HNSW** / PQ). Высокая метрика не значит, что сеть решает задачу: **Grad-CAM** показывает, куда она
смотрит, а проверяют гипотезу данными.

| Термин | Определение |
|---|---|
| Open set / one-shot | на тесте новые классы; на класс — один пример |
| L2-нормировка | $\lVert a-b\rVert^2=2-2\cos$; расстояния $\in[0,2]$ |
| Contrastive / triplet | $y d^2+(1-y)\max(0,m-d)^2$ / $\max(0,d_{ap}-d_{an}+m)$ |
| PK-sampler, batch-hard | $P$ классов × $K$ примеров; самый далёкий свой и самый близкий чужой |
| InfoNCE (SimCLR) | кросс-энтропия «своя пара среди кандидатов батча», температура $\tau$ |
| Recall@k | доля запросов, у которых среди $k$ соседей есть свой класс |
| IVF / HNSW / PQ | ячейки k-means / граф соседей / сжатие кодами центров |
| Grad-CAM | $\operatorname{ReLU}\big(\sum_k\alpha_kA^k\big)$, $\alpha_k$ — средний градиент по каналу |

$$
L_{\text{triplet}}=\max\big(0,\;d(a,p)-d(a,n)+m\big),\qquad
\ell_{i}=-\ln\frac{e^{\cos(z_i,z_j)/\tau}}{\sum_{k\ne i}e^{\cos(z_i,z_k)/\tau}},\qquad
\alpha_k^c=\frac1Z\sum_{ij}\frac{\partial y^c}{\partial A^k_{ij}}
$$

## Конспект

Разделы раскрываются по клику.

**Часть A. Учим близость, а не класс**

<details>
<summary><b>1. Закрытое и открытое множество классов</b></summary>

До сих пор классы были заданы заранее (1000 ImageNet, 19 Cityscapes, 80 COCO), и последний слой сети — ровно они.

| Closed set | Open set |
|---|---|
| список классов известен и не меняется | на применении встречаются **новые** классы |
| много примеров на класс | на класс бывает **один** пример (one-shot) |
| ответ — распределение по известным классам | нужен ответ «ни один из известных» |
| ImageNet, Cityscapes, COCO | лица, товары маркетплейса, места на карте, поиск похожих картинок |

**Пример — турникет:** в компании 1000 человек, каждую неделю кто-то приходит и уходит. Классификатор пришлось бы переобучать еженедельно,
и он не умеет сказать «этого человека я не знаю». **Эмбеддер** обучается один раз (на других людях), узнавание — сравнение векторов, база
сотрудников живёт отдельно от сети.

**Идея метрического обучения:** учить не границу, а **пространство** — объекты одного класса близко, разных — далеко. Классов в сети нет:
есть функция «картинка → вектор» и обычная метрика.

**Нормировка:** почти все эмбеддеры выдают L2-нормированный вектор; тогда $\|a - b\|^2 = 2 - 2\cos(a, b)$ — порядок соседей по евклиду и косинусу
совпадает (вывод и следствия — см. раздел «1. Геометрия L2-нормированных эмбеддингов»).

**Бейзлайн — признаки классификатора:** предобученный ResNet без последнего слоя (работает по той же причине, что transfer learning).
Но классификатору нужна разделимость, а не компактность; расстояния он не оптимизировал; признаки заточены под ImageNet (различать двух
похожих людей его не просили). Семинар (Stanford Cars, обучение на 98 моделях, тест на других 98): сырые пиксели Recall@1 0.039, случайная
ResNet18 0.042, **замороженный ImageNet 0.195** — соседи того же цвета, ракурса, типа кузова, но не той модели.

</details>

<details>
<summary><b>2. Сиамская сеть, contrastive и triplet loss</b></summary>

**Сиамская сеть** — одна и та же сеть, прогоняемая по очереди на двух входах («сиамская» — схема обучения, не архитектура). Пример — пара
«свои/чужие»; из $N$ объектов пар ~$N^2$.

| Loss | Формула | Смысл |
|---|---|---|
| **Contrastive** | $L = y\,d^2 + (1-y)\max(0, m - d)^2$ | своих тянем всегда, чужих отталкиваем, пока ближе margin $m$; без margin — тривиальное решение |
| **Triplet** (FaceNet, 2015) | $L = \max\big(0, d(a,p) - d(a,n) + m\big)$ | относительное требование: чужой дальше своего на $m$ |

Важно: Margin измеряется в единицах расстояния: на единичной сфере $d \le 2$, учебниковое $m = 1$ (из работы без нормировки) приводит к
схлопыванию (семинар: Recall@1 0.167 — хуже заморозки; $m = 2$ — 0.239).

</details>

<details>
<summary><b>3. Выбор троек — главная проблема</b></summary>

Троек ~$N^3$; случайная почти всегда **easy** (штраф 0, градиент 0) → плато. **Выбор троек влияет сильнее архитектуры.**

- **PK-sampler:** $P$ классов × $K$ примеров (3 × 4 = 12 фото → 288 троек). Случайный батч из 64 при 98 классах — у якоря в среднем 0.56 своих: выбирать не из чего.
- **Batch-all** — все тройки (усреднение по тройкам с loss > 0); **batch-hard** — для каждого якоря самый далёкий свой и самый близкий чужой;
  **semi-hard** — чужой дальше своего, но меньше чем на margin.

Семинар (3 сида): случайные 0.213 < semi-hard 0.220 < batch-hard 0.228; доля активных троек у случайных падает 0.84 → 0.59, у hard — ~1.0;
batch-hard почти не зависит от margin. Честно подобранный contrastive — на уровне batch-hard (*A Metric Learning Reality Check*: прогресс
функций потерь преувеличен).

</details>

<details>
<summary><b>4. Распознавание лиц — пайплайн</b></summary>

Детекция лица → **выравнивание по опорным точкам** (чаще всего забывают, а даёт больше всего качества) → эмбеддер → сравнение с базой
→ порог. Классический набор: детектор MTCNN / RetinaFace, эмбеддер **ArcFace** на ResNet-100, библиотека InsightFace. На практике лица учат
не тройками, а **классификацией с угловым margin** (формула ArcFace — см. раздел «3. Классификация с угловым margin (ArcFace)»).

Важно: TAR@FAR=0.001 в 99% **не** годится для турникета на 20 000 человек: при сравнении со всей базой почти каждый посторонний найдёт «двойника»
($1 - (1 - 0.001)^{20000} \approx 1$) (расчёт — см. раздел «5. Верификация и порог: почему 99% на валидации мало»).

**Где ещё:** Visual Place Recognition («где снят кадр», пары из GPS бесплатно), поиск похожих товаров, копий и дубликатов, переидентификация
людей на камерах, поиск трека по отрывку.

</details>

<details>
<summary><b>5. Обучение без разметки: InfoNCE</b></summary>

**SimCLR:** пара = две аугментации одной картинки; **InfoNCE (NT-Xent)**: кросс-энтропия, где «классы» — кандидаты батча, своя пара — правильный
ответ; $\tau$ — температура (малая сильнее выделяет самых похожих кандидатов) (формула и градиент — см. раздел «4. InfoNCE / NT-Xent»).
**Аугментации — это постановка задачи**: решает пара «случайный кроп + искажение цвета» (без цвета сеть решает задачу по гистограмме).

| Метод | Идея |
|---|---|
| **MoCo** | негативы копятся в очереди от прошлых батчей — запас без гигантского батча |
| **BYOL, SimSiam** | негативов нет: одна ветвь предсказывает выход другой; от коллапса спасает асимметрия |
| **DINO** | самодистилляция на ViT; карты внимания сами выделяют объект без масок |
| **CLIP** (тема [26](../26_llm_rag_agents/README.md)) | пара «картинка ↔ подпись», тот же InfoNCE |

Зачем: размеченных данных мало, неразмеченных — сколько угодно; предобучение без разметки → бэкбон, дообучаемый на крошечной выборке.
Одна идея — «сближай своё, отталкивай чужое» — покрывает лица, места, self-supervised предобучение и связку картинок с текстом.

</details>

<details>
<summary><b>6. Поиск по векторам</b></summary>

Один раз прогоняем базу → матрица $N\times d$; на запрос — вектор и $k$ ближайших. **До ~миллиона векторов честный перебор — правильный выбор.**
Важно: Сменили модель — пересчитайте всю базу (векторы разных моделей несравнимы).

| Подход | Идея | Ручки |
|---|---|---|
| **IVF** | k-means делит пространство на ячейки; ищем в `nprobe` ближайших ячейках | `nlist`, `nprobe` (больше → выше recall, медленнее) |
| **HNSW** | многоуровневый граф соседей, жадный спуск | `M`, `efConstruction`, `efSearch` |
| **PQ** | вектор режем на $M$ частей, каждую — номером ближайшего из 256 центров (1 байт): 2048 → 64 байта | `M`, число центров |

Обе идеи ANN — не сравнивать запрос со всей базой; обе могут пропустить ответ — плата за скорость, а не баг. Семинар (свой IVF на эмбеддингах
машин): recall@10 ≥ 0.9 только при просмотре 31% базы; на «учебных» гауссовых кластерах при 3% просмотра — 0.98. **Цифры индексов
меряйте на своих векторах.**

**Векторные базы:** FAISS (библиотека: IVF, HNSW, PQ и комбинации — стандарт), Qdrant, Milvus, Weaviate (системы), pgvector (PostgreSQL).
Проверять: Recall@k относительно точного поиска, задержку и пропускную способность, память и время построения, работу с **фильтрами**
(категория, цена, права доступа). Это же — основа RAG (тема [26](../26_llm_rag_agents/README.md)).

```python
import torch, torch.nn.functional as F

def batch_hard_triplet_loss(z, y, margin=0.2):
    """z: [B, D] эмбеддинги, y: [B] метки. Квадраты евклидовых расстояний, batch-hard."""
    z = F.normalize(z, dim=1)
    D = (2 - 2 * z @ z.T).clamp(min=0)                          # ||a-b||² = 2 - 2cos
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos, neg = same & ~eye, ~same
    d_ap = D.masked_fill(~pos, float("-inf")).max(1).values    # самый далёкий свой
    d_an = D.masked_fill(~neg, float("inf")).min(1).values     # самый близкий чужой
    valid = pos.any(1) & neg.any(1)
    if not valid.any():
        return z.sum() * 0.0
    return F.relu(d_ap[valid] - d_an[valid] + margin).mean()

def info_nce(z1, z2, tau=0.1):
    """SimCLR NT-Xent: z1[i] и z2[i] — два вида одной картинки."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))                           # себя исключаем
    n = len(z1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)

# HNSW (ДЗ 3)
import hnswlib, numpy as np
index = hnswlib.Index(space="cosine", dim=512)
index.init_index(max_elements=len(X), M=16, ef_construction=100, random_seed=0)
index.add_items(X, np.arange(len(X)))
index.set_ef(64)
ids, dists = index.knn_query(Q, k=10)
```

</details>

**Часть B. Что выучила сеть: интерпретация**

<details>
<summary><b>7. Зачем заглядывать внутрь</b></summary>

**Отладка** (метрика высокая, в проде не работает — выучила не то), **доверие** (врачу нужно основание), **требования** (медицина, кредиты,
найм — объяснение по закону). Классические провалы: **хаски vs волк** — сеть смотрела на снег (LIME, Ribeiro et al., 2016); **рак кожи** — модель
нашла линейку рядом с подозрительной родинкой; **ковид по рентгену** — выучили разметку и тип аппарата больницы (обзор 2021: ни одной
пригодной модели из 62). Везде: высокая метрика на валидации, бесполезная модель в жизни — ошибка данных, сеть нашла самый простой признак.

**Три вопроса — три метода (не путать!):**
1. Что вообще умеет находить сеть? → фильтры и карты признаков (про сеть целиком);
2. **Куда сеть смотрела на этой картинке?** → saliency, occlusion, Grad-CAM (90% отладки);
3. Почему она выбрала этот ответ? → честного ответа нет; все методы отвечают на 2-й, а слышим мы ответ на 3-й — источник неправильных выводов.

</details>

<details>
<summary><b>8. Методы</b></summary>

| Метод | Как | Плюсы | Минусы |
|---|---|---|---|
| Карты признаков | нарисовать активации промежуточных слоёв | иерархия признаков | не отвечает «почему»; яркие края — след паддинга |
| **Saliency** | $\lvert\partial(\text{оценка класса})/\partial(\text{пиксель})\rvert$ | 1 обратная прогонка | шумная; насыщение |
| SmoothGrad | среднее saliency по зашумлённым копиям | меньше шума | десятки прогонов; плохо проходит sanity check |
| **Occlusion** | закрыть кусок и смотреть падение вероятности | любая модель, чёрный ящик, понятно | сотни прогонов |
| **Grad-CAM** | веса каналов последнего свёрточного слоя = средний градиент; ReLU(взвешенная сумма карт) | 1 обратная прогонка, гладкая осмысленная карта | разрешение последнего слоя (7×7) — область, а не контур |
| Integrated Gradients | интеграл градиента по пути от пустого входа | лечит насыщение | дороже |
| LIME | простая модель вокруг одного примера | модель-агностик | объясняет свою аппроксимацию, а не сеть |
| SHAP | значения Шепли | теоретические свойства | дорогой счёт |

Формулы Grad-CAM, CAM, IG (см. раздел «7. Методы интерпретации»)

```python
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, cls=None):
        self.model.eval(); x = x.requires_grad_(True)
        logits = self.model(x)
        cls = logits.argmax(1) if cls is None else torch.as_tensor(cls).expand(len(x))
        self.model.zero_grad(); logits.gather(1, cls[:, None]).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)             # α_k = средний градиент по позициям
        cam = torch.relu((w * self.acts).sum(1))                    # ReLU(Σ α_k A^k)
        cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)

# cam = GradCAM(resnet, resnet.layer4)(images)
```

</details>

<details>
<summary><b>9. «Умный Ганс» и ограничения методов</b></summary>

Задача-ловушка (лекция и семинар): яркая метка в углу, которая в обучении стоит ровно у одного класса. Две сети дают 100% на своей валидации,
но на честном тесте сеть с подсказкой — **49.8%** (лекция) / **0.375 — хуже константы 0.625** (семинар, внедорожник vs кабриолет); Grad-CAM:
40.8% внимания в углу, занимающем 4.3% кадра, у честной сети — 7.7%, она смотрит на кузов.

**Чего методы не доказывают:**
- отвечают на «где», а не «почему»: «смотрит на колесо» ≠ «узнала машину по колесу»;
- **sanity checks** (Adebayo et al., 2018): у части методов карта почти не меняется на случайных весах — это свойство архитектуры
  (семинар: корреляция карт обученной и случайной сети — Grad-CAM 0.08, SmoothGrad 0.54);
- красивая карта убеждает сильнее, чем имеет право; разные методы дают разные карты.

**Правило:** карта — повод проверить данные, а не вывод. Проверка — собрать тест, где подсказка не работает. Порядок в семинаре: сначала
провал в таблице точности, потом объяснение картинкой.

</details>

<details>
<summary><b>10. Что делать на практике</b></summary>

**Метрическое обучение:** начать с признаков предобученного бэкбона (бейзлайн); есть классы в разметке — ArcFace; есть только пары —
triplet с batch-hard и PK-батчами; мерить **Recall@1 на классах, которых не было в обучении** (деление выборки по классам!); индекс — до миллиона
перебор, дальше HNSW, при нехватке памяти PQ.
**Интерпретация:** Grad-CAM на десятке ошибок модели — 20 минут работы и самая высокая отдача; смотреть и на **уверенные правильные** ответы
(подсказка видна там); подозрение проверять данными; Важно: не показывать карту заказчику как «объяснение решения».

**Одной строкой:** сеть всегда что-то выучивает — вопрос только в том, то ли это, что вы имели в виду.

</details>

<details>
<summary><b>ДЗ 3: поиск похожих птиц (CUB-200-2011)</b></summary>

200 видов; обучение на 80 видах, валидация на 20, тест — виды 101–200 (не пересекаются). Задания: точный top-k с исключением себя;
Recall@k по видам; маски позитивов/негативов; batch-hard тройки (`-inf` перед max, `+inf` перед min — не умножение на маску!); triplet loss
(квадрат расстояния, $m = 0.2$, среднее по валидным якорям включая нулевые); HNSW через `hnswlib` (`M=16`, `ef_construction=100`); сравнение с
полным перебором по двум метрикам: **совпадение с точным top-10** (точность ANN) и **Recall@1 по видам** (качество задачи) — их нельзя путать.

</details>

**Математика и выводы**

<details>
<summary><b>1. Геометрия L2-нормированных эмбеддингов</b></summary>

Для $\|a\| = \|b\| = 1$:

$$
\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2\langle a, b\rangle = 2 - 2\cos(a, b) .
$$

Следствия: порядок соседей по евклиду и косинусу совпадает (монотонное преобразование); расстояние на сфере $d = \sqrt{2 - 2\cos} \in [0, 2]$
(2 — противоположные векторы, $\sqrt2$ — ортогональные). Без нормировки длина вектора начинает означать «уверенность» и вмешивается в
расстояние. **Margin живёт в единицах расстояния:** на сфере contrastive с $m = 1$ требует лишь «чужие дальше 60°» ($d = 1 \iff \cos = \frac12$) —
семинар: $m = 1$ ухудшил Recall@1 до 0.167 против 0.195 у заморозки, $m = 2$ дал 0.239.

</details>

<details>
<summary><b>2. Contrastive и triplet loss</b></summary>

**Contrastive** (Hadsell, Chopra, LeCun, 2006): для пары с меткой $y \in \lbrace 0, 1\rbrace$ (1 — свои), $d = \|f(x_1) - f(x_2)\|$:

$$
L = y\,d^2 + (1 - y)\max(0, m - d)^2 .
$$

Без второго слагаемого оптимум тривиален — всё в одну точку (коллапс); без margin — чужих можно «разгонять» бесконечно. Слабость —
**абсолютный** порог $m$ на весь датасет.

**Triplet** (FaceNet, Schroff et al., 2015): якорь $a$, свой $p$, чужой $n$:

$$
L = \max\big(0,\; d(a, p) - d(a, n) + m\big) .
$$

Требование **относительное**: чужой дальше своего на $m$. Без $m$ решение $f \equiv \text{const}$ даёт $L = 0$.

**Число троек и выбор.** PK-батч ($P$ классов × $K$ примеров): для каждого из $PK$ якорей $K - 1$ своих и $(P - 1)K$ чужих → $PK(K-1)(P-1)K$ троек
(лекция: $3\times4$ → $12\cdot3\cdot8 = 288$). Случайная тройка почти всегда **easy** ($d(a,n) > d(a,p) + m$): loss и градиент нулевые — семинар: доля активных
случайных троек падает с 0.84 до 0.59 за 30 эпох.
- **batch-hard** (Hermans et al., 2017): $L_a = \max\big(0, \max_{p}d(a,p) - \min_{n}d(a,n) + m\big)$ — самый далёкий свой и самый близкий чужой;
- **semi-hard** (FaceNet): чужой из «кольца» $d(a,p) < d(a,n) < d(a,p) + m$ — не самые трудные (спасает от шумных меток и коллапса в начале).

Семинар: random 0.213 < semi-hard 0.220 < batch-hard 0.228 (средние по 3 сидам); batch-hard почти не зависит от margin (0.230 при
$m = 0.2…1.5$), случайные — разваливаются. Разницу меньше разброса по сидам нельзя объявлять эффектом.

**Матрица расстояний батча одним умножением:** $D_{ij} = \|z_i\|^2 + \|z_j\|^2 - 2\langle z_i, z_j\rangle$; для нормированных $D = 2 - 2ZZ^\top$.

</details>

<details>
<summary><b>3. Классификация с угловым margin (ArcFace)</b></summary>

Лица на практике учат не тройками, а классификацией по identity с нормированными весами и признаками: логит класса $j$ — $s\cos\theta_j$,
$\theta_j$ — угол между эмбеддингом и вектором класса. **ArcFace** (Deng et al., 2019) добавляет к углу правильного класса margin $m$:

$$
L = -\ln\frac{e^{s\cos(\theta_y + m)}}{e^{s\cos(\theta_y + m)} + \sum_{j\ne y}e^{s\cos\theta_j}} .
$$

Эмбеддинг должен быть ближе к своему центру **на угол $m$**, чем требуется для классификации — классы компактнее, зазоры больше.
Использует все классы сразу (не нужен майнинг троек); на тесте классификационный слой выбрасывают, остаётся эмбеддер.

</details>

<details>
<summary><b>4. InfoNCE / NT-Xent</b></summary>

SimCLR: $N$ картинок → $2N$ аугментированных видов; для якоря $i$ с парой $j$:

$$
\ell_{i,j} = -\ln\frac{\exp(\operatorname{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k \ne i]}\exp(\operatorname{sim}(z_i, z_k)/\tau)},\qquad \operatorname{sim} = \cos .
$$

Это **кросс-энтропия**, где «классы» — кандидаты батча, а правильный — своя пара. Loss батча — среднее по всем $2N$ якорям.

**Градиент и температура.** По $s_k = \operatorname{sim}(z_i, z_k)$: $\frac{\partial\ell}{\partial s_k} = \frac{1}{\tau}\big(p_k - [k = j]\big)$, $p_k = \operatorname{softmax}(s/\tau)_k$. Негатив
отталкивается с весом $p_k$ — **чем он похожее, тем сильнее** (встроенный hard-negative mining). Малое $\tau$ делает softmax острее —
отталкиваются почти только самые похожие негативы (в том числе «ложные» негативы того же класса); большое $\tau$ — все поровну.

**Связь со взаимной информацией** (Oord et al., 2018): $I(x; y) \ge \ln N - \mathcal{L}_{\text{InfoNCE}}$ — минимизация InfoNCE максимизирует нижнюю
оценку MI между видами; оценка ограничена $\ln N$ ⇒ нужны большие батчи / много негативов (MoCo — очередь негативов из прошлых батчей).

**Аугментации — это постановка задачи.** Решает пара «случайный кроп + искажение цвета»: без цвета два кропа имеют почти одну
гистограмму цветов, и сеть «решает» задачу по средней яркости, ничего не выучив.

**Без негативов (BYOL, SimSiam):** онлайн-ветвь с предиктором $q$ предсказывает выход целевой ветви; целевая ветвь — без градиента
(stop-gradient) или EMA-копия. Тривиальное решение (константа) существует, но динамика обучения с асимметрией (предиктор + stop-grad)
его избегает. **DINO** — самодистилляция на ViT: ученик предсказывает распределение учителя (EMA), центрирование + заострение
(низкая температура учителя) против коллапса; карты внимания сами выделяют объекты.

**CLIP** — тот же InfoNCE, пары «картинка ↔ подпись»: симметричная кросс-энтропия по матрице сходств $N\times N$ батча.

</details>

<details>
<summary><b>5. Верификация и порог: почему 99% на валидации мало</b></summary>

Верификация (1:1): «это один человек?» — порог на расстояние. Метрики: **TAR** (true accept rate) при фиксированном **FAR** (false accept rate).
Идентификация (1:N): запрос сравнивается со всей базой. Если пары сравнений независимы, вероятность хотя бы одного ложного совпадения
с базой из $N$ чужих:

$$
P(\text{ложное совпадение}) = 1 - (1 - \text{FAR})^N \approx 1 - e^{-N\cdot\text{FAR}} .
$$

Турникет на 20 000 человек при FAR = 0.001: $N\cdot\text{FAR} = 20$, $P \approx 1 - e^{-20} \approx 1$ — практически каждый посторонний «узнан». Для 1:N
нужен FAR порядка $10^{-6}$–$10^{-7}$. (Ответ на вопрос из конца лекции.)

**Метрики поиска:** Recall@k — доля запросов, у которых среди $k$ соседей есть правильный класс (сам запрос исключается); mAP@R — средняя
точность по $R$ ближайшим, где $R$ — число объектов класса.

</details>

<details>
<summary><b>6. Приближённый поиск: IVF, PQ, HNSW</b></summary>

**IVF (inverted file):** k-means на $n_{\text{list}}$ центроидов; каждый вектор базы — в список своего центроида. Запрос: найти $n_{\text{probe}}$ ближайших
центроидов и перебрать только их списки. Доля просмотра ≈ $n_{\text{probe}}/n_{\text{list}}$. Ошибка — ближайший сосед в непросмотренной ячейке
(у границ ячеек). Семинар: на эмбеддингах машин recall@10 = 0.90 только при просмотре 31% базы, а на «учебных» гауссовых кластерах
той же размерности при 3% просмотра — 0.98. **Паспортные цифры индекса меряют на своих векторах.**

**Product Quantization** (Jégou et al., 2011): вектор $x \in \mathbb{R}^D$ режем на $M$ кусков по $D/M$; для каждой позиции — своя кодовая книга
из 256 центров (k-means); хранится $M$ байт. **ADC** (asymmetric distance computation): запрос не квантуется,

$$
\|x - y\|^2 \approx \sum_{m=1}^{M}\big\|x^{(m)} - c^{(m)}_{q_m(y)}\big\|^2 ,
$$

для запроса заранее считаются таблицы $M\times256$ расстояний до центров, и расстояние до любого вектора базы — $M$ поисков в таблице.
Сжатие: 512 float32 = 2048 байт → 64 байта при $M = 64$ (32×). Расстояния приближённые — порядок соседей может измениться.
IVF-PQ — квантование остатков относительно центроида ячейки.

**HNSW** (Malkov, Yashunin): многослойный граф. Каждый элемент получает уровень $l = \lfloor-\ln(U)\cdot m_L\rfloor$ ($U \sim U(0,1)$) — верхние
слои экспоненциально разрежены («скоростные шоссе»), нижний содержит всё. Поиск: жадный спуск сверху (переход к соседу, ближе к запросу),
на нижнем слое — поиск с очередью кандидатов размера **efSearch**. Параметры: $M$ — число связей узла (память и качество), efConstruction —
качество построения, efSearch — компромисс скорость/recall на запросе. Сложность запроса ≈ $O(\log N)$. Минусы — память на граф, сложные
удаления. Правило лекции: до ~1 млн векторов — полный перебор; дальше HNSW; при нехватке памяти — PQ.

</details>

<details>
<summary><b>7. Методы интерпретации</b></summary>

**Saliency:** $S = \big|\frac{\partial y^c}{\partial x}\big|$ (максимум по каналам) — одна обратная прогонка; шумная (градиент локален) и насыщается: если признак
«сработал до упора», производная ≈ 0, и важный пиксель выглядит неважным.
**SmoothGrad:** $\frac1n\sum_{k=1}^{n}\nabla_xy^c(x + \varepsilon_k)$, $\varepsilon_k \sim \mathcal{N}(0, \sigma^2)$ — усреднение снижает шум.

**Integrated Gradients** (Sundararajan et al., 2017): от базового входа $x'$ (чёрная картинка) к $x$:

$$
\text{IG}_i(x) = (x_i - x_i')\int_0^1\frac{\partial y^c\big(x' + \alpha(x - x')\big)}{\partial x_i}\,d\alpha ,
$$

**аксиома полноты**: $\sum_i\text{IG}_i = y^c(x) - y^c(x')$ (по теореме о градиенте вдоль пути) — лечит насыщение.

**Occlusion:** закрываем окно серым и меряем падение $p(c)$; работает с чёрным ящиком, но сотни прогонов.

**CAM** (Zhou et al., 2016): для сети «свёртки → GAP → линейный слой» логит $y^c = \sum_k w_k^c\cdot\frac{1}{Z}\sum_{ij}A^k_{ij}$, значит
$\text{CAM}^c_{ij} = \sum_k w_k^cA^k_{ij}$ — вклад каждой позиции в логит. **Grad-CAM** (Selvaraju et al., 2017) обобщает на любую архитектуру, заменяя
веса усреднёнными градиентами:

$$
\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}, \qquad L^c_{\text{Grad-CAM}} = \operatorname{ReLU}\Big(\sum_k\alpha_k^cA^k\Big) .
$$

Для архитектуры CAM $\alpha_k^c = w_k^c/Z$ — совпадает с CAM с точностью до множителя. ReLU оставляет только области, **увеличивающие**
логит класса. Разрешение карты — разрешение последнего свёрточного слоя ($7\times7$ у ResNet на 224) → «область, а не контур».
Реализация — forward hook сохраняет $A$, backward hook (или `torch.autograd.grad`) — $\partial y^c/\partial A$.

**Sanity checks** (Adebayo et al., 2018): перезапустить метод на сети со **случайными весами**. Если карта почти не изменилась — она отражает
архитектуру/вход (например, работает как детектор краёв), а не выученное. Семинар: корреляция карт обученной и случайной сети:
Grad-CAM +0.08 (проходит), SmoothGrad +0.54 (во многом про архитектуру).

**Что эти методы не доказывают:** отвечают на «где доводы за класс $c$», а не «почему»; разные методы дают разные карты; красивая карта
убеждает сильнее, чем имеет право. Проверка — **данными**: собрать тест, где подозреваемая подсказка не работает (семинар: сеть с
«жёлтым квадратиком у внедорожников» — 1.000 на своей валидации и 0.375 на честном тесте, хуже константы 0.625; Grad-CAM: 40.8%
внимания в углу, занимающем 4.3% кадра).

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему triplet без margin не работает?</summary>

$f\equiv\text{const}$ даёт нулевой loss (коллапс).

</details>

<details>
<summary>Почему случайные тройки плохи?</summary>

почти все easy: нулевой градиент; нужен PK-батч + hard-mining.

</details>

<details>
<summary>Можно ли ставить модель с TAR@FAR=0.001 = 99% на турникет на 20 000 человек?</summary>

нет: $1-(1-10^{-3})^{20000}\approx1$ ложных совпадений.

</details>

<details>
<summary>Что следует из того, что Grad-CAM «смотрит на объект»?</summary>

только «где доводы за класс»; не «почему»; проверять данными и sanity check.

</details>

<details>
<summary>Чем аугментации в SimCLR отличаются от обычных?</summary>

они и есть постановка задачи (что считать «одним и тем же»).

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- ArcFace: $-\ln\frac{e^{s\cos(\theta_y+m)}}{e^{s\cos(\theta_y+m)}+\sum_{j\ne y}e^{s\cos\theta_j}}$ — угловой margin, так на практике учат лица.
- InfoNCE: $\partial\ell/\partial s_k=\frac1\tau(p_k-[k=j])$ — встроенный hard-negative mining; $I(x;y)\ge\ln N-\mathcal L$.
- PQ: 512 float32 (2 КБ) → 64 байта; расстояния через таблицы $M\times256$ (ADC). HNSW: уровни с вероятностью $\propto e^{-l/m_L}$, жадный спуск, `efSearch`.
- CAM — частный случай Grad-CAM для «GAP + линейный слой»; Integrated Gradients удовлетворяет аксиоме полноты.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
torch.manual_seed(0)

# Open set: цифры в случайном месте холста 20×20; учим эмбеддер на 0–4, Recall@1 меряем на 5–9 (их сеть не видела)
d = load_digits(); IM = d.images.astype(np.float32) / 16; Y = torch.tensor(d.target)
rng = np.random.default_rng(0)
X = np.zeros((len(IM), 1, 20, 20), np.float32)
for i, im in enumerate(IM):
    dy, dx = rng.integers(0, 13, 2); X[i, 0, dy:dy + 8, dx:dx + 8] = im
X = torch.tensor(X + 0.1 * rng.random(X.shape).astype(np.float32))
tr, te = Y < 5, Y >= 5

def recall_at_1(Z, labels):
    Z = F.normalize(Z.flatten(1), dim=1); S = Z @ Z.T; S.fill_diagonal_(-2)   # себя исключаем
    return (labels[S.argmax(1)] == labels).float().mean().item()

def batch_hard(z, lab, m=0.2):
    z = F.normalize(z, dim=1); D = (2 - 2 * z @ z.T).clamp(min=0)            # ||a-b||² = 2 - 2cos
    same = lab[:, None] == lab[None, :]; eye = torch.eye(len(lab), dtype=torch.bool)
    d_ap = D.masked_fill(~(same & ~eye), -1e9).max(1).values                  # самый далёкий свой
    d_an = D.masked_fill(same, 1e9).min(1).values                             # самый близкий чужой
    return F.relu(d_ap - d_an + m).mean()

net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(64, 32))
with torch.no_grad():
    print(f"сырые пиксели: {recall_at_1(X[te], Y[te]):.3f} | необученная CNN: {recall_at_1(net(X[te]), Y[te]):.3f}")
opt = torch.optim.Adam(net.parameters(), 2e-3); Xtr, ytr = X[tr], Y[tr]
for step in range(400):                                                       # PK-батч: 5 классов × 8 примеров
    idx = torch.cat([torch.nonzero(ytr == c)[torch.randperm(int((ytr == c).sum()))[:8], 0] for c in range(5)])
    loss = batch_hard(net(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    print(f"triplet batch-hard: на невиданных классах 5–9 {recall_at_1(net(X[te]), Y[te]):.3f}, "
          f"на классах обучения 0–4 {recall_at_1(net(X[tr]), Y[tr]):.3f}  ← open-set разрыв")
```

Результат запуска:

```text
сырые пиксели: 0.651 | необученная CNN: 0.695
triplet batch-hard: на невиданных классах 5–9 0.746, на классах обучения 0–4 1.000  ← open-set разрыв
```

```python
# Свой IVF: k-means на ячейки, ищем в nprobe ближайших ячейках; recall@10 против доли просмотра базы
from sklearn.cluster import KMeans
rng = np.random.default_rng(0)
base = rng.normal(size=(5000, 64)).astype(np.float32); base /= np.linalg.norm(base, axis=1, keepdims=True)
queries = base[rng.choice(5000, 200, replace=False)] + 0.05 * rng.normal(size=(200, 64)).astype(np.float32)
exact = np.argsort(-(queries @ base.T), axis=1)[:, :10]
km = KMeans(64, n_init=1, random_state=0).fit(base); cells = km.labels_
for nprobe in (1, 4, 16, 64):
    near_cells = np.argsort(-(queries @ km.cluster_centers_.T), axis=1)[:, :nprobe]
    hits, scanned = 0, 0
    for q, cl, ex in zip(queries, near_cells, exact):
        cand = np.flatnonzero(np.isin(cells, cl)); scanned += len(cand)
        top = cand[np.argsort(-(base[cand] @ q))[:10]]; hits += len(set(top) & set(ex))
    print(f"nprobe={nprobe:2d}: recall@10={hits / 2000:.3f}, просмотрено {scanned / 200 / 5000:.1%} базы")
```

Результат запуска:

```text
nprobe= 1: recall@10=0.190, просмотрено 1.6% базы
nprobe= 4: recall@10=0.383, просмотрено 6.3% базы
nprobe=16: recall@10=0.728, просмотрено 25.0% базы
nprobe=64: recall@10=1.000, просмотрено 100.0% базы
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Обучение представлений](https://education.yandex.ru/handbook/ml/article/obuchenie-predstavlenij)
- Статьи: [FaceNet](https://arxiv.org/abs/1503.03832) · [In Defense of the Triplet Loss](https://arxiv.org/abs/1703.07737) · [ArcFace](https://arxiv.org/abs/1801.07698) · [SimCLR](https://arxiv.org/abs/2002.05709) · [MoCo](https://arxiv.org/abs/1911.05722) · [BYOL](https://arxiv.org/abs/2006.07733) · [DINO](https://arxiv.org/abs/2104.14294) · [Metric Learning Reality Check](https://arxiv.org/abs/2003.08505) · [HNSW](https://arxiv.org/abs/1603.09320) · [Product Quantization](https://hal.inria.fr/inria-00514462v2/document) · [Grad-CAM](https://arxiv.org/abs/1610.02391) · [Integrated Gradients](https://arxiv.org/abs/1703.01365) · [LIME](https://arxiv.org/abs/1602.04938) · [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) · [hnswlib](https://github.com/nmslib/hnswlib) · [Distill.pub](https://distill.pub/)
- [DL2 HSE, неделя 7: Self-Supervised Vision Models](https://github.com/thecrazymage/DL2_HSE/tree/main/week_07)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Лекция_6_Contrastive_learning_и_интерпретация.pptx` (DEEP LEARNING, день 6, 42 слайда).
- `Семинар_6_Эмбеддинги,_поиск_по_векторам_и_Grad_CAM.ipynb` (Stanford Cars): open-set сплит, бейзлайны, contrastive с подбором margin, triplet с тремя стратегиями майнинга и PK-сэмплером, пространство эмбеддингов, свой IVF, «умный Ганс», Grad-CAM на хуках, SmoothGrad, occlusion, sanity check.
- `task_for_students_3.ipynb` — ДЗ 3 «Поиск похожих птиц: эмбеддинги и HNSW».

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Hadsell R., Chopra S., LeCun Y. *Dimensionality Reduction by Learning an Invariant Mapping* (2006); Schroff F. et al. *FaceNet* (CVPR 2015); Hermans A. et al. *In Defense of the Triplet Loss* (2017)
- Deng J. et al. *ArcFace* (CVPR 2019); Musgrave K. et al. *A Metric Learning Reality Check* (ECCV 2020)
- Oord A. et al. *CPC / InfoNCE* (2018); Chen T. et al. *SimCLR* (ICML 2020); He K. et al. *MoCo* (CVPR 2020); Grill J.-B. et al. *BYOL* (2020); Chen X., He K. *SimSiam* (2021); Caron M. et al. *DINO* (ICCV 2021); Radford A. et al. *CLIP* (2021)
- Arandjelović R. et al. *NetVLAD* (CVPR 2016); Berton G. et al. *CosPlace* (CVPR 2022)
- Jégou H., Douze M., Schmid C. *Product Quantization for Nearest Neighbor Search* (TPAMI 2011); Malkov Y., Yashunin D. *HNSW* (TPAMI 2020); Johnson J. et al. *FAISS* (2017)
- Zhou B. et al. *CAM* (CVPR 2016); Selvaraju R. et al. *Grad-CAM* (ICCV 2017); Smilkov D. et al. *SmoothGrad* (2017); Sundararajan M. et al. *Integrated Gradients* (ICML 2017); Ribeiro M. et al. *LIME* (KDD 2016); Adebayo J. et al. *Sanity Checks for Saliency Maps* (NeurIPS 2018)

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL, лекция 6: contrastive learning и интерпретация
- **Семинары:** [Семинар 6: эмбеддинги, векторный поиск, Grad-CAM](../../materials/seminars/19_Семинар_6_Эмбеддинги_GradCAM.ipynb)
- **Домашние задания:** [DL, ДЗ 3: поиск похожих птиц, HNSW (задание)](../../materials/homework/19_DL_ДЗ_3_Поиск_похожих_птиц.ipynb)
- **Видео и внешние курсы:** [Stanford CS231n — CNN for Visual Recognition](https://cs231n.stanford.edu/) · [DL2 HSE (thecrazymage/DL2_HSE)](https://github.com/thecrazymage/DL2_HSE)

</details>

---

[← 18. Детекция объектов](../18_detection/README.md) · [Оглавление курса](../../README.md) · [20. Рекуррентные сети: RNN, LSTM, GRU →](../20_rnn/README.md)
