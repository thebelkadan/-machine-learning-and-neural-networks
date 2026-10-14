# 32. Графовые нейросети

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Многие данные — **графы**: соцсети, молекулы, транзакции, дороги, граф «пользователь — товар». Графовые нейросети
(GNN) учат представления узлов, рёбер или целого графа по принципу **передачи сообщений**: на каждом слое узел собирает
векторы соседей, агрегирует их (сумма, среднее, внимание) и обновляет свой вектор. $k$ слоёв — информация из $k$-окрестности.
Базовые модели: **GCN** (нормированное усреднение по соседям), **GraphSAGE** (сэмплирование соседей, работает на новых
узлах), **GAT** (веса соседей через внимание), **GIN** (максимальная различающая сила). Ограничения: **пересглаживание**
при многих слоях, плохая работа при **гетерофилии** (соседи непохожи), масштабирование на огромные графы.

| Термин | Определение |
|---|---|
| Матрица смежности $A$ | $A_{ij}=1$, если есть ребро $i\text{—}j$; степень $d_i=\sum_jA_{ij}$ |
| Message passing | $h_v^{(k+1)}=\mathrm{UPDATE}\big(h_v^{(k)},\ \mathrm{AGG}\lbrace h_u^{(k)}:u\in N(v)\rbrace\big)$ |
| GCN | $H'=\sigma(\hat AHW)$, $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$ |
| Трансдуктивная / индуктивная | граф известен целиком, размечена часть узлов / предсказываем для новых узлов и графов |
| Readout | агрегация узлов в вектор всего графа (сумма, среднее, attention pooling) |
| Гомофилия | соседи похожи (одного класса); гетерофилия — наоборот |
| Пересглаживание | с ростом числа слоёв векторы всех узлов становятся одинаковыми |
| Link prediction | предсказать, есть ли ребро между узлами (рекомендации, достройка графа знаний) |

**Ключевые формулы**

$$
h_v^{(k+1)}=\sigma\Big(W^{(k)}\sum_{u\in N(v)\cup\lbrace v\rbrace}\frac{h_u^{(k)}}{\sqrt{\tilde d_u\tilde d_v}}\Big),\qquad
\alpha_{vu}=\operatorname{softmax}_u\Big(\mathrm{LeakyReLU}\big(a^\top[Wh_v\,\Vert\,Wh_u]\big)\Big),\qquad
h_G=\sum_vh_v^{(K)}
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>1. Графы в машинном обучении</b></summary>

| Уровень задачи | Примеры |
|---|---|
| **Узлы** | тип пользователя в соцсети, мошеннический аккаунт, тематика статьи в графе цитирований |
| **Рёбра** | рекомендации (пользователь — товар), предсказание связей, взаимодействие лекарств |
| **Граф целиком** | свойства молекулы (токсичность, растворимость), классификация программ |

Почему не обычная сеть: число соседей разное, у узлов нет порядка (модель должна быть **инвариантна к перестановке** узлов),
граф может быть огромным. CNN — частный случай GNN на решётке; трансформер — GNN на полном графе, где веса рёбер даёт
внимание (тема 23).

**Представление:** матрица смежности (плотная — только для маленьких графов), список рёбер `edge_index` размера $2\times E$ (стандарт
PyTorch Geometric), признаки узлов $X\in\mathbb R^{n\times f}$, признаки рёбер.

</details>

<details>
<summary><b>2. Признаки без нейросетей</b></summary>

- **Структурные признаки узла:** степень, PageRank, коэффициент кластеризации, центральность (betweenness, closeness), число
  треугольников; для пар узлов — число общих соседей, Adamic–Adar, Jaccard. С бустингом (тема 09) это сильный бейзлайн.
- **Спектральная кластеризация** (тема 11): собственные векторы лапласиана $L=D-A$ — координаты для разбиения графа на сообщества.
- **Эмбеддинги узлов:** DeepWalk и **node2vec** — случайные блуждания как «предложения» и word2vec на них (тема 21); узлы, часто
  встречающиеся рядом в блужданиях, получают близкие векторы. Минус — трансдуктивность: для нового узла эмбеддинга нет.

</details>

<details>
<summary><b>3. Передача сообщений</b></summary>

Общая схема слоя:
1. **Сообщение** от соседа $u$ к $v$: $m_{uv}=\mathrm{MSG}(h_u,h_v,e_{uv})$.
2. **Агрегация** по соседям — перестановочно-инвариантная функция: сумма, среднее, максимум, внимание.
3. **Обновление**: $h_v'=\mathrm{UPDATE}(h_v,\mathrm{AGG}_u\,m_{uv})$ — обычно линейный слой + нелинейность (+ residual, нормализация).

После $k$ слоёв вектор узла зависит от его $k$-окрестности. Веса $W$ общие для всех узлов — число параметров не зависит от размера
графа. Задача на узлах — голова на $h_v$; на рёбрах — на паре $(h_u,h_v)$; на графе — **readout** (сумма/среднее/максимум узлов).

</details>

<details>
<summary><b>4. GCN</b></summary>

Kipf & Welling (2017): $H^{(k+1)}=\sigma\big(\hat AH^{(k)}W^{(k)}\big)$, $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$.
- $A+I$ — петли: узел учитывает и собственный вектор.
- Симметричная нормировка на $\sqrt{\tilde d_u\tilde d_v}$ — чтобы узлы с большой степенью не «раздували» сумму, а масштаб признаков не
  менялся от слоя к слою.
- Интерпретация: сглаживание признаков по графу (усреднение с соседями) + линейный слой. Вывод из спектральных свёрток — раздел
  «2. GCN как фильтр низких частот».

Пример кода этой темы: клуб карате Закари — 34 человека, клуб раскололся на два лагеря; размечены только тренер и президент. GCN на
одной структуре графа (признаки — one-hot номера узлов) правильно относит 97% остальных; MLP без графа — на уровне угадывания (0.5).

</details>

<details>
<summary><b>5. GraphSAGE, GAT, GIN</b></summary>

- **GraphSAGE** (Hamilton et al., 2017): $h_v'=\sigma\big(W[h_v\,\Vert\,\mathrm{AGG}(h_u)]\big)$ — свой вектор отдельно от агрегата соседей;
  **сэмплирование** фиксированного числа соседей на каждом слое → мини-батчи на графах в миллиарды рёбер; **индуктивность** —
  работает на новых узлах по их признакам и соседям (PinSage в Pinterest).
- **GAT** (Veličković et al., 2018): веса соседей — внимание $\alpha_{vu}$ по паре векторов, несколько голов; полезно, когда соседи неравноценны.
- **GIN** (Xu et al., 2019): $h_v'=\mathrm{MLP}\big((1+\varepsilon)h_v+\sum_uh_u\big)$ — **сумма** (а не среднее или максимум) с MLP различает графы так же
  хорошо, как тест Вейсфейлера–Лемана — верхняя граница для обычных GNN (раздел «4. Различающая сила и тест WL»).
- Признаки рёбер: MPNN, Edge-conditioned convolutions; для молекул — учёт 3D-координат (SchNet, эквивариантные сети).

</details>

<details>
<summary><b>6. Задачи на рёбрах и графах</b></summary>

- **Link prediction:** кодировщик (GNN) даёт $h_u,h_v$, декодер — скор $\langle h_u,h_v\rangle$ или MLP от пары; обучение — бинарная
  кросс-энтропия на существующих рёбрах против **негативных** (случайных) пар, как в word2vec (тема 21). Важно разделять рёбра на
  train/val/test и не пропускать тестовые рёбра через сообщения (утечка).
- **Рекомендации** (тема 13): двудольный граф «пользователь — товар»; **LightGCN** — только усреднение по соседям без нелинейностей
  и весов в слоях, эмбеддинги пользователей и товаров учатся BPR-лоссом; сильный бейзлайн для коллаборативной фильтрации.
- **Классификация графов:** GIN + readout (сумма по слоям), иерархический пулинг (DiffPool, TopK). Молекулы: свойства, поиск
  лекарств; бенчмарки OGB.

</details>

<details>
<summary><b>7. Ограничения и как с ними жить</b></summary>

- **Пересглаживание (oversmoothing):** каждый слой усредняет соседей; через много слоёв векторы в связной компоненте
  сходятся. Пример кода этой темы: на графе с гомофилией GCN из 4 слоёв — 0.96, из 16 — 0.30 (уровень угадывания). Средства:
  residual-связи, нормализации (PairNorm), JKNet (конкатенация выходов всех слоёв), DropEdge, неглубокие сети (2–3 слоя — норма).
- **Гетерофилия:** если соседи обычно из другого класса, усреднение смешивает признаки разных классов. В примере GCN при
  гетерофилии не лучше MLP на признаках. Средства: отдельная обработка своего вектора и агрегата соседей (GraphSAGE-подобные
  модели), учёт соседей второго порядка (H2GCN), знаковые веса.
- **Oversquashing:** информация с далёких узлов сжимается в векторы фиксированного размера через «узкие» места графа; средства —
  перестройка графа (rewiring), графовые трансформеры.
- **Масштаб:** полный граф не влезает в память → сэмплирование соседей (GraphSAGE), подграфы (ClusterGCN, GraphSAINT),
  предвычисление агрегатов (SIGN).

</details>

<details>
<summary><b>8. Графовые трансформеры и практика</b></summary>

**Графовые трансформеры** (Graphormer, GPS): внимание между всеми узлами + структурные/позиционные кодировки (собственные векторы
лапласиана, кратчайшие пути, случайные блуждания), часто в сочетании с локальной передачей сообщений. Хороши на небольших графах
(молекулы), дороги на больших ($O(n^2)$).

**Инструменты:** PyTorch Geometric (`GCNConv`, `SAGEConv`, `GATConv`, `GINConv`, загрузчики с сэмплированием, датасеты Cora, OGB),
DGL, NetworkX для анализа и признаков.

```python
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, f_in, hidden, n_cls):
        super().__init__(); self.c1, self.c2 = GCNConv(f_in, hidden), GCNConv(hidden, n_cls)
    def forward(self, x, edge_index):
        return self.c2(F.dropout(F.relu(self.c1(x, edge_index)), 0.5, self.training), edge_index)
```

**Чек-лист**
- Бейзлайн: признаки узлов + структурные признаки (степень, PageRank) + бустинг; MLP без графа.
- 2–3 слоя GNN, dropout, weight decay; residual при более глубоких.
- Проверить гомофилию (доля рёбер между узлами одного класса) — при низкой GCN может навредить.
- Link prediction: честное разбиение рёбер, негативное сэмплирование, метрики ранжирования (Hits@k, MRR).
- Большие графы — сэмплирование соседей или подграфов.

</details>

**Математика и выводы**

<details>
<summary><b>1. Нормированная матрица смежности и её спектр</b></summary>

$\tilde A=A+I$, $\tilde D=\operatorname{diag}(\tilde A\mathbf 1)$, $\hat A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ — симметричная, собственные значения в $(-1,1]$, наибольшее равно 1 с
собственным вектором $\tilde D^{1/2}\mathbf 1$ (пропорционален $\sqrt{\tilde d}$). Применение $\hat A$ к сигналу на узлах $x$:
$(\hat Ax)_v=\sum_{u\in N(v)\cup\lbrace v\rbrace}\frac{x_u}{\sqrt{\tilde d_u\tilde d_v}}$ — взвешенное среднее по окрестности.

**Пересглаживание:** без весов и нелинейностей $\hat A^kx=\sum_i\lambda_i^k\langle u_i,x\rangle u_i\to\langle u_1,x\rangle u_1$ при $k\to\infty$ (для связного
неразрывного графа все $\lvert\lambda_i\rvert<1$, кроме $\lambda_1=1$) — сигнал на всех узлах становится пропорционален $\sqrt{\tilde d_v}$, информация
о том, где был узел, исчезает. Скорость — $\lvert\lambda_2\rvert^k$: в хорошо перемешивающихся графах сглаживание наступает за несколько
слоёв. Веса и нелинейности замедляют, но не отменяют процесс (Oono & Suzuki, 2020).

</details>

<details>
<summary><b>2. GCN как фильтр низких частот</b></summary>

**Спектральная свёртка** на графе: разложение по собственным векторам нормированного лапласиана $L=I-D^{-1/2}AD^{-1/2}=U\Lambda U^\top$
(аналог базиса Фурье), фильтр — функция от собственных значений: $g_\theta\star x=U\,g_\theta(\Lambda)\,U^\top x$. Дорого (нужно $U$) и не
локально. **ChebNet** приближает $g_\theta$ многочленом Чебышёва степени $K$ от $L$ — фильтр становится $K$-локальным и считается
умножениями на разреженную $L$. GCN берёт $K=1$ и один параметр: $g_\theta\star x\approx\theta(I+D^{-1/2}AD^{-1/2})x$, а затем
**renormalization trick** $I+D^{-1/2}AD^{-1/2}\to\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ для устойчивости.

Частотная характеристика: $\hat A=I-\tilde L$, т.е. компонента с «частотой» $\tilde\lambda$ умножается на $1-\tilde\lambda$ — низкие частоты (гладкие
по графу сигналы) проходят, высокие (резко меняющиеся между соседями) подавляются. Поэтому GCN хорош при гомофилии (метки гладкие
по графу) и плох при гетерофилии (полезный сигнал — высокочастотный).

</details>

<details>
<summary><b>3. Внимание в GAT</b></summary>

$e_{vu}=\mathrm{LeakyReLU}\big(a^\top[Wh_v\,\Vert\,Wh_u]\big)$ для $u\in N(v)\cup\lbrace v\rbrace$, $\alpha_{vu}=\frac{\exp e_{vu}}{\sum_{w}\exp e_{vw}}$,
$h_v'=\sigma\big(\sum_u\alpha_{vu}Wh_u\big)$; $K$ голов конкатенируются (на последнем слое усредняются). Так как $a^\top[Wh_v\Vert Wh_u]=a_1^\top Wh_v+a_2^\top Wh_u$,
после монотонного LeakyReLU и softmax по $u$ ранжирование соседей **одинаково для всех** $v$ («статическое внимание»); GATv2 переставляет
нелинейность: $a^\top\mathrm{LeakyReLU}(W[h_v\Vert h_u])$ — внимание становится действительно зависящим от запроса.

</details>

<details>
<summary><b>4. Различающая сила и тест WL</b></summary>

**Тест Вейсфейлера–Лемана (1-WL):** раскрасить узлы одинаково; на каждом шаге новый цвет узла — хэш от (его цвет, **мультимножество**
цветов соседей); если гистограммы цветов двух графов различаются — графы не изоморфны. Обычные GNN с передачей сообщений —
не сильнее 1-WL (Xu et al., 2019): если 1-WL не различает графы, их не различит и GNN.

Чтобы достичь 1-WL, агрегация должна быть **инъективной** на мультимножествах: **среднее** не различает $\lbrace a,b\rbrace$ и $\lbrace a,a,b,b\rbrace$,
**максимум** — $\lbrace a,b\rbrace$ и $\lbrace a,a,b\rbrace$, а **сумма** (на подходящем счётном пространстве признаков) инъективна. Отсюда GIN: сумма соседей +
MLP (универсальная аппроксимация инъективной функции). Предел 1-WL: регулярные графы с одинаковой степенью (например, два
треугольника и шестиугольник) неразличимы — нужны более сильные модели (высшие WL, подграфовые признаки, позиционные кодировки).

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему для графов не подходит обычная полносвязная сеть?</summary>

разное число соседей и отсутствие порядка узлов: модель должна быть инвариантна к перестановке.

</details>

<details>
<summary>Зачем в GCN петли и нормировка $\tilde D^{-1/2}$?</summary>

петли — учитывать собственный вектор; нормировка — не раздувать сумму у узлов с большой степенью.

</details>

<details>
<summary>Почему GNN редко делают глубокими?</summary>

пересглаживание: после многих слоёв усреднения векторы узлов неразличимы (в примере 16 слоёв — 0.30 против 0.96 у 4 слоёв).

</details>

<details>
<summary>Когда GCN хуже MLP?</summary>

при гетерофилии: усреднение по соседям другого класса портит признаки.

</details>

<details>
<summary>Чем GraphSAGE отличается от GCN?</summary>

отдельный вектор узла и агрегат соседей, сэмплирование соседей, индуктивность (работает на новых узлах).

</details>

<details>
<summary>Почему в GIN сумма, а не среднее?</summary>

сумма инъективна на мультимножествах соседей, среднее и максимум теряют информацию о количестве.

</details>

<details>
<summary>Как не допустить утечку в link prediction?</summary>

убрать тестовые рёбра из графа, по которому передаются сообщения при обучении.

</details>

<details>
<summary>Чем трансформер похож на GNN?</summary>

это GNN на полном графе, где веса рёбер вычисляются вниманием.

</details>

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# GCN с нуля: клуб карате (2 размеченных узла из 34) и синтетика с гомофилией / гетерофилией / пересглаживанием
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, networkx as nx
def normalize_adj(A):                                            # Â = D^-1/2 (A + I) D^-1/2
    A = A + torch.eye(len(A)); d = A.sum(1); return A / torch.sqrt(d[:, None] * d[None, :])
class GCN(nn.Module):
    def __init__(self, f_in, hidden, n_cls, layers=2):
        super().__init__(); dims = [f_in] + [hidden] * (layers - 1) + [n_cls]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
    def forward(self, X, A_hat):
        for k, lin in enumerate(self.lins):
            X = A_hat @ lin(X)                                     # передача сообщений = умножение на Â
            if k < len(self.lins) - 1: X = F.relu(X)
        return X
def train_eval(model, X, A_hat, y, train_idx, test_idx, epochs=200):
    opt = torch.optim.Adam(model.parameters(), 0.01, weight_decay=5e-4)
    for _ in range(epochs):
        loss = F.cross_entropy(model(X, A_hat)[train_idx], y[train_idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return (model(X, A_hat)[test_idx].argmax(1) == y[test_idx]).float().mean().item()

G = nx.karate_club_graph()
A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
y = torch.tensor([0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in G.nodes])
X = torch.eye(34)                                                  # признаков нет — только номер узла
train_idx = torch.tensor([0, 33]); test_idx = torch.tensor([v for v in range(34) if v not in (0, 33)])
torch.manual_seed(0); print(f"карате, GCN по структуре графа: {train_eval(GCN(34, 16, 2), X, normalize_adj(A), y, train_idx, test_idx):.3f}")
torch.manual_seed(0); print(f"карате, MLP без графа:          {train_eval(GCN(34, 16, 2), X, torch.eye(34), y, train_idx, test_idx):.3f}")

def sbm_graph(p_same, p_diff, n=300, k=3, seed=0):                # стохастическая блочная модель + шумные признаки
    rng = np.random.default_rng(seed); labels = rng.integers(0, k, n)
    P = np.where(labels[:, None] == labels[None, :], p_same, p_diff)
    A = np.triu((rng.random((n, n)) < P).astype(np.float32), 1); A = A + A.T
    X = np.hstack([np.eye(k)[labels] * 0.6 + rng.normal(size=(n, k)), rng.normal(size=(n, 13))])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(A), torch.tensor(labels)
for name, p_same, p_diff in [("гомофилия (свои чаще)  ", 0.06, 0.005), ("гетерофилия (чужие чаще)", 0.005, 0.03)]:
    X, A, y = sbm_graph(p_same, p_diff)
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(0)); tr, te = perm[:60], perm[60:]
    torch.manual_seed(0); res = {"MLP": train_eval(GCN(16, 32, 3), X, torch.eye(300), y, tr, te)}
    for L in (2, 4, 8, 16):
        torch.manual_seed(0); res[f"GCN-{L}"] = train_eval(GCN(16, 32, 3, L), X, normalize_adj(A), y, tr, te)
    print(name, " ".join(f"{k}: {v:.2f}" for k, v in res.items()))
```

Результат запуска:

```text
карате, GCN по структуре графа: 0.969
карате, MLP без графа:          0.500
гомофилия (свои чаще)   MLP: 0.38 GCN-2: 0.89 GCN-4: 0.96 GCN-8: 0.93 GCN-16: 0.30
гетерофилия (чужие чаще) MLP: 0.38 GCN-2: 0.38 GCN-4: 0.29 GCN-8: 0.29 GCN-16: 0.28
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Графовые нейронные сети](https://education.yandex.ru/handbook/ml/article/grafovye-nejronnye-seti)
- Курсы и книги: [Stanford CS224W — Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/) · [W. Hamilton — Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) · [Distill — A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- Инструменты: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) · [DGL](https://www.dgl.ai/) · [NetworkX](https://networkx.org/) · [Open Graph Benchmark](https://ogb.stanford.edu/)
- Статьи: [GCN](https://arxiv.org/abs/1609.02907) · [ChebNet](https://arxiv.org/abs/1606.09375) · [GraphSAGE](https://arxiv.org/abs/1706.02216) · [GAT](https://arxiv.org/abs/1710.10903) · [GATv2](https://arxiv.org/abs/2105.14491) · [GIN](https://arxiv.org/abs/1810.00826) · [MPNN](https://arxiv.org/abs/1704.01212) · [DeepWalk](https://arxiv.org/abs/1403.6652) · [node2vec](https://arxiv.org/abs/1607.00653) · [LightGCN](https://arxiv.org/abs/2002.02126) · [PinSage](https://arxiv.org/abs/1806.01973) · [Deeper Insights into GCN (oversmoothing)](https://arxiv.org/abs/1801.07606) · [Graph Neural Networks Exponentially Lose Expressive Power](https://arxiv.org/abs/1905.10947) · [H2GCN (гетерофилия)](https://arxiv.org/abs/2006.11468) · [GPS Graph Transformer](https://arxiv.org/abs/2205.12454)

</details>

<details>
<summary><b>Источники курса</b></summary>

- Тема дополнительная: в лекциях курса её нет. Материал по учебнику ML Яндекса, курсу CS224W, книге У. Гамильтона и оригинальным статьям; связь с темами 11 (спектральная кластеризация), 13 (рекомендации), 21 (word2vec), 23 (трансформер).

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Внешние курсы:** [Учебник ML Яндекса: графовые нейросети](https://education.yandex.ru/handbook/ml/article/grafovye-nejronnye-seti) · [Stanford CS224W](https://web.stanford.edu/class/cs224w/)
- **Связанные темы:** [11. Кластеризация](../11_clustering_dim_reduction/README.md) · [13. Рекомендательные системы](../13_recommender_systems/README.md) · [21. word2vec](../21_word_embeddings_tokenization/README.md) · [23. Трансформер](../23_transformer_bert/README.md)

</details>

---

[← 31. Обучение с подкреплением: от уравнения Беллмана до PPO и RLHF](../31_reinforcement_learning/README.md) · [Оглавление курса](../../README.md) · [33. Временные ряды: прогнозирование, валидация, модели →](../33_time_series/README.md)
