# 15. Обучение нейросетей: инициализация, оптимизаторы, регуляризация, нормализация

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Глубокую сеть трудно учить, потому что градиент — произведение якобианов всех слоёв и легко затухает или
взрывается. Масштаб удерживают: **стандартизация входов** и **инициализация** (Xavier/He) — на старте; **нормализация**
(BatchNorm/LayerNorm) — во время обучения; **оптимизатор** (momentum, Adam/AdamW) и **расписание** темпа (warmup + cosine)
выбирают шаг; **регуляризация** (weight decay, dropout, early stopping, аугментации) борется с переобучением;
**клиппинг** спасает от взрывов.

| Термин | Определение |
|---|---|
| Xavier / He | $\operatorname{Var}w=\frac{2}{n_{in}+n_{out}}$ / $\frac{2}{n_{in}}$ (для ReLU) |
| Momentum | $v\leftarrow\beta v+g$, $w\leftarrow w-\eta v$ — инерция |
| Adam | momentum + RMSprop + коррекция смещения $1/(1-\beta^t)$ |
| AdamW | weight decay отдельно от адаптивного шага: $w\leftarrow(1-\eta\lambda)w-\dots$ |
| Warmup / cosine | разогрев темпа в начале / плавное затухание до нуля |
| Dropout | обнуление нейронов с вероятностью $p$ на обучении, отключение на тесте |
| BatchNorm | нормировка по батчу + обучаемые $\gamma,\beta$; на тесте — скользящие статистики |
| Gradient clipping | $g\leftarrow g\min(1,\theta/\lVert g\rVert)$ |

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\quad w_t=w_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon},\qquad
\text{BN}: y=\gamma\frac{x-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}}+\beta
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>1. Главная проблема: затухание и взрыв градиента</b></summary>

Градиент по ранним слоям — **произведение якобианов** всех последующих:
$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_3}\cdot\frac{\partial h_3}{\partial h_2}\cdot\frac{\partial h_2}{\partial h_1}$; в сети из $L$ слоёв — $L$ множителей:
$0.9^{50} \approx 0.005$ (затухание), $1.1^{50} \approx 117$ (взрыв).

- **Затухание:** ранние слои почти не обновляются, обучение стоит (сигмоида режет градиент минимум в 4 раза на слой).
- **Взрыв:** loss скачет и уходит в NaN, веса разлетаются (особенно в RNN, тема [20](../20_rnn/README.md)).

Вся «кухня» обучения — про то, как **удержать это произведение около единицы**: нормировка входов и инициализация задают
масштаб на старте, нормализация удерживает его во время обучения, оптимизатор и расписание выбирают шаг,
регуляризация борется с переобучением.

</details>

<details>
<summary><b>2. Нормировка входов и инициализация</b></summary>

**Нормировка данных (стандартизация):** $x' = (x - \mu)/\sigma$ по каждому признаку, статистики — **только по train**
(иначе утечка). Константный на train признак → нули, а не NaN.

**Инициализация весов — ключевые идеи:**
- **нарушить симметрию** — чтобы нейроны были разными (одинаковые веса → одинаковые градиенты → один нейрон вместо $k$);
- **не допустить «насыщения»** нейронов (почти всегда 0 или 1);
- **входы всех слоёв должны иметь одинаковую дисперсию** — сигнал проходит, не затухая и не взрываясь;
- **градиенты всех слоёв — одинаковую дисперсию** — тогда один темп обучения подходит всем слоям;
- смещения — нулями.

| Инициализация | Дисперсия весов | Для чего |
|---|---|---|
| **Xavier / Glorot** (2010) | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$; uniform: $U\big[\pm\sqrt{6/(n_{\text{in}} + n_{\text{out}})}\big]$ | tanh, линейные, sigmoid |
| **Kaiming / He** (2015) | $\frac{2}{n_{\text{in}}}$ (gain $= \sqrt2$: ReLU «съедает» половину сигнала) | **ReLU** и её варианты |

Формула — выкладка про дисперсию, верная **на инициализации** (веса независимы, вход независим от весов)
(вывод — см. раздел «1. Инициализация: вывод Xavier и He»). В PyTorch `nn.Linear`/`nn.Conv2d` по умолчанию используют вариант Kaiming-uniform.
Поиграйте: [deeplearning.ai — Initialization](https://www.deeplearning.ai/ai-notes/initialization/).

```python
torch.nn.init.xavier_uniform_(layer.weight)
torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
torch.nn.init.zeros_(layer.bias)
```

</details>

<details>
<summary><b>3. Мини-батчи</b></summary>

- градиенты менее случайны (оценка по подвыборке, разброс падает как $1/\sqrt B$);
- батч из $B$ объектов — **одно матричное умножение** вместо $B$ отдельных → быстро на GPU;
- можно нормировать по батчу (BatchNorm); можно специально составлять батчи (например, с представителями всех классов —
  это уже выход за рамки теории SGD);
- новый гиперпараметр — **размер батча**, связан с темпом обучения: **увеличение батча ≈ уменьшение темпа** (рычаг — $\eta/B$).

**Какой брать:** максимальный, при котором максимальное ускорение (а не просто «влезает в память»). Замер из лекции
(MLP на CIFAR-10, SGD, 20 эпох, $\eta$ подобран под каждый батч): от 8 до 1024 тест почти не меняется (51–52%) — выбирайте
по скорости и памяти; лучший $\eta$ растёт вместе с батчем до порога расходимости; при слишком большом батче шагов за
эпоху мало — сеть недоучивается.

**Зачем большой батч и меньше шагов:** внутри шага работа параллелится, между шагами — нет (шаг $t+1$ ждёт весов шага $t$);
время ≈ число шагов × время шага. ResNet-50 на ImageNet за 1 час — 256 GPU, батч 8192 (Goyal и др., 2017). На одном GPU,
где большой батч считается пропорционально дольше, выигрыша нет; накопление градиентов даёт большой батч, но не скорость.

Практика: **перемешивать данные перед каждой эпохой**; эпоха — проход по выборке, итерация — один батч.

</details>

<details>
<summary><b>4. Оптимизаторы</b></summary>

| Метод | Обновление | Идея |
|---|---|---|
| **SGD** | $w \leftarrow w - \eta g$ | базовый |
| **SGD + momentum** | $v \leftarrow \beta v + g$, $w \leftarrow w - \eta v$ | инерция: разгон вдоль оврага, гашение колебаний, проскакивание седловых точек |
| **Нестеров** | градиент в точке «с заглядыванием вперёд» | корректирует разгон заранее |
| **Adagrad** (2011) | $w \leftarrow w - \frac{\eta}{\sqrt{\sum g^2} + \varepsilon}g$ | свой шаг для каждого параметра; шаг монотонно падает |
| **RMSprop** (Hinton, 2012) | как Adagrad, но скользящее среднее $g^2$ | шаг не затухает до нуля |
| **Adam** (2014) | momentum + RMSprop + **коррекция смещения** | стандарт по умолчанию |
| **AdamW** (2017) | Adam + **отдельный** weight decay | правильная L2-регуляризация для Adam |

Когда что: Adam/AdamW — по умолчанию (трансформеры, большинство задач); SGD + momentum (+ хорошее расписание) часто
даёт чуть лучшую обобщающую способность в CNN, но требует подбора.
Формулы, коррекция смещения, почему `Adam(weight_decay=…)` ≠ `AdamW(weight_decay=…)` — см. раздел «4. Адаптивные методы».

```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

# weight decay только для матриц весов (не для bias и параметров нормализации)
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
```

**Расписания темпа обучения**

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
total = n_epochs * len(train_loader); warm = int(0.05 * total)
sched = SequentialLR(opt, [LinearLR(opt, start_factor=0.01, total_iters=warm),       # разогрев 5% шагов
                           CosineAnnealingLR(opt, T_max=total - warm)], milestones=[warm])
# в цикле — sched.step() после каждого optimizer.step() (шаг по батчам, как в ДЗ)
```

</details>

<details>
<summary><b>5. Регуляризация</b></summary>

| Метод | Суть |
|---|---|
| **Weight decay / L2** | штраф за большие веса; для Adam — AdamW |
| **Dropout** | на обучении случайно обнуляем нейроны с вероятностью $p$ (и масштабируем остальные на $\frac{1}{1-p}$); **на тесте отключаем** |
| **Early stopping** | останавливаемся, когда валидационный loss перестал улучшаться |
| **Аугментация данных** | искусственно расширяем выборку (тема [16](../16_cnn/README.md)) |
| **BatchNorm** | побочный регуляризующий эффект (шум статистик батча) |
| Label smoothing | вместо one-hot цели $1 - \varepsilon$ / $\frac{\varepsilon}{K-1}$ — сеть не становится «сверхуверенной» |

**Dropout:** бонус — оценка уверенности в ответе (MC dropout: несколько прогонов с включённым dropout) и анализ
возможности удаления нейронов (сжатие модели). Не забывать `model.train()` / `model.eval()`!

</details>

<details>
<summary><b>6. Обрезка градиентов (gradient clipping)</b></summary>

Если «попали на утёс», градиент огромный и шаг уводит далеко от минимума → **укорачиваем вектор градиента, сохраняя
направление**: если $\|g\| > \theta$, то $g \leftarrow \theta\frac{g}{\|g\|}$. $\theta$ — чуть выше типичной нормы (часто 1; Pascanu, Mikolov, Bengio, 2013).

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

</details>

<details>
<summary><b>7. Нормализация слоёв</b></summary>

**Batch Normalization** (Ioffe, Szegedy, 2015): для каждого признака по мини-батчу — среднее и дисперсия, нормировка,
затем **растяжение и сдвиг** обучаемыми $\gamma$, $\beta$:
$\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$, $y = \gamma\hat x + \beta$.

- **Проблема, которую решает:** градиент показывает, как менять вес, если остальные стоят на месте, но шаг меняет все слои
  сразу — вход слоя зависит от всех весов до него. После BN среднее и разброс выхода слоя задают **два параметра**.
- Авторы объясняли через covariate shift; позже показали — **BN в основном сглаживает loss** (Santurkar et al., 2018).
- Эффект: Inception с BN — **в 14 раз меньше шагов** до той же точности; можно темп ×5…×30; с BN учится даже сигмоида.
- На тесте — накопленные скользящие средние (`model.eval()`).
- **Смещение перед BN не нужно** (`bias=False`): BN вычитает среднее, сдвиг заменяет $\beta$.

| Нормализация | По каким осям (тензор $N\times C\times H\times W$) | Где |
|---|---|---|
| **BatchNorm** | $N, H, W$ для каждого канала | CNN, MLP с нормальным батчем |
| **LayerNorm** | $C, H, W$ для каждого объекта | трансформеры, RNN |
| **InstanceNorm** | $H, W$ для каждого объекта и канала | перенос стиля |
| **GroupNorm** | группы каналов × $H, W$ для каждого объекта | детекция/сегментация с маленькими батчами |

```python
import torch.nn as nn
def make_mlp(d_in, hidden, n_classes, bn=True, p_drop=0.1):
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h, bias=not bn)]           # bias не нужен перед BN
        if bn:
            layers += [nn.BatchNorm1d(h)]
        layers += [nn.ReLU(), nn.Dropout(p_drop)]
        prev = h
    layers += [nn.Linear(prev, n_classes)]                    # выход — логиты
    return nn.Sequential(*layers)                             # НЕ обычный список Python — иначе нет параметров!
```

</details>

<details>
<summary><b>8. ДЗ 1: MLP против бустинга (Covertype)</b></summary>

Задача: по картографическим признакам участка леса (10 числовых + 4 one-hot зоны + 40 one-hot почв) — один из 7 типов леса,
581 012 участков, классы несбалансированы (константа даёт 0.488). Соперник — CatBoost: 0.890 «по умолчанию»
(1000 деревьев глубины 6, ~10 мин на 8 ядрах) и **0.960** посильнее (4000 деревьев глубины 8, ~70 мин). Пять пропусков:
1. **стандартизация** по train-строкам (std признаков от 7.5 до ~1500!);
2. **cross-entropy на логитах** — «в лоб» (ломается на логитах ±1000: inf/NaN) и **устойчиво** через log-sum-exp;
3. **своя MLP** (`nn.Sequential`/`nn.ModuleList`, не список Python);
4. **BatchNorm**: `Linear(bias=False) → BatchNorm1d → ReLU`; темп 3e-3, разогрев 5% шагов + косинус до нуля;
5. **свой Adam и AdamW**: финальная сеть 512×4, 60 эпох, батч 2048, lr 5e-3, weight decay 0.01 → обходит CatBoost за минуты на GPU.

Covertype для сети удобен: полмиллиона строк и всего 54 признака. **На типичной таблице в несколько тысяч строк расклад
обычно обратный** — бустинг лучше. Хитрость скорости: данные целиком на GPU, батч — срез по перемешанным индексам
(стандартный `DataLoader(TensorDataset)` собирает батч циклом Python — секунды на эпоху вместо долей секунды).

</details>

<details>
<summary><b>9. Рецепт отладки обучения</b></summary>

1. Сначала **переобучиться на одном батче** (loss → ~0) — проверка, что модель и цикл корректны.
2. Проверить loss на старте: для $K$ классов при случайной инициализации ≈ $\ln K$.
3. Смотреть кривые train/val loss, норму градиентов, долю «мёртвых» ReLU, распределения активаций.
4. Подобрать **темп обучения** первым (LR range test), затем размер модели и регуляризацию.
5. `model.train()` / `model.eval()`, `optimizer.zero_grad()`, `torch.no_grad()` на валидации — частые источники багов.
6. Фиксировать seed, сравнивать с простым бейзлайном.

</details>

**Математика и выводы**

<details>
<summary><b>1. Инициализация: вывод Xavier и He</b></summary>

Рассмотрим слой $z_i = \sum_{j=1}^{n_{\text{in}}}w_{ij}x_j$ на старте обучения. Предположения (они верны **только на
инициализации**): веса i.i.d. с нулевым средним и дисперсией $\operatorname{Var}w$; входы независимы от весов, i.i.d.,
$\mathbb{E}x = 0$. Тогда

$$
\operatorname{Var}z_i = \sum_j\operatorname{Var}(w_{ij}x_j) = n_{\text{in}}\operatorname{Var}w\cdot\mathbb{E}x^2 .
$$

(Дисперсия произведения независимых величин с нулевым средним: $\operatorname{Var}(wx) = \mathbb{E}w^2\mathbb{E}x^2$.)

**Прямой проход.** Чтобы масштаб сигнала не рос и не падал от слоя к слою, нужно $n_{\text{in}}\operatorname{Var}w = 1$.
**Обратный проход** ($\delta^{(l)} = W^\top\delta^{(l+1)}\odot f'$) аналогично требует $n_{\text{out}}\operatorname{Var}w = 1$.

**Xavier / Glorot** (2010, для tanh/линейных, где около нуля $f' \approx 1$) — компромисс:

$$
\operatorname{Var}w = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad
w \sim U\Big[-\sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}}\Big] \;\text{или}\; \mathcal{N}\Big(0, \tfrac{2}{n_{\text{in}} + n_{\text{out}}}\Big)
$$

(дисперсия $U[-a, a]$ равна $a^2/3$, отсюда $a = \sqrt{6/(n_{\text{in}}+n_{\text{out}})}$).

**Kaiming / He** (2015, для ReLU). Для $x = \operatorname{ReLU}(y)$ с симметричным $y$: $\mathbb{E}x^2 = \frac12\operatorname{Var}y$ — ReLU обнуляет половину
сигнала. Тогда $\operatorname{Var}z^{(l)} = \frac12n_{\text{in}}\operatorname{Var}w\cdot\operatorname{Var}z^{(l-1)}$ и условие сохранения масштаба:

$$
\operatorname{Var}w = \frac{2}{n_{\text{in}}} \qquad(\text{gain} = \sqrt2;\;\; \text{для tanh gain} = 5/3,\; \text{Xavier} \leftrightarrow \text{gain} = 1).
$$

**Нарушение симметрии.** Если все веса слоя одинаковы (например, нули), все нейроны получают одинаковые градиенты и
остаются одинаковыми навсегда — слой из $k$ нейронов работает как один. Поэтому веса — случайные; смещения можно нулями.

**Нормировка входов** (стандартизация признаков) — та же идея для первого слоя; плюс она улучшает обусловленность задачи
(тема [04](../04_linear_models/README.md)). В ДЗ на Covertype std признаков различались от 7.5
(уклон) до ~1500 (расстояние до дорог) — без стандартизации сеть учится плохо.

```python
import torch
for name, init in [("N(0,1)", lambda w: torch.nn.init.normal_(w, 0, 1)),
                   ("N(0,0.01)", lambda w: torch.nn.init.normal_(w, 0, 0.01)),
                   ("Kaiming", lambda w: torch.nn.init.kaiming_normal_(w, nonlinearity="relu"))]:
    x = torch.randn(512, 256)
    for _ in range(30):
        W = torch.empty(256, 256); init(W)
        x = torch.relu(x @ W.T)
    print(name, x.std().item())      # взрыв / затухание / ≈ 1
```

</details>

<details>
<summary><b>2. Мини-батчи: шум градиента и размер батча</b></summary>

Градиент по батчу $g_B = \frac1B\sum_{i\in B}\nabla L_i$ — несмещённая оценка полного градиента; при выборке объектов
дисперсия **каждой координаты** падает как $\frac{\sigma^2}{B}$, стандартное отклонение — как $\frac{1}{\sqrt B}$.

Шаг SGD: $w \leftarrow w - \eta g_B = w - \eta\nabla L - \eta(g_B - \nabla L)$; «шум» шага имеет масштаб $\eta\sigma/\sqrt B$. Для ряда анализов
(SDE-приближение SGD) существенна **«температура»** $\eta/B$: увеличение батча в $k$ раз при росте шага в $k$ раз сохраняет
динамику (**правило линейного масштабирования**, Goyal et al., 2017: ResNet-50 на ImageNet за час на 256 GPU с батчем
8192). Отсюда тезис слайда: «увеличение батча — тот же эффект, что уменьшение темпа обучения» (Smith et al., *Don't
Decay the Learning Rate, Increase the Batch Size*).

Ограничения: выше **критического размера батча** шум уже не мешает, а меньше шагов на эпоху → точность на эпоху падает;
$\eta$ нельзя бесконечно увеличивать — выше порога обучение расходится. Замер из лекции (MLP на CIFAR-10, SGD, 20 эпох):
батчи от 8 до 1024 дают почти одинаковый тест (51–52%), лучший $\eta$ растёт с батчем, пока не упрётся в порог.

</details>

<details>
<summary><b>3. Momentum и метод Нестерова</b></summary>

**Тяжёлый шар (Поляк):**

$$
v_{t+1} = \beta v_t + \nabla L(w_t), \qquad w_{t+1} = w_t - \eta v_{t+1} .
$$

$v$ — экспоненциально взвешенная сумма градиентов. При постоянном градиенте $v \to \frac{g}{1-\beta}$: эффективный шаг $\frac{\eta}{1-\beta}$
(при $\beta = 0.9$ — в 10 раз больше). Вдоль «дна оврага» (согласованные градиенты) скорость накапливается, поперёк
(осциллирующие градиенты) — гасится. На квадратичной функции с числом обусловленности $\kappa$ оптимальный momentum
улучшает число итераций с $O(\kappa)$ до $O(\sqrt\kappa)$. Помогает «проскакивать» седловые точки и мелкие локальные минимумы.

**Нестеров:** градиент берётся в «заглядывающей вперёд» точке: $v_{t+1} = \beta v_t + \nabla L(w_t - \eta\beta v_t)$ — корректирует
разгон до того, как перескочим.

</details>

<details>
<summary><b>4. Адаптивные методы</b></summary>

**Adagrad:** $G_t = G_{t-1} + g_t^2$ (покоординатно), $w \leftarrow w - \frac{\eta}{\sqrt{G_t} + \varepsilon}g_t$ — редкие признаки (малая сумма квадратов)
получают большой шаг. Минус: $G_t$ только растёт → шаг стремится к нулю.

**RMSprop (Hinton, 2012) = Adagrad + скользящее среднее:** $v_t = \beta v_{t-1} + (1 - \beta)g_t^2$, $w \leftarrow w - \frac{\eta}{\sqrt{v_t} + \varepsilon}g_t$.

**Adam (Kingma, Ba, 2014) = RMSprop + momentum + коррекция смещения:**

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t,\quad v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2,\quad
\hat m_t = \frac{m_t}{1 - \beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t},\quad
w_t = w_{t-1} - \eta\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} .
$$

**Откуда коррекция смещения.** $m_0 = 0$, поэтому $m_t = (1-\beta_1)\sum_{s=1}^{t}\beta_1^{t-s}g_s$. Если $\mathbb{E}g_s = \mathbb{E}g$, то

$$
\mathbb{E}m_t = \mathbb{E}g\cdot(1-\beta_1)\sum_{k=0}^{t-1}\beta_1^k = \mathbb{E}g\cdot(1 - \beta_1^t) .
$$

На первых шагах $m_t$ занижено (при $\beta_1 = 0.9$, $t = 1$ — в 10 раз); деление на $1 - \beta_1^t$ это исправляет. Для $v_t$ с $\beta_2 = 0.999$
занижение ещё сильнее (на первом шаге в 1000 раз) — без коррекции первые шаги были бы огромными. Типичные параметры:
$\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\eta = 10^{-3}$. (Исходное доказательство сходимости Adam содержало ошибку —
AMSGrad её исправляет; на практике Adam/AdamW — стандарт.)

**Почему адаптивность помогает:** нормировка на $\sqrt{\hat v}$ приближённо делает шаг каждой координаты одного масштаба
($\approx\eta$) независимо от масштаба её градиентов — это грубая диагональная замена предобуславливания для «вытянутых»
линий уровня.

</details>

<details>
<summary><b>5. Adam + L2 ≠ AdamW</b></summary>

**L2-регуляризация в SGD = weight decay:** градиент $\frac{\lambda}{2}\|w\|^2$ равен $\lambda w$, шаг $w \leftarrow w - \eta(g + \lambda w) = (1 - \eta\lambda)w - \eta g$ —
веса «затухают» в $(1 - \eta\lambda)$ раз за шаг. Для SGD это одно и то же.

**В Adam — разные вещи.** Если $\lambda w$ добавить к градиенту (Adam с L2), он пройдёт через деление на $\sqrt{\hat v}$:
эффективное затухание координаты $\approx\eta\lambda w/\sqrt{\hat v}$ — **веса с большими градиентами почти не штрафуются, с
маленькими — штрафуются сильнее**. **AdamW** (Loshchilov, Hutter, 2017) отделяет (decouples) затухание:

$$
w \leftarrow (1 - \eta\lambda)w - \eta\frac{\hat m}{\sqrt{\hat v} + \varepsilon} .
$$

Эксперимент из ДЗ: две группы весов по 1000, старт 1.0, $\lambda = 1$, $\eta = 10^{-3}$, 1000 шагов, градиент — чистый шум с
$\sigma = 0.001$ и $\sigma = 10$. У AdamW обе группы затухают ровно в $(1 - 10^{-3})^{1000} \approx e^{-1} \approx 0.37$ раза; у Adam+L2
затухание зависит от масштаба шума. Weight decay обычно **не** применяют к смещениям и параметрам нормализации
(одномерным тензорам) — они не отвечают за «сложность» функции.

```python
def adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=True):
    if wd and not decoupled:
        g = g + wd * p                       # Adam + L2: штраф идёт через адаптивную нормировку
    if wd and decoupled:
        p = p * (1 - lr * wd)                # AdamW: затухание отдельно
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
    return p - lr * m_hat / (v_hat ** 0.5 + eps), m, v
```

</details>

<details>
<summary><b>6. Расписания темпа обучения</b></summary>

- **Warmup** (линейный разогрев первые ~1–5% шагов): в начале $\hat v$ оценено по нескольким градиентам и шумно, веса
  случайны, и большие шаги могут «сломать» модель (особенно с нормализациями и в трансформерах).
- **Cosine annealing:** $\eta_t = \eta_{\min} + \frac12(\eta_{\max} - \eta_{\min})\big(1 + \cos\frac{\pi t}{T}\big)$ — плавное затухание до нуля.
- **Step decay**, **ReduceLROnPlateau** (уменьшать при остановке улучшения валидации), **OneCycle**.
- Уменьшение $\eta$ к концу снижает «шумовой пол» SGD (радиус блуждания около минимума $\propto\eta$).

</details>

<details>
<summary><b>7. Dropout</b></summary>

На обучении каждый нейрон независимо **обнуляется** с вероятностью $p$: $\tilde h = h\odot\frac{m}{1-p}$, $m_j \sim \text{Bernoulli}(1-p)$.
Масштаб $\frac{1}{1-p}$ (**inverted dropout**) сохраняет математическое ожидание: $\mathbb{E}\tilde h_j = h_j$, поэтому на тесте dropout
просто **отключают** (`model.eval()`).

Интерпретации: (1) обучение экспоненциально большого ансамбля «прореженных» сетей с общими весами, тест — приближённое
усреднение; (2) нейроны не могут полагаться на конкретных соседей (меньше ко-адаптации). Для линейной регрессии dropout
на входах эквивалентен адаптивной L2-регуляризации. **MC dropout:** оставить dropout на тесте и усреднить несколько
прогонов — оценка неопределённости (бонус со слайда).

</details>

<details>
<summary><b>8. Batch Normalization</b></summary>

Для мини-батча $\lbrace x_1, \dots, x_B\rbrace$ (покоординатно):

$$
\mu_B = \frac1B\sum_i x_i,\quad \sigma_B^2 = \frac1B\sum_i(x_i - \mu_B)^2,\quad \hat x_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}},\quad y_i = \gamma\hat x_i + \beta .
$$

- Среднее и разброс выхода слоя задают **два параметра** $\gamma, \beta$, а не все веса до него — шаг оптимизатора меньше меняет
  распределение входов следующих слоёв.
- **Train / eval:** на обучении — статистики батча; на тесте — **скользящие средние** `running_mean/var`, накопленные при
  обучении (`model.eval()` обязателен; батч из одного объекта иначе не нормализовать).
- **Смещение перед BN не нужно:** $\operatorname{BN}(Wx + b) = \operatorname{BN}(Wx)$ — постоянный сдвиг вычитается со средним; его роль
  выполняет $\beta$. Поэтому `Linear(bias=False) → BatchNorm1d → ReLU` (пропуск 4 в ДЗ).
- Зачем: позволяет больший темп обучения и сигмоиды (Inception с BN догнал исходный за **14 раз меньше шагов**:
  2.1 млн против 31 млн, вариант с темпом ×30). Авторы объясняли это «internal covariate shift», но Santurkar et al. (2018)
  показали, что главное — **сглаживание ландшафта функции потерь** (градиенты более предсказуемы, меньше константа Липшица).
- Минусы: зависимость от размера батча (маленькие батчи — шумные статистики), разное поведение train/eval, неудобна для
  RNN и последовательностей переменной длины.

**Другие нормализации** (тензор $N\times C\times H\times W$): **LayerNorm** — по всем признакам одного объекта (трансформеры, RNN;
не зависит от батча); **InstanceNorm** — по $H, W$ одного канала одного объекта (перенос стиля); **GroupNorm** — по группам
каналов (детекция/сегментация с маленькими батчами). Для MLP ($N\times$ признаки) важны BN и LN.

</details>

<details>
<summary><b>9. Обрезка градиента</b></summary>

Если норма градиента больше порога $\theta$, укорачиваем вектор, **сохраняя направление**:

$$
g \leftarrow g\cdot\min\Big(1, \frac{\theta}{\|g\|}\Big) .
$$

Помогает, когда попали на «утёс» функции потерь (резкий скачок градиента, типично для RNN, Pascanu et al., 2013). $\theta$ берут
чуть выше типичной нормы градиента (часто 1) — обрезаются только редкие выбросы; если $\theta$ меньше типичной нормы,
обрезается каждый шаг, и это просто меньший темп обучения. `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему нельзя инициализировать нулями?</summary>

симметрия: все нейроны слоя получат одинаковые градиенты.

</details>

<details>
<summary>Зачем коррекция смещения в Adam?</summary>

$m_0=v_0=0$, первые оценки занижены в $1-\beta^t$ раз.

</details>

<details>
<summary>Чем AdamW отличается от Adam с `weight_decay`?</summary>

в Adam L2 проходит через $1/\sqrt{\hat v}$ и штрафует «не те» веса.

</details>

<details>
<summary>Почему перед BatchNorm `bias=False`?</summary>

BN вычитает среднее; сдвиг заменяет $\beta$.

</details>

<details>
<summary>Забыли `model.eval()`</summary>

dropout активен, BN берёт статистики батча — метрики на тесте «прыгают».

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- He: ReLU обнуляет половину сигнала, $\mathbb{E}x^2=\frac12\operatorname{Var}y$ → нужно $n_{in}\operatorname{Var}w=2$.
- Шум мини-батча $\propto1/\sqrt B$; «температура» SGD $\sim\eta/B$ → правило линейного масштабирования $\eta\propto B$ (до критического батча).
- Inverted dropout: делим на $1-p$ на обучении, чтобы $\mathbb{E}\tilde h=h$ и на тесте ничего не менять.
- BN, по Santurkar et al., в основном сглаживает ландшафт loss; LayerNorm не зависит от батча — для трансформеров и RNN.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
import torch, torch.nn as nn
torch.manual_seed(0)

# Инициализация: std активаций после 30 слоёв ReLU
for name, init in [("N(0, 1)", lambda w: nn.init.normal_(w, 0, 1.0)),
                   ("N(0, 0.01)", lambda w: nn.init.normal_(w, 0, 0.01)),
                   ("Kaiming (He)", lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))]:
    h = torch.randn(512, 256)
    for _ in range(30):
        W = torch.empty(256, 256); init(W); h = torch.relu(h @ W.T)
    print(f"{name:13} std активаций на 30-м слое = {h.std().item():.3e}")
```

Результат запуска:

```text
N(0, 1)       std активаций на 30-м слое = 1.984e+31
N(0, 0.01)    std активаций на 30-м слое = 3.334e-29
Kaiming (He)  std активаций на 30-м слое = 3.287e-01
```

```python
# Оптимизаторы на одной задаче (two moons, MLP), 300 шагов полным батчем
from sklearn.datasets import make_moons
Xm, ym = make_moons(1000, noise=0.2, random_state=0)
Xm, ym = torch.tensor(Xm, dtype=torch.float32), torch.tensor(ym, dtype=torch.float32)

def train(opt_fn, bn=False, steps=300):
    torch.manual_seed(0)
    layers = [nn.Linear(2, 64, bias=not bn)] + ([nn.BatchNorm1d(64)] if bn else []) + [nn.ReLU(), nn.Linear(64, 1)]
    model = nn.Sequential(*layers); opt = opt_fn(model.parameters())
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(model(Xm).squeeze(1), ym)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

for name, fn in [("SGD lr=0.05", lambda p: torch.optim.SGD(p, lr=0.05)),
                 ("SGD+momentum", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
                 ("Adam lr=1e-2", lambda p: torch.optim.Adam(p, lr=1e-2)),
                 ("AdamW lr=1e-2", lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=0.01))]:
    print(f"{name:14} final loss = {train(fn):.3f}")
print(f"SGD lr=0.05 + BatchNorm  final loss = {train(lambda p: torch.optim.SGD(p, lr=0.05), bn=True):.3f}")
```

Результат запуска:

```text
SGD lr=0.05    final loss = 0.286
SGD+momentum   final loss = 0.161
Adam lr=1e-2   final loss = 0.080
AdamW lr=1e-2  final loss = 0.080
SGD lr=0.05 + BatchNorm  final loss = 0.277
```

```python
# Adam + L2 ≠ AdamW: две группы весов, градиент — шум разного масштаба, decay λ=1, lr=1e-3, 1000 шагов
def run(decoupled):
    p = torch.ones(2, 1000); m = torch.zeros_like(p); v = torch.zeros_like(p)
    scale = torch.tensor([[1e-3], [10.0]]); gen = torch.Generator().manual_seed(0)
    for t in range(1, 1001):
        g = torch.randn(p.shape, generator=gen) * scale
        if not decoupled: g = g + 1.0 * p               # L2 внутри градиента
        else: p = p * (1 - 1e-3 * 1.0)                  # AdamW: отдельное затухание
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        p = p - 1e-3 * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8)
    return p.mean(1).tolist()
print("Adam+L2 (малый, большой градиент):", [round(x, 3) for x in run(False)])
print("AdamW   (малый, большой градиент):", [round(x, 3) for x in run(True)], "  ожидание e^-1 ≈ 0.368")
```

Результат запуска:

```text
Adam+L2 (малый, большой градиент): [0.258, 0.906]
AdamW   (малый, большой градиент): [0.369, 0.368]   ожидание e^-1 ≈ 0.368
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Тонкости обучения](https://education.yandex.ru/handbook/ml/article/tonkosti-obucheniya) · [Оптимизация в ML](https://education.yandex.ru/handbook/ml/article/optimizaciya-v-ml)
- [CS231n: Neural Networks Part 2 (init, BN, regularization)](https://cs231n.github.io/neural-networks-2/) · [Part 3 (learning, optimizers)](https://cs231n.github.io/neural-networks-3/)
- [Why Momentum Really Works (distill)](https://distill.pub/2017/momentum/) · [UvA DL: optimization](https://uvadlc.github.io/lectures/dec2020/lecture3.1.pdf) · [12 видов dropout](https://towardsdatascience.com/12-main-dropout-methods-mathematical-and-visual-explanation-58cdc2112293) · [All about normalization](https://becominghuman.ai/all-about-normalization-6ea79e70894b)
- [Andrej Karpathy — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- [DL2 HSE, неделя 2: GPU, узкие места DL, бенчмаркинг](https://github.com/thecrazymage/DL2_HSE/tree/main/week_02)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Повторение.pptx` (день 11), слайды 16–35: затухание/взрыв, нормировки, инициализация, мини-батчи, momentum, Adagrad/RMSprop/Adam, dropout, клиппинг, BatchNorm и другие нормализации. (Лекции 2 курса DL в папке нет — её материал повторён здесь.)
- `task_for_students.ipynb` — ДЗ 1 «MLP против бустинга на табличных данных».

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Glorot X., Bengio Y. *Understanding the difficulty of training deep feedforward neural networks* (2010)
- He K. et al. *Delving Deep into Rectifiers* (2015)
- Kingma D., Ba J. *Adam* (2014) — https://arxiv.org/abs/1412.6980; Loshchilov I., Hutter F. *Decoupled Weight Decay Regularization* (2017)
- Srivastava N. et al. *Dropout* (2014); Gal Y., Ghahramani Z. *Dropout as a Bayesian Approximation* (2016)
- Ioffe S., Szegedy C. *Batch Normalization* (2015) — https://arxiv.org/abs/1502.03167; Santurkar S. et al. *How Does Batch Normalization Help Optimization?* (2018)
- Goyal P. et al. *Accurate, Large Minibatch SGD* (2017); Smith S. et al. *Don't Decay the Learning Rate, Increase the Batch Size* (2017)
- Pascanu R., Mikolov T., Bengio Y. *On the difficulty of training recurrent neural networks* (2013)
- [Why Momentum Really Works (distill.pub)](https://distill.pub/2017/momentum/) · [deeplearning.ai: Initialization](https://www.deeplearning.ai/ai-notes/initialization/)

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL: повторение (обзор курса)
- **Домашние задания:** [DL, ДЗ 1: MLP против бустинга (задание)](../../materials/homework/09-15_DL_ДЗ_1_MLP_против_бустинга.ipynb)
- **Видео и внешние курсы:** [Stanford CS231n — CNN for Visual Recognition](https://cs231n.stanford.edu/)

</details>

---

[← 14. Нейронные сети и backpropagation](../14_neural_networks_backprop/README.md) · [Оглавление курса](../../README.md) · [16. Свёрточные сети (CNN) →](../16_cnn/README.md)
