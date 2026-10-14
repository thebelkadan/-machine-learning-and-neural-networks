# 15. Обучение нейросетей — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

---

## 1. Инициализация: вывод Xavier и He

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
(тема [04](../04_linear_models/deep.md#5-градиентный-спуск-и-sgd)). В ДЗ на Covertype std признаков различались от 7.5
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

---

## 2. Мини-батчи: шум градиента и размер батча

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

---

## 3. Momentum и метод Нестерова

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

## 4. Адаптивные методы

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

---

## 5. Adam + L2 ≠ AdamW

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

---

## 6. Расписания темпа обучения

- **Warmup** (линейный разогрев первые ~1–5% шагов): в начале $\hat v$ оценено по нескольким градиентам и шумно, веса
  случайны, и большие шаги могут «сломать» модель (особенно с нормализациями и в трансформерах).
- **Cosine annealing:** $\eta_t = \eta_{\min} + \frac12(\eta_{\max} - \eta_{\min})\big(1 + \cos\frac{\pi t}{T}\big)$ — плавное затухание до нуля.
- **Step decay**, **ReduceLROnPlateau** (уменьшать при остановке улучшения валидации), **OneCycle**.
- Уменьшение $\eta$ к концу снижает «шумовой пол» SGD (радиус блуждания около минимума $\propto\eta$).

## 7. Dropout

На обучении каждый нейрон независимо **обнуляется** с вероятностью $p$: $\tilde h = h\odot\frac{m}{1-p}$, $m_j \sim \text{Bernoulli}(1-p)$.
Масштаб $\frac{1}{1-p}$ (**inverted dropout**) сохраняет математическое ожидание: $\mathbb{E}\tilde h_j = h_j$, поэтому на тесте dropout
просто **отключают** (`model.eval()`).

Интерпретации: (1) обучение экспоненциально большого ансамбля «прореженных» сетей с общими весами, тест — приближённое
усреднение; (2) нейроны не могут полагаться на конкретных соседей (меньше ко-адаптации). Для линейной регрессии dropout
на входах эквивалентен адаптивной L2-регуляризации. **MC dropout:** оставить dropout на тесте и усреднить несколько
прогонов — оценка неопределённости (бонус со слайда).

## 8. Batch Normalization

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

---

## 9. Обрезка градиента

Если норма градиента больше порога $\theta$, укорачиваем вектор, **сохраняя направление**:

$$
g \leftarrow g\cdot\min\Big(1, \frac{\theta}{\|g\|}\Big) .
$$

Помогает, когда попали на «утёс» функции потерь (резкий скачок градиента, типично для RNN, Pascanu et al., 2013). $\theta$ берут
чуть выше типичной нормы градиента (часто 1) — обрезаются только редкие выбросы; если $\theta$ меньше типичной нормы,
обрезается каждый шаг, и это просто меньший темп обучения. `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.

---

## Ссылки

- Glorot X., Bengio Y. *Understanding the difficulty of training deep feedforward neural networks* (2010)
- He K. et al. *Delving Deep into Rectifiers* (2015)
- Kingma D., Ba J. *Adam* (2014) — https://arxiv.org/abs/1412.6980; Loshchilov I., Hutter F. *Decoupled Weight Decay Regularization* (2017)
- Srivastava N. et al. *Dropout* (2014); Gal Y., Ghahramani Z. *Dropout as a Bayesian Approximation* (2016)
- Ioffe S., Szegedy C. *Batch Normalization* (2015) — https://arxiv.org/abs/1502.03167; Santurkar S. et al. *How Does Batch Normalization Help Optimization?* (2018)
- Goyal P. et al. *Accurate, Large Minibatch SGD* (2017); Smith S. et al. *Don't Decay the Learning Rate, Increase the Batch Size* (2017)
- Pascanu R., Mikolov T., Bengio Y. *On the difficulty of training recurrent neural networks* (2013)
- [Why Momentum Really Works (distill.pub)](https://distill.pub/2017/momentum/) · [deeplearning.ai: Initialization](https://www.deeplearning.ai/ai-notes/initialization/)

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb)
