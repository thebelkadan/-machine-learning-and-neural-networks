# 25. Дообучение LLM — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

---

## 1. Почему низкий ранг работает

**Внутренняя размерность** (Aghajanyan et al., 2020): будем дообучать не все $D$ параметров, а только $z\in\mathbb R^{d}$ в случайном
подпространстве: $\theta = \theta_0 + Pz$, $P\in\mathbb R^{D\times d}$ — случайная проекция. Наименьшее $d$, при котором достигается 90%
качества полного дообучения ($d_{90}$), для предобученных моделей оказывается порядка сотен–тысяч при $D\sim10^8$–$10^9$ и **уменьшается** с ростом
размера и длительности предобучения. То есть дообучение «живёт» в очень маломерном подпространстве — LoRA предлагает
структурированный вариант: малый ранг каждой матрицы.

Эксперименты LoRA: даже $r = 1$–$4$ на $W_Q, W_V$ GPT-3 почти не уступает $r = 64$; подпространства, выученные при разных $r$, сильно
пересекаются в верхних направлениях; $\Delta W$ усиливает направления, которые в $W_0$ были **не главными** (не повторяет его топ-сингулярные векторы).

---

## 2. Градиенты LoRA и инициализация

$h = W_0x + s\,BAx$, $s = \alpha/r$. Пусть $G = \partial\mathcal L/\partial h\,x^{\top}$ — градиент по «полному» $\Delta W$ (для батча — сумма). Тогда

$$
\frac{\partial\mathcal L}{\partial B} = s\,G\,A^{\top},\qquad \frac{\partial\mathcal L}{\partial A} = s\,B^{\top}G .
$$

- При $B = 0$: $\partial\mathcal L/\partial A = 0$, но $\partial\mathcal L/\partial B = sGA^{\top}\ne0$ — на первом шаге учится $B$, затем оживает и $A$.
- Если бы $A = B = 0$, оба градиента нулевые навсегда (седловая точка). Если обе случайные — $\Delta W\ne0$ на старте, модель испорчена.
- Обновление эффективного $\Delta W$ за шаг SGD с lr $\eta$ (с точностью до второго порядка):
  $\Delta(BA) \approx -\eta s\,(GA^{\top}A + BB^{\top}G)$ — это проекция полного градиента $G$ на подпространства строк $A$ и столбцов $B$.
  Отсюда интуиция: LoRA — градиентный спуск в изменяющемся подпространстве ранга $r$.
- **Масштаб:** при инициализации $A$ с дисперсией $\sim1/k$ величина $\lVert\Delta W\rVert$ после обучения растёт с $r$; множитель $s = \alpha/r$
  компенсирует это для первых шагов. rsLoRA (Kalajdzievski, 2023) показывает, что для стабильности при больших $r$ правильно $s = \alpha/\sqrt r$.

**Счёт параметров** ($r(d_{\text{in}} + d_{\text{out}})$ на матрицу):
- SmolLM2-135M (семинар), $r = 16$, слой: $q$ 576→576: $16\cdot1152 = 18\,432$; $k$, $v$ 576→192: по $16\cdot768 = 12\,288$; $o$: 18 432 — итого
  61 440 на слой × 30 слоёв = **1 843 200** (1.35%).
- GPT-2, `c_attn` 768→2304, $r = 8$: $8\cdot3072 = 24\,576$ × 12 слоёв = **294 912** (0.24%).
- Prompt tuning: $m\cdot d = 20\cdot768 = 15\,360$. Prefix tuning: $m\cdot L\cdot2\cdot d = 10\cdot12\cdot2\cdot768 = 184\,320$ (K и V в каждом слое).

---

## 3. Память: откуда 16 байт на параметр

Смешанная точность с Adam на параметр:

| Что | Байт |
|---|---|
| веса bf16 (для прямого/обратного прохода) | 2 |
| градиенты bf16 | 2 |
| мастер-копия весов fp32 | 4 |
| первый момент Adam $m$ fp32 | 4 |
| второй момент Adam $v$ fp32 | 4 |
| **итого** | **16** |

Плюс **активации**: $\approx$ $L\cdot T\cdot B\cdot d\cdot c$ байт ($c$ — десятки, зависит от архитектуры); **gradient checkpointing** хранит только входы
блоков и пересчитывает остальное на обратном проходе — память $O(\sqrt L)$ или $O(1)$ на слой вместо $O(L)$ ценой ~+33% вычислений.

LoRA: база — 2 байта на параметр (bf16) без градиентов и состояний; адаптер — 16 байт на свой параметр. QLoRA: база — 4 бита + шкалы.

---

## 4. DPO из RLHF

Задача выравнивания с KL-регуляризацией:

$$
\max_\pi\ \mathbb E_{x,\,y\sim\pi}\big[r(x, y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$

Для фиксированного $x$ это $\sum_y\pi(y)\big[r(y) - \beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\big]$ при $\sum_y\pi(y) = 1$. Перепишем как
$-\beta\,\mathrm{KL}\big(\pi\,\|\,\tfrac1Z\pi_{\text{ref}}e^{r/\beta}\big) + \beta\log Z$, откуда оптимум (распределение Гиббса):

$$
\pi^{*}(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\Big(\frac{r(x, y)}{\beta}\Big)
\quad\Longleftrightarrow\quad
r(x, y) = \beta\log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x).
$$

Модель предпочтений **Брэдли–Терри**: $p(y_w\succ y_l\mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$. В разности наград **$\log Z(x)$ сокращается** —
не нужно вычислять сумму по всем текстам. Подставляя параметризацию $\pi_\theta$ вместо $\pi^{*}$ и максимизируя правдоподобие пар:

$$
\mathcal L_{\mathrm{DPO}}(\theta) = -\mathbb E_{(x, y_w, y_l)}\log\sigma\Big(\underbrace{\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\hat r_\theta(x, y_w)} - \underbrace{\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\hat r_\theta(x, y_l)}\Big).
$$

«Языковая модель — тайно reward-модель». **Градиент:**

$$
\nabla_\theta\mathcal L_{\mathrm{DPO}} = -\beta\,\mathbb E\Big[\sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big]:
$$

повышаем вероятность предпочтённого, понижаем отвергнутого, с весом тем большим, чем сильнее неявная награда **ошибается** в порядке пары.
$\log\pi(y\mid x) = \sum_t\log\pi(y_t\mid x, y_{<t})$ — сумма по токенам ответа (как в SFT, с маской промпта).

---

## 5. Reward-модель и PPO

**RM:** $\mathcal L = -\mathbb E\log\sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$ — логистическая регрессия на разностях; если на промпт $K$ ответов,
используют все $\binom K2$ пар (InstructGPT: $K = 4$…$9$, пары из одного промпта — в одном батче, чтобы не переобучаться). Награда определена
с точностью до константы — её нормируют.

**PPO для LM:** эпизод — генерация ответа; награда на последнем токене $r_\phi(x, y)$, а на каждом токене — KL-штраф:

$$
R_t = -\beta\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)} + [t = T]\,r_\phi(x, y).
$$

Политика обновляется обрезанной целью $\mathbb E\big[\min(\rho_tA_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t)\big]$, $\rho_t = \pi_\theta/\pi_{\text{old}}$, преимущества
$A_t$ — по GAE с value-моделью (critic). **GRPO** выкидывает critic: для $G$ ответов на промпт $A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r)}$ —
групповой бейзлайн, плюс KL к референсу в лоссе.

---

## 6. Квантование NF4

**Absmax-квантование блока** $w\in\mathbb R^{64}$: $c = \max_i\lvert w_i\rvert$, $q_i = \operatorname{round}(w_i/c\cdot\text{уровень})$. Для 4 бит есть 16 значений; вопрос —
где их расставить. Если веса $\sim\mathcal N(0, \sigma^2)$, то после деления на absmax они лежат в $[-1, 1]$ с известным распределением, и
**равновероятные** ячейки (каждая получает $1/16$ весов) — квантили нормального распределения:

$$
q_i = \frac12\Big(Q_{\mathcal N}\Big(\frac{i}{2^4 + 1}\Big) + Q_{\mathcal N}\Big(\frac{i + 1}{2^4 + 1}\Big)\Big),
$$

нормированные в $[-1, 1]$, с точным нулём (асимметрично: 8 положительных, 7 отрицательных уровней + 0). Это минимизирует ожидаемую
ошибку при равномерном использовании кодов — «информационно оптимальный» тип для нормальных данных.

**Биты на параметр:** 4 бита + шкала fp32 на блок 64 = $4 + 32/64 = 4.5$. **Double quantization:** шкалы квантуются в 8 бит блоками по 256 с
одной fp32-шкалой второго уровня: $8/64 + 32/(64\cdot256)\approx0.127$ бита вместо 0.5 — экономия ≈ 0.37 бита на параметр (≈ 3 ГБ на 65B модели).

Матричное умножение: $Y = X\cdot\mathrm{dequant}(W_{\mathrm{NF4}}, c) + s\,XA^{\top}B^{\top}$ в bf16; градиенты по $X$ проходят через
деквантованный вес, по $W$ не считаются.

---

## Ссылки

- Aghajanyan et al. [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255), 2020.
- Hu et al. [LoRA](https://arxiv.org/abs/2106.09685), 2021 · Kalajdzievski. [rsLoRA](https://arxiv.org/abs/2312.03732), 2023 · Hayou et al. [LoRA+](https://arxiv.org/abs/2402.12354), 2024.
- Dettmers et al. [QLoRA](https://arxiv.org/abs/2305.14314), 2023 · Dettmers et al. [LLM.int8()](https://arxiv.org/abs/2208.07339), 2022.
- Rafailov et al. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023 · Ouyang et al. [InstructGPT](https://arxiv.org/abs/2203.02155), 2022 · Schulman et al. [PPO](https://arxiv.org/abs/1707.06347), 2017 · Shao et al. [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300), 2024.
- Chen et al. [Training Deep Nets with Sublinear Memory Cost (checkpointing)](https://arxiv.org/abs/1604.06174), 2016 · Rajbhandari et al. [ZeRO](https://arxiv.org/abs/1910.02054), 2019 (16 байт на параметр и их шардирование).
