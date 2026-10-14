# 23. Трансформер и BERT — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

---

## 1. Self-attention эквивариантен к перестановкам

Пусть $P$ — матрица перестановки строк ($P^{\top}P = I$). Для входа $PX$: $Q' = PQ$, $K' = PK$, $V' = PV$, и

$$
\operatorname{softmax}\Big(\frac{PQK^{\top}P^{\top}}{\sqrt{d_k}}\Big)PV = P\,\operatorname{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)P^{\top}P\,V = P\cdot\operatorname{Attention}(Q, K, V),
$$

так как построчный softmax коммутирует с перестановкой строк и столбцов. FFN, LayerNorm и residual действуют на каждый токен отдельно —
тоже эквивариантны. Значит, весь энкодер без позиций — функция **множества** токенов: переставили вход — так же переставился выход.
В L1-примере задача «сколько раз встречается токен» (не зависит от порядка) решается без PE (0.993), а «какой токен слева» — нет (0.239).

---

## 2. Синусоидальные позиции: сдвиг = поворот

Пара координат $(2i, 2i+1)$ с частотой $\omega_i = 10000^{-2i/d}$: $\mathrm{PE}_p^{(i)} = (\sin\omega_ip,\ \cos\omega_ip)$. По формулам синуса и
косинуса суммы

$$
\begin{pmatrix}\sin\omega_i(p+k)\\ \cos\omega_i(p+k)\end{pmatrix} =
\begin{pmatrix}\cos\omega_ik & \sin\omega_ik\\ -\sin\omega_ik & \cos\omega_ik\end{pmatrix}
\begin{pmatrix}\sin\omega_ip\\ \cos\omega_ip\end{pmatrix},
$$

то есть $\mathrm{PE}_{p+k} = R_k\,\mathrm{PE}_p$, где $R_k$ — блочно-диагональный поворот, **не зависящий от $p$**. Относительный сдвиг выражается
линейным преобразованием, которое модель может выучить. Скалярное произведение зависит только от расстояния:

$$
\langle\mathrm{PE}_p, \mathrm{PE}_q\rangle = \sum_i\big(\sin\omega_ip\sin\omega_iq + \cos\omega_ip\cos\omega_iq\big) = \sum_i\cos\omega_i(p - q).
$$

**RoPE** (Su et al., 2021) применяет поворот прямо к запросам и ключам: $\tilde q_m = R_mq_m$, $\tilde k_n = R_nk_n$. Тогда
$\tilde q_m^{\top}\tilde k_n = q_m^{\top}R_m^{\top}R_nk_n = q_m^{\top}R_{n-m}k_n$ — оценка внимания зависит от содержимого и **относительной**
позиции $n - m$; ничего не прибавляется к эмбеддингам, информация о позиции есть в каждом слое.

---

## 3. Параметры и вычисления

Слой энкодера с размерностью $d$, FFN $4d$:

| Часть | Параметров |
|---|---|
| $W_Q, W_K, W_V, W_O$ | $4d^2 + 4d$ |
| FFN $d\to4d\to d$ | $8d^2 + 5d$ |
| 2 × LayerNorm | $4d$ |
| **итого** | $\approx 12d^2$ |

**BERT-Base:** $12\cdot12\cdot768^2\approx85$M в слоях + эмбеддинги $30522\cdot768\approx23.4$M + позиции $512\cdot768\approx0.4$M + сегменты + pooler
$\approx110$M. Значит, у небольших моделей заметная доля параметров — в словаре.

**FLOPs на токен** при прямом проходе: $\approx 2\cdot12d^2 = 24d^2$ на матричные умножения слоя плюс $\approx4Td$ на $QK^{\top}$ и $AV$. Для
всей модели (decoder-only, $N$ параметров без эмбеддингов) — знаменитое $\approx2N$ FLOPs на токен вперёд и $\approx6N$ на обучение
(вперёд + назад, назад ≈ 2 × вперёд). Квадратичная часть $4Td$ сравнима с линейной при $T\approx6d$; память под матрицу весов
$O(hT^2)$ на слой — именно её избегает FlashAttention, считая softmax блоками с онлайн-нормировкой.

**Ранговое ограничение головы.** Оценки $XW_QW_K^{\top}X^{\top}$ — билинейная форма ранга $\le d_k$. Одна голова с $d_k = d/h$ не может
выразить произвольную матрицу сходства; несколько голов дают сумму разных «взглядов» ранга $d_k$ при той же стоимости.

---

## 4. Pre-LN vs Post-LN

**LayerNorm:** для вектора токена $x\in\mathbb R^d$

$$
\mu = \frac1d\sum_jx_j,\quad \sigma^2 = \frac1d\sum_j(x_j - \mu)^2,\quad \mathrm{LN}(x) = \gamma\odot\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta;\qquad
\mathrm{RMSNorm}(x) = \gamma\odot\frac{x}{\sqrt{\frac1d\sum_jx_j^2 + \epsilon}} .
$$

Статистики по признакам одного токена → не зависят от батча и длины.

**Post-LN:** $x_{l+1} = \mathrm{LN}(x_l + F_l(x_l))$. Нормировка стоит **на** residual-пути, и градиент к ранним слоям каждый раз проходит
через якобиан LN. **Pre-LN:** $x_{l+1} = x_l + F_l(\mathrm{LN}(x_l))$, выход $= \mathrm{LN}(x_L)$; тогда

$$
x_L = x_0 + \sum_{l<L}F_l(\mathrm{LN}(x_l)),\qquad \frac{\partial x_L}{\partial x_l} = I + \sum_{m\ge l}\frac{\partial F_m}{\partial x_l},
$$

тождественное слагаемое даёт «чистый» путь градиента, как в ResNet. Xiong et al. (2020) показали при инициализации: у post-LN норма
градиента параметров последнего слоя $O(d\sqrt{\ln d})$ — не убывает с глубиной, поэтому большой lr в начале разрушителен и **нужен warm-up**;
у pre-LN — $O\big(d\sqrt{\ln d / L}\big)$, обучение устойчиво и без длинного прогрева. Цена pre-LN: норма $x_l$ растёт с глубиной, вклад
поздних слоёв относительно уменьшается; post-LN при удачном обучении иногда даёт чуть лучшее качество.

---

## 5. Label smoothing

Цель $q_k = (1 - \epsilon)[k = y] + \frac{\epsilon}{K - 1}[k\ne y]$ (вариант с $\frac{\epsilon}{K}$ на все классы тоже распространён). Лосс

$$
\mathrm{CE}(q, p) = -\sum_kq_k\log p_k = (1 - \epsilon)\,\big(-\log p_y\big) + \frac{\epsilon}{K - 1}\sum_{k\ne y}\big(-\log p_k\big).
$$

Минимум по $p$ — при $p = q$ (неравенство Гиббса), то есть оптимальные логиты **конечны**: $z_y - z_k = \ln\frac{(1 - \epsilon)(K - 1)}{\epsilon}$
(для $\epsilon = 0.1$, $K = 37000$ — около 12.7). Без сглаживания оптимум $z_y - z_k\to\infty$: модель бесконечно наращивает уверенность,
веса растут, калибровка портится. Минимальное значение $\mathrm{CE} = H(q) > 0$, поэтому удобнее KL: $\mathrm{KL}(q\,\|\,p) = \mathrm{CE}(q, p) - H(q)$
равна 0 в оптимуме (как в `nn.KLDivLoss` семинара). Перплексия на one-hot целях ухудшается (модель намеренно оставляет массу другим
токенам), но BLEU и accuracy растут. В PyTorch: `nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=pad)`.

---

## 6. Расписание Noam и зачем прогрев

$\mathrm{lr}(s) = d^{-0.5}\min(s^{-0.5}, s\,w^{-1.5})$: две ветви равны при $s = w$, пик $\mathrm{lr}_{\max} = d^{-0.5}w^{-0.5}$ (для $d = 512$,
$w = 4000$ — $\approx7\cdot10^{-4}$); до пика рост линейный, после — убывание $\propto1/\sqrt s$. Множитель $d^{-0.5}$ уменьшает lr у широких моделей.

**Почему Adam нужен прогрев** (Liu et al., 2019). Адаптивный шаг $\propto1/\sqrt{\hat v_t}$, где $\hat v_t$ — экспоненциальное среднее $g^2$.
На первых шагах оно оценено по считанным градиентам: если приблизить $g\sim\mathcal N(0, \sigma^2)$, то $1/\sqrt{\hat v_t}$ имеет огромную
(при $t\to1$ — неограниченную) дисперсию. Отдельные координаты получают гигантские шаги. Прогрев гасит это, пока статистики не
накопятся; RAdam делает то же аналитически — выключает адаптивность, пока оценка дисперсии ненадёжна, и затем вводит поправку
$r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}{(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}}$, $\rho_\infty = \frac{2}{1 - \beta_2} - 1$.

---

## 7. MLM как задача и её цена

Пусть $\mathcal M$ — множество выбранных позиций (15%), $\tilde x$ — испорченный вход. Цель

$$
\mathcal L_{\text{MLM}} = -\mathbb E_{x,\mathcal M}\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x).
$$

Это **шумоподавляющий автокодировщик**: восстановить исходник по испорченной копии. В отличие от LM, предсказания условно независимы
по замаскированным позициям («New [MASK] [MASK]» → «New York City» и «New Delhi Sea» одинаково допустимы по отдельности) — поэтому
BERT не генерирует связный текст. **Эффективность:** обучающий сигнал только от 15% токенов против 100% у LM; ELECTRA исправляет это —
маленький генератор подменяет токены, а основная модель для **каждого** токена решает «оригинал или подмена» (сигнал со всех позиций,
то же качество при ~1/4 вычислений).

Вероятности при выборе позиции $i$: `[MASK]` — 0.8, случайный токен — 0.1, исходный — 0.1. Доля входа, видимая как `[MASK]`, —
$0.15\cdot0.8 = 12\%$; доля «подменённых» токенов — $1.5\%$, «нетронутых, но предсказываемых» — $1.5\%$.

---

## 8. BERT для QA: span-предсказание

Выходы $h_i\in\mathbb R^{d}$ для токенов абзаца, два обучаемых вектора $S, E$:

$$
p^{\text{start}}_i = \frac{e^{S^{\top}h_i}}{\sum_je^{S^{\top}h_j}},\qquad p^{\text{end}}_i = \frac{e^{E^{\top}h_i}}{\sum_je^{E^{\top}h_j}},\qquad
\mathcal L = -\log p^{\text{start}}_{s^{*}} - \log p^{\text{end}}_{e^{*}} .
$$

На выводе ищется $\arg\max_{i\le j\le i+L_{\max}}\big(S^{\top}h_i + E^{\top}h_j\big)$ за $O(T\cdot L_{\max})$; для вопросов без ответа (SQuAD 2.0)
сравнивают со скором $S^{\top}h_{\mathrm{CLS}} + E^{\top}h_{\mathrm{CLS}}$ и порогом.

---

## 9. NER: метрика по сущностям

seqeval собирает из BIO-тегов **спаны** (тип, начало, конец) и считает

$$
P = \frac{\lvert\text{предсказанные}\cap\text{истинные}\rvert}{\lvert\text{предсказанные}\rvert},\quad
R = \frac{\lvert\text{предсказанные}\cap\text{истинные}\rvert}{\lvert\text{истинные}\rvert},\quad F_1 = \frac{2PR}{P + R},
$$

где совпадение — точное (тип и обе границы). Пример семинара с логитами $[[7.12, -0.81], [0.23, 8.1], [5.3, 0.5]]$ и целью $[0, 1, 1]$:
предсказание $[0, 1, 0]$ → token accuracy $2/3$; сущностей предсказано 1 и она верна ($P = 1$), истинных 2 ($R = 0.5$), $F_1 = 2/3$.

---

## Ссылки

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017 · Rush. [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/).
- Xiong et al. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), 2020 · Zhang, Sennrich. [RMSNorm](https://arxiv.org/abs/1910.07467), 2019.
- Liu et al. [On the Variance of the Adaptive Learning Rate and Beyond (RAdam)](https://arxiv.org/abs/1908.03265), 2019.
- Su et al. [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021 · Press et al. [ALiBi](https://arxiv.org/abs/2108.12409), 2021.
- Bhojanapalli et al. [Low-Rank Bottleneck in Multi-head Attention](https://arxiv.org/abs/2002.07028), 2020 · Dao et al. [FlashAttention](https://arxiv.org/abs/2205.14135), 2022 · Kaplan et al. [Scaling Laws](https://arxiv.org/abs/2001.08361) (оценка $6N$ FLOPs/токен), 2020.
- Müller, Kornblith, Hinton. [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629), 2019.
- Devlin et al. [BERT](https://arxiv.org/abs/1810.04805), 2018 · Clark et al. [ELECTRA](https://arxiv.org/abs/2003.10555), 2020.
