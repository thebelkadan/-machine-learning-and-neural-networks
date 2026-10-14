# 22. Seq2seq и внимание — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

---

## 1. Условная языковая модель и функция потерь

По цепному правилу (без всяких допущений) $p(y\mid x) = \prod_{t=1}^{T_y}p(y_t\mid y_{<t}, x)$. Для обучающей пары максимизируем
логарифм правдоподобия, то есть минимизируем

$$
\mathcal L(\theta) = -\sum_{t=1}^{T_y}\log p_\theta(y_t\mid y_{<t}, x) = \sum_{t=1}^{T_y}\mathrm{CE}\big(\text{one-hot}(y_t),\ p_\theta(\cdot\mid y_{<t}, x)\big).
$$

Каждый множитель — softmax над словарём, поэтому лосс — сумма кросс-энтропий по позициям; условие $y_{<t}$ берётся из данных
(teacher forcing). Нормировка: средний лосс на токен $\bar{\mathcal L}$ связан с **перплексией** $\mathrm{PPL} = e^{\bar{\mathcal L}}$ —
«эффективное число равновероятных вариантов» на шаге ([тема 24](../24_language_models_gpt/README.md)).

---

## 2. Внимание как дифференцируемый поиск по словарю

Хеш-таблица по запросу $q$ возвращает значение $v_i$ с ключом $k_i = q$. «Мягкая» версия возвращает смесь всех значений с весами
по сходству:

$$
c(q) = \sum_i\frac{\exp(\operatorname{score}(q, k_i)/\tau)}{\sum_j\exp(\operatorname{score}(q, k_j)/\tau)}\,v_i .
$$

При $\tau\to0$ softmax становится argmax — точный поиск ближайшего ключа (недифференцируемо); при $\tau\to\infty$ — простое среднее.
Это **регрессия Надарая–Ватсона** с ядром $K(q, k) = e^{\operatorname{score}(q,k)}$: оценка в точке — взвешенное ядром среднее
наблюдений. Отличие нейросетевого внимания — **ключи, запросы и значения обучаемы**.

**Градиент.** Пусть $e_k$ — оценки, $a = \operatorname{softmax}(e)$, $c = \sum_k a_kv_k$. Якобиан softmax
$\partial a_i/\partial e_k = a_i(\delta_{ik} - a_k)$, откуда

$$
\frac{\partial c}{\partial e_k} = \sum_i v_i\,a_i(\delta_{ik} - a_k) = a_k\,(v_k - c),\qquad \frac{\partial c}{\partial v_k} = a_k I .
$$

Оценка $e_k$ растёт, если $v_k$ отличается от текущего контекста $c$ в «полезную» сторону ($\langle\nabla_c\mathcal L, v_k - c\rangle < 0$),
и сила сигнала пропорциональна $a_k$: позиции с почти нулевым весом почти не получают градиента (проблема насыщения, §3).

---

## 3. Зачем делить на корень из d

Пусть компоненты $q$ и $k$ независимы, со средним 0 и дисперсией 1. Тогда

$$
\mathbb E[q^{\top}k] = 0,\qquad \operatorname{Var}(q^{\top}k) = \sum_{i=1}^{d}\operatorname{Var}(q_ik_i) = \sum_{i=1}^{d}\mathbb E[q_i^2]\,\mathbb E[k_i^2] = d .
$$

При $d = 512$ оценки имеют стандартное отклонение $\approx22.6$: разница соседних логитов в десятки единиц делает softmax почти
one-hot, а его якобиан $\operatorname{diag}(a) - aa^{\top}\to0$ — градиенты исчезают, и внимание перестаёт учиться (застревает на
случайном выборе). Деление на $\sqrt d$ возвращает дисперсию к 1 при любой размерности. Это же объясняет наблюдение Vaswani et al.:
без масштабирования additive-внимание выигрывало у dot-product при больших $d$ — $\tanh$ в additive само ограничивает масштаб.

---

## 4. Additive vs multiplicative: стоимость и выразительность

- **Dot / general:** все оценки — одно матричное произведение $E = HWS^{\top}\in\mathbb R^{T_y\times T_x}$, стоимость $O(T_yT_xd)$ на
  высокооптимизированном GEMM. General — билинейная форма; при $W = I$ — dot.
- **Additive:** $e_{tk} = v^{\top}\tanh(W_1h_t + W_2s_k)$. Проекции $W_1h_t$ и $W_2s_k$ считаются один раз ($O(T_yd^2 + T_xd^2)$), но затем
  нужен тензор $T_y\times T_x\times d'$ с нелинейностью — нельзя свести к одному матричному умножению, больше памяти. Зато это MLP —
  оценка нелинейна по паре $(h, s)$.

Поэтому в трансформере выбрано scaled dot-product: быстро и параллельно, а выразительность добирается обучаемыми проекциями $W_Q, W_K$
и несколькими «головами».

---

## 5. Модель Bahdanau полностью

Энкодер — bidirectional RNN, аннотации $h_j = [\overrightarrow h_j; \overleftarrow h_j]$ содержат информацию о всём предложении с
фокусом вокруг слова $j$. Декодер с состоянием $s_i$:

$$
e_{ij} = v_a^{\top}\tanh(W_as_{i-1} + U_ah_j),\qquad \alpha_{ij} = \frac{\exp e_{ij}}{\sum_k\exp e_{ik}},\qquad c_i = \sum_j\alpha_{ij}h_j,
$$

$$
s_i = f(s_{i-1}, y_{i-1}, c_i),\qquad p(y_i\mid y_{<i}, x) = g(y_{i-1}, s_i, c_i).
$$

$\alpha_{ij}$ — вероятность того, что целевое слово $y_i$ «выровнено» с исходным $x_j$ (мягкое выравнивание вместо скрытой
переменной в статистическом MT). Вектор контекста свой на каждом шаге — узкого горла нет.

**Luong:** $h_t = \mathrm{RNN}(h_{t-1}, [y_{t-1}; \tilde h_{t-1}])$ (input feeding), $a_t = \operatorname{softmax}(\operatorname{score}(h_t, \bar h_s))$,
$c_t = \sum_s a_{ts}\bar h_s$, $\tilde h_t = \tanh(W_c[c_t; h_t])$, $p(y_t) = \operatorname{softmax}(W_s\tilde h_t)$. **Local:** позиция
$p_t = T_x\cdot\sigma(v_p^{\top}\tanh(W_ph_t))$ и веса умножаются на $\exp\big(-\frac{(s - p_t)^2}{2\sigma^2}\big)$, $\sigma = D/2$ в окне $[p_t - D, p_t + D]$.

---

## 6. Длина пути градиента

Без внимания влияние $x_1$ на выход $y_t$ проходит через $T_x - 1$ шагов энкодера и $t$ шагов декодера — произведение $\sim T_x + t$
якобианов, со всеми проблемами из [темы 20](../20_rnn/deep.md#2-произведение-якобианов-затухание-и-взрыв). С вниманием есть путь
$x_1\to h_1\to c_t\to y_t$ длины $O(1)$: $\partial c_t/\partial h_1 = \alpha_{t1}I + (\text{через веса})$. Пока вес не нулевой,
градиент доходит без затухания. В L1-примере это видно как разница точностей на длинах 13–16: 0.00 против 0.92.

Ёмкостный аргумент про узкое горло: фиксированный вектор $\mathbb R^{H}$ должен кодировать вход любой длины; энтропия входа растёт
линейно с длиной (для случайных цифр — $T\log_2 10$ бит), а сеть, обученная с ограниченной точностью, хранит в $H$ числах ограниченный
объём. С вниманием «память» — все $T_x$ состояний, она растёт вместе со входом.

---

## 7. Beam search: что именно ищем

Цель вывода — $y^{*} = \arg\max_y\log p(y\mid x) = \arg\max_y\sum_t\log p(y_t\mid y_{<t}, x)$. Жадный поиск оптимален только если
решения на шагах независимы. Beam search ширины $k$: на шаге $t$ из $k\lvert V\rvert$ продолжений оставляем $k$ с наибольшей суммой;
время $O(T k\lvert V\rvert)$ (на деле выбор top-$k$ из каждого распределения, потом top-$k$ из $k^2$), память — $k$ состояний декодера.

**Ошибки поиска vs ошибки модели.** Если поиск не нашёл $y^{*}$ — ошибка поиска; если $y^{*}$ — плохой перевод — ошибка модели.
Stahlberg & Byrne (2019) точным поиском показали, что для многих предложений **мода** NMT-модели — **пустой перевод**:
$p(\langle\mathrm{eos}\rangle\mid x)$ на первом шаге больше произведения десятков вероятностей < 1. То есть небольшой beam работает
«благодаря» ошибкам поиска, а увеличение $k$ ухудшает BLEU (beam search curse). Отсюда нормировки:

$$
\mathrm{score}(y) = \frac{\log p(y\mid x)}{\mathrm{lp}(y)} + \mathrm{cp}(x; y),\qquad
\mathrm{lp}(y) = \frac{(5 + \lvert y\rvert)^{\alpha}}{(5 + 1)^{\alpha}},\qquad
\mathrm{cp} = \beta\sum_{i=1}^{T_x}\log\Big(\min\Big(\sum_j a_{ij},\ 1\Big)\Big)
$$

(GNMT, Wu et al. 2016): coverage penalty штрафует слова входа, на которые суммарно почти не смотрели, — борьба с пропусками.

---

## 8. BLEU подробно

Для гипотез $\lbrace c\rbrace$ и референсов $\lbrace r\rbrace$ корпуса модифицированная точность n-грамм

$$
p_n = \frac{\sum_{c}\sum_{g\in\text{n-gram}(c)}\min\big(\mathrm{Count}_c(g),\ \max_{r}\mathrm{Count}_r(g)\big)}{\sum_{c}\sum_{g\in\text{n-gram}(c)}\mathrm{Count}_c(g)} .
$$

«Обрезка» (clipping) не даёт «the the the …» набрать высокую $p_1$. Точность не наказывает за **пропуски**, а полноту при нескольких
референсах определить трудно, поэтому вводится штраф за краткость по суммарным длинам $c$ (гипотезы) и $r$ (ближайшие по длине
референсы): $\mathrm{BP} = 1$ при $c > r$, иначе $e^{1 - r/c}$. Итог

$$
\mathrm{BLEU} = \mathrm{BP}\cdot\exp\Big(\sum_{n=1}^{4}w_n\log p_n\Big),\qquad w_n = \tfrac14 .
$$

Геометрическое среднее требует ненулевых $p_n$ — на корпусе это выполняется, на одном предложении 4-граммы часто не совпадают, отсюда
сглаживание (Chen & Cherry, 2014). BLEU — точностная метрика на поверхностных n-граммах: синонимы, морфология и порядок слов
наказываются; для русского полезнее chrF:
$\mathrm{chrF}_\beta = (1 + \beta^2)\frac{\mathrm{chrP}\cdot\mathrm{chrR}}{\beta^2\,\mathrm{chrP} + \mathrm{chrR}}$, $\beta = 2$ (полнота важнее).

---

## Ссылки

- Bahdanau, Cho, Bengio. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2015.
- Luong, Pham, Manning. [Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025), 2015.
- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017 (§3.2.1 — масштабирование).
- Wu et al. [Google's NMT System](https://arxiv.org/abs/1609.08144), 2016 (length и coverage penalty).
- Stahlberg, Byrne. [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://arxiv.org/abs/1908.10090), 2019 · Koehn, Knowles. [Six Challenges for NMT](https://arxiv.org/abs/1706.03872), 2017.
- Papineni et al. [BLEU](https://aclanthology.org/P02-1040/), 2002 · Chen, Cherry. [Smoothing techniques for sentence-level BLEU](https://aclanthology.org/W14-3346/), 2014 · Popović. [chrF](https://aclanthology.org/W15-3049/), 2015.
- Bishop, Bishop. Deep Learning: Foundations and Concepts, гл. 12 (внимание как ядерная регрессия) · d2l.ai: [Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html).
