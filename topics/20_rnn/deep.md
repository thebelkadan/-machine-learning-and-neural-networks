# 20. Рекуррентные сети — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

---

## 1. BPTT для vanilla RNN

Модель: $a_t = W_{xh}x_t + W_{hh}h_{t-1} + b$, $h_t = \tanh(a_t)$, $\hat y_t = \operatorname{softmax}(W_{hy}h_t + c)$, потери
$\mathcal L = \sum_{t=1}^{T}\mathcal L_t$. Обозначим $\delta_t = \partial\mathcal L/\partial h_t$ — полный градиент по состоянию (через
текущий выход **и** через все будущие шаги), $g_t = \partial\mathcal L/\partial a_t$.

$h_t$ влияет на $\mathcal L_t$ напрямую и на будущее через $a_{t+1}$:

$$
\delta_t = \frac{\partial\mathcal L_t}{\partial h_t} + W_{hh}^{\top}g_{t+1},\qquad
g_t = \delta_t\odot\big(1 - h_t^2\big),\qquad \delta_{T+1}\equiv 0 ,
$$

где $\partial\mathcal L_t/\partial h_t = W_{hy}^{\top}(\hat y_t - y_t)$ для кросс-энтропии с softmax. Рекурсия идёт **назад во времени**.
Веса общие, поэтому градиенты по ним — суммы по шагам:

$$
\frac{\partial\mathcal L}{\partial W_{hh}} = \sum_{t=1}^{T} g_t\,h_{t-1}^{\top},\qquad
\frac{\partial\mathcal L}{\partial W_{xh}} = \sum_{t=1}^{T} g_t\,x_t^{\top},\qquad
\frac{\partial\mathcal L}{\partial b} = \sum_{t=1}^{T} g_t ,\qquad
\frac{\partial\mathcal L}{\partial W_{hy}} = \sum_{t=1}^{T}(\hat y_t - y_t)\,h_t^{\top} .
$$

**Truncated BPTT** с окном $k$: рекурсию для $\delta$ обрывают через $k$ шагов (в коде — `h = h.detach()` между кусками). Получается
смещённая оценка градиента: вклады зависимостей длиннее $k$ просто отсутствуют.

---

## 2. Произведение якобианов: затухание и взрыв

Вклад шага $k$ в градиент потерь на шаге $T$ проходит через

$$
\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T}\frac{\partial h_t}{\partial h_{t-1}}
= \prod_{t=k+1}^{T}\operatorname{diag}\big(\tanh'(a_t)\big)\,W_{hh} .
$$

**Линейный случай** ($\tanh$ убрали): $\partial h_T/\partial h_k = W_{hh}^{T-k}$. Если $W_{hh} = Q\Lambda Q^{-1}$, то
$W_{hh}^{n} = Q\Lambda^{n}Q^{-1}$ — компонента вдоль собственного вектора умножается на $\lambda^{n}$:

- $\lvert\lambda\rvert < 1$: экспоненциальное убывание ($0.9^{50}\approx 0.005$) — **затухание**;
- $\lvert\lambda\rvert > 1$: экспоненциальный рост ($1.1^{50}\approx 117$) — **взрыв**;
- собственные значения случайны — дисперсия нарастает с длиной.

**С нелинейностью** (Pascanu, Mikolov, Bengio, 2013). Пусть $\lvert\tanh'\rvert\le\gamma$ ($\gamma = 1$ для $\tanh$, $\frac14$ для
сигмоиды), $\sigma_{\max}$ — наибольшее сингулярное число $W_{hh}$. Тогда

$$
\Big\lVert\frac{\partial h_T}{\partial h_k}\Big\rVert \le \prod_{t=k+1}^{T}\lVert\operatorname{diag}(\tanh'(a_t))\rVert\,\lVert W_{hh}\rVert
\le (\gamma\,\sigma_{\max})^{T-k} .
$$

Значит, $\sigma_{\max} < 1/\gamma$ — **достаточное** условие затухания; $\rho(W_{hh}) > 1/\gamma$ — **необходимое** (но не достаточное)
условие взрыва. Насыщение $\tanh$ ($\tanh'\to0$) только усиливает затухание.

**Почему это мешает учиться.** Полный градиент

$$
\frac{\partial\mathcal L_T}{\partial W} = \sum_{k\le T}\frac{\partial\mathcal L_T}{\partial h_T}\,\frac{\partial h_T}{\partial h_k}\,\frac{\partial^{+}h_k}{\partial W}
$$

(где $\partial^{+}$ — «непосредственная» производная шага $k$) — сумма вкладов разной «дальности». При затухании дальние слагаемые
экспоненциально малы по сравнению с ближними, и оптимизатор видит только короткие зависимости. При взрыве один шаг уводит веса
далеко («обрыв» на поверхности потерь).

**Ортогональная инициализация:** у ортогональной $W$ все $\lvert\lambda\rvert = 1$ и $\lVert Wv\rVert = \lVert v\rVert$ — в линейном режиме
норма сигнала сохраняется на любой глубине. Нелинейность всё равно немного сжимает ($\tanh'\le1$), но старт из «нейтральной» точки.

---

## 3. Клиппинг градиента

Клиппинг по норме (Pascanu et al., 2013):

$$
g \leftarrow g\cdot\min\Big(1,\;\frac{\tau}{\lVert g\rVert}\Big).
$$

Направление сохраняется, ограничивается только длина шага — защита от «обрывов»: в узкой области поверхности потерь градиент
огромен, и шаг SGD без ограничения выбросил бы параметры далеко. Клиппинг **по значению** (`clip_grad_value_`) режет каждую координату
отдельно и меняет направление. `torch.nn.utils.clip_grad_norm_` возвращает норму **до** клиппинга — её полезно логировать: частые
срабатывания означают слишком большой learning rate. Против затухания клиппинг бесполезен.

---

## 4. Почему LSTM не затухает: путь через cell state

$$
c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t \quad\Rightarrow\quad
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) + \underbrace{(\ldots)}_{\text{через } h_{t-1}\text{ в } f_t,\,i_t,\,\tilde c_t} .
$$

По **прямому** пути

$$
\frac{\partial c_T}{\partial c_k}\Big|_{\text{прямой}} = \prod_{t=k+1}^{T}\operatorname{diag}(f_t)
$$

нет ни умножения на матрицу весов, ни производной $\tanh$, только поэлементные гейты. Если сеть выучила $f\approx1$ для нужных
координат, градиент течёт почти без потерь. Это аналог шортката ResNet: $c_t = c_{t-1} + \Delta_t$ при $f=1$.

- **LSTM 1997** не имела forget-гейта: $f\equiv1$, $\partial c_T/\partial c_k = I$ — «constant error carousel». Gers et al. (2000)
  добавили forget-гейт: без сброса память на длинном потоке переполняется.
- **Время памяти.** При постоянном $f$ вклад шага $k$ затухает как $f^{\,T-k}$; характерное время $\tau = -1/\ln f\approx 1/(1-f)$:

| $b_f$ | $f = \sigma(b_f)$ | $\tau$, шагов | $f^{50}$ |
|---|---|---|---|
| 0 | 0.50 | 1.4 | $9\cdot10^{-16}$ |
| 1 | 0.73 | 3.2 | $1.6\cdot10^{-7}$ |
| 3 | 0.953 | 21 | 0.09 |
| 5 | 0.993 | 149 | 0.71 |

Отсюда рекомендация $b_f = 1$ (Gers 2000, Jozefowicz 2015) и больше для длинных зависимостей. **Chrono-инициализация** (Tallec &
Ollivier, 2018): $b_f\sim\ln\mathcal U[1, T_{\max}-1]$, $b_i = -b_f$. Тогда $f = \frac{u}{1+u}$, $\tau\approx 1+u$ — времена памяти
нейронов равномерно покрывают диапазон до $T_{\max}$. В примере шпаргалки при $T=50$: LSTM с $b_f=0$ — accuracy 0.51 (угадывание),
с $b_f=3$ — 0.995.

---

## 5. GRU как обучаемый «протекающий» интегратор

Протекающий нейрон (leaky unit): $h_t = \alpha h_{t-1} + (1-\alpha)\tanh(Wx_t + Uh_{t-1})$ с фиксированным $\alpha$ — это экспоненциальное
сглаживание, $\partial h_t/\partial h_{t-1}\approx\alpha I$. GRU (в конвенции PyTorch и Cho et al.)

$$
h_t = z_t\odot h_{t-1} + (1 - z_t)\odot\tilde h_t,\qquad
\tilde h_t = \tanh\big(W_{in}x_t + b_{in} + r_t\odot(W_{hn}h_{t-1} + b_{hn})\big)
$$

делает $\alpha$ **обучаемым, своим для каждого нейрона и шага**: $z_t\to1$ — «держать память», градиент идёт через $\operatorname{diag}(z_t)$.
Reset-гейт $r_t\to0$ позволяет кандидату «забыть» прошлое и смотреть только на вход (начало новой фразы). Bias update-гейта, сдвинутый
к «помнить» ($z\approx1$), даёт тот же эффект, что $b_f$ у LSTM.

---

## 6. Число параметров

Для входа размерности $D$ и скрытого $H$ один «блок» — это $W\in\mathbb R^{H\times D}$, $U\in\mathbb R^{H\times H}$ и bias:

| Ячейка | Блоков | Параметров (PyTorch: два bias) | $D=4,\ H=5$ |
|---|---|---|---|
| RNN | 1 | $H(D+H) + 2H$ | 55 |
| GRU | 3 (r, z, n) | $3\,[H(D+H) + 2H]$ | 165 |
| LSTM | 4 (i, f, g, o) | $4\,[H(D+H) + 2H]$ | 220 |

Bidirectional удваивает число параметров слоя; во втором и следующих слоях вход имеет размер $H\cdot n_{\text{dirs}}$. В «бумажной»
формулировке bias один — $H(D+H) + H$ на блок; второй bias в PyTorch избыточен (сумма двух векторов), но остался ради совместимости с cuDNN.

---

## 7. Teacher forcing = максимум правдоподобия, и откуда exposure bias

Цепное правило — точная факторизация: $p(y_{1:T}) = \prod_t p(y_t\mid y_{<t})$. Логарифм правдоподобия обучающей последовательности

$$
\log p_\theta(y_{1:T}) = \sum_{t=1}^{T}\log p_\theta\big(y_t\mid y^{\text{true}}_{<t}\big)
$$

в каждом слагаемом условен на **истинный** префикс — это и есть teacher forcing. Значит, teacher forcing — не эвристика, а обычный
MLE, и все шаги считаются за один проход (в трансформере — параллельно).

**Exposure bias:** на обучении префиксы берутся из распределения данных, на генерации — из распределения модели. Ошибка на шаге $t$
сдвигает вход шага $t+1$ туда, где модель не обучалась, и ошибки накапливаются. В теории имитационного обучения (Ross & Bagnell, 2010)
это даёт рост суммарной ошибки как $O(T^{2}\varepsilon)$ вместо $O(T\varepsilon)$ при обучении на собственных траекториях.

**Scheduled sampling** (Bengio et al., 2015): на шаге обучения $i$ подаём истинный токен с вероятностью $\varepsilon_i$:

$$
\varepsilon_i = \max(\varepsilon_{\min}, k - ci)\ \text{(линейно)},\qquad
\varepsilon_i = k^{\,i},\ k<1\ \text{(экспоненциально)},\qquad
\varepsilon_i = \frac{k}{k + e^{i/k}},\ k\ge1\ \text{(обратная сигмоида)} .
$$

Оговорка: Huszár (2015) показал, что такая цель — несостоятельная оценка: в пределе она не восстанавливает истинное распределение.
На практике в больших моделях exposure bias лечат стратегиями декодирования и дообучением на своих генерациях (RL-подходы, темы 24–25).

---

## 8. Dropout во времени

Если на каждом шаге брать **новую** маску на рекуррентной связи, шум накапливается за $T$ шагов и разрушает память; поэтому Zaremba
(2014) применяет dropout только к нерекуррентным связям. **Вариационный dropout** (Gal & Ghahramani, 2016) берёт **одну** маску на всю
последовательность:

$$
h_t = \tanh\big(W(x_t\odot m_x) + U(h_{t-1}\odot m_h)\big),\qquad m_x, m_h\ \text{фиксированы для всех } t ,
$$

что соответствует приближённому байесовскому выводу с одним сэмплом весов на последовательность (тот же сэмпл $W, U$ на всех шагах).
**Recurrent dropout** (Semeniuta, 2016): $c_t = f_t\odot c_{t-1} + i_t\odot d(\tilde c_t)$ — шумит только добавка, память $c_{t-1}$ не
трогается. **Zoneout:** $h_t = m\odot h_{t-1} + (1-m)\odot h_t^{\text{new}}$ — часть нейронов случайно «замирает», градиент через них
проходит как через тождество.

---

## 9. Нормировки

**BatchNorm** в RNN неудобен: статистики должны быть **свои на каждом шаге** (Cooijmans, 2016), а для длин больше, чем в обучении,
статистик нет; применение только к «вертикальным» связям (Laurent, 2015) помогает слабо. **LayerNorm** (Ba et al., 2016) нормирует
по признакам **внутри объекта и шага**:

$$
\operatorname{LN}(a) = \gamma\odot\frac{a - \mu}{\sqrt{s^{2} + \epsilon}} + \beta,\qquad \mu = \frac1H\sum_j a_j,\quad s^{2} = \frac1H\sum_j(a_j - \mu)^2 ,
$$

не зависит ни от батча, ни от длины. Применяется к предактивациям гейтов; так же устроены нормировки в трансформерах (тема 23).

---

## 10. Сложность: RNN против self-attention

| | RNN-слой | Self-attention |
|---|---|---|
| Вычисления на слой | $O(T\cdot H^{2})$ | $O(T^{2}\cdot d)$ |
| Последовательных операций | $O(T)$ | $O(1)$ |
| Длина пути между позициями $i, j$ | $O(\lvert i-j\rvert)$ | $O(1)$ |
| Генерация одного токена | $O(H^{2})$, память $O(H)$ | $O(T\cdot d)$, KV-кэш $O(T\cdot d)$ |

Обучение RNN не параллелится по времени, а дальние зависимости идут длинным путём (затухание). Зато генерация дешёвая и с
постоянной памятью — отсюда интерес к линейным рекуррентностям и state-space-моделям, у которых переход **линеен** по $h$, поэтому
рекурсию можно развернуть в свёртку или параллельный scan для обучения.

---

## 11. Временные ряды: почему log и сезонные разности

Ряд авиапассажиров — **мультипликативный**: $y_t = \text{тренд}_t\cdot\text{сезон}_t\cdot\varepsilon_t$ (амплитуда сезонности растёт с
уровнем). Логарифм делает его аддитивным: $\log y_t = \log\text{тренд}_t + \log\text{сезон}_t + \log\varepsilon_t$. Сезонная разность

$$
d_t = \log y_t - \log y_{t-12}\approx\frac{y_t - y_{t-12}}{y_{t-12}}
$$

— годовой темп роста: сезонность сокращается, линейный (в логарифмах) тренд превращается в константу, и ряд становится примерно
**стационарным**. Модели остаётся предсказать $\hat d_t$, а прогноз восстанавливается как $\hat y_t = y_{t-12}\,e^{\hat d_t}$.

- **Сезонный наивный** прогноз — это $\hat d_t = 0$; «сезонный × годовой рост» — $\hat d_t = d_{t-1}$. Нейросеть на $d$ учит поправку к
  ним, и её стоит сравнивать именно с ними.
- На сыром ряде тестовые значения (до 622) выходят за диапазон train (до 467): модель с насыщающимися активациями не может
  экстраполировать, и нормировка, подогнанная на train, не спасает.
- $e^{\hat d}$ от несмещённого прогноза логарифма даёт **медиану**, а не среднее; поправка при нормальных остатках — множитель $e^{\sigma^{2}/2}$.
- Масштабно-независимая метрика **MASE**: MAE модели, делённое на MAE сезонного наивного прогноза на train; MASE < 1 — лучше наивного.

---

## Ссылки

- Pascanu, Mikolov, Bengio. [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063), 2013.
- Hochreiter, Schmidhuber. Long short-term memory, 1997 · Gers, Schmidhuber, Cummins. Learning to forget, 2000.
- Tallec, Ollivier. [Can recurrent neural networks warp time?](https://arxiv.org/abs/1804.11188) (chrono-инициализация), 2018.
- Bengio et al. [Scheduled Sampling](https://arxiv.org/abs/1506.03099), 2015 · Huszár. [How (not) to train your generative model](https://arxiv.org/abs/1511.05101), 2015 · Ross, Bagnell. Efficient reductions for imitation learning, 2010.
- Gal, Ghahramani. [A theoretically grounded application of dropout in RNNs](https://arxiv.org/abs/1512.05287) · Semeniuta et al. [Recurrent dropout without memory loss](https://arxiv.org/abs/1603.05118) · Krueger et al. [Zoneout](https://arxiv.org/abs/1606.01305).
- Ba, Kiros, Hinton. [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016 · Cooijmans et al. [Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025).
- Le, Jaitly, Hinton. [A simple way to initialize recurrent networks of ReLUs (IRNN)](https://arxiv.org/abs/1504.00941), 2015.
- Hyndman, Athanasopoulos. [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — сезонные разности, наивные прогнозы, MASE.
