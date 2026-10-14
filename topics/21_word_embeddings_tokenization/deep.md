# 21. Представления текста — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

---

## 1. TF-IDF: откуда логарифм

Если документ выбирается случайно, $p(w\in d) = \mathrm{df}(w)/N$, и $\text{idf}(w) = \log\frac{N}{\mathrm{df}(w)} = -\log p(w\in d)$ —
**собственная информация** события «слово есть в документе»: редкое слово сообщает много, слово из каждого документа — ноль.
TF-IDF — «частота × информативность». Варианты sklearn:

$$
\text{idf}_{\text{smooth}}(w) = \ln\frac{1+N}{1+\mathrm{df}(w)} + 1,\qquad \text{tf}_{\text{sublinear}} = 1 + \ln n_{wd},\qquad x_d \leftarrow \frac{x_d}{\lVert x_d\rVert_2} .
$$

«+1» в числителе и знаменателе — как будто есть документ со всеми словами (нет деления на 0); «+1» снаружи — слово из каждого
документа не обнуляется. L2-нормировка делает скалярное произведение косинусом и убирает зависимость от длины текста.

---

## 2. Skip-gram: градиент полного softmax

$p(o\mid c) = \dfrac{\exp(u_o^{\top}v_c)}{\sum_{w\in V}\exp(u_w^{\top}v_c)}$, потери пары $\ell = -\log p(o\mid c) = -u_o^{\top}v_c + \log\sum_w\exp(u_w^{\top}v_c)$.

$$
\frac{\partial\ell}{\partial v_c} = -u_o + \sum_{w\in V}\frac{\exp(u_w^{\top}v_c)}{\sum_{w'}\exp(u_{w'}^{\top}v_c)}\,u_w = -u_o + \sum_{w\in V}p(w\mid c)\,u_w = \mathbb E_{w\sim p(\cdot\mid c)}[u_w] - u_o ,
$$

$$
\frac{\partial\ell}{\partial u_w} = \big(p(w\mid c) - [w = o]\big)\,v_c .
$$

Смысл: центральный вектор тянется к наблюдаемому контексту $u_o$ и отталкивается от «ожидаемого моделью» среднего; каждый вектор
контекста получает поправку, пропорциональную ошибке вероятности (как в softmax-регрессии: «предсказание минус цель»). Стоимость —
$O(\lvert V\rvert d)$ на **каждую** пару (слово, контекст) — при $\lvert V\rvert = 10^5$–$10^6$ и миллиардах пар неприемлемо.

**Зачем два вектора.** С одним вектором на слово скор пары $(w, w)$ равен $\lVert v_w\rVert^2$ — максимизация вероятности контекста
раздувала бы норму и заставляла слово предсказывать само себя, хотя слова редко стоят рядом с собой.

---

## 3. Hierarchical softmax

Слова — листья бинарного дерева (Хаффмана: частые слова ближе к корню). Каждому внутреннему узлу $n$ — вектор $u_n$; путь к слову
$w$ — узлы $n_1,\ldots,n_L$ с направлениями $d_j\in\lbrace +1, -1\rbrace$:

$$
p(w\mid c) = \prod_{j=1}^{L}\sigma\big(d_j\,u_{n_j}^{\top}v_c\big).
$$

Так как $\sigma(x) + \sigma(-x) = 1$, в каждом узле вероятность делится между ветками, и $\sum_w p(w\mid c) = 1$ автоматически.
Стоимость — $O(\log\lvert V\rvert\cdot d)$, для частых слов путь ещё короче.

---

## 4. Negative sampling

Вместо «какое слово из $\lvert V\rvert$» — бинарный вопрос «эта пара из корпуса ($D = 1$) или из шума ($D = 0$)»,
$p(D = 1\mid w, c) = \sigma(u_c^{\top}v_w)$. На пару $(c, o)$ и $k$ негативов $n_i\sim P_n$:

$$
\ell = -\log\sigma(s_o) - \sum_{i=1}^{k}\log\sigma(-s_{n_i}),\qquad s_x = u_x^{\top}v_c .
$$

Используя $\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x) = \sigma(-x)$:

$$
\frac{\partial\ell}{\partial v_c} = -\big(1 - \sigma(s_o)\big)u_o + \sum_i\sigma(s_{n_i})\,u_{n_i},\qquad
\frac{\partial\ell}{\partial u_o} = -\big(1 - \sigma(s_o)\big)v_c,\qquad
\frac{\partial\ell}{\partial u_{n_i}} = \sigma(s_{n_i})\,v_c .
$$

Стоимость $O(kd)$. Обновляются только $k+1$ векторов контекста — разреженные градиенты.

**Почему $P_n(w)\propto\mathrm{count}(w)^{3/4}$.** Степень < 1 сглаживает распределение: частоты $0.9 / 0.09 / 0.01$ превращаются в
$0.825 / 0.147 / 0.028$ — редкие слова чаще становятся негативами, частые — реже. Значение 3/4 подобрано эмпирически.

**Связь с NCE.** Noise-Contrastive Estimation учит отличать данные от шума с поправкой $\log kP_n(w)$ в логите и в пределе даёт
нормированную модель; negative sampling — её упрощение без поправки: хорошие векторы, но не корректные вероятности.

---

## 5. SGNS неявно раскладывает матрицу PMI (Levy & Goldberg, 2014)

Сгруппируем потери по парам. Пусть $\#(w,c)$ — число совместных появлений, $\#(w)$, $\#(c)$ — частоты, $\lvert D\rvert$ — число пар,
негативы из униграммного $P_n(c) = \#(c)/\lvert D\rvert$. Вклад пары $(w,c)$ в целевую функцию (максимизируем) при $x = u_c^{\top}v_w$:

$$
\ell(x) = \#(w,c)\log\sigma(x) + k\,\#(w)\,\frac{\#(c)}{\lvert D\rvert}\,\log\sigma(-x).
$$

Если размерность достаточно велика, чтобы каждое $x$ подбиралось независимо, приравняем производную к нулю:

$$
\#(w,c)\,\sigma(-x) = k\,\frac{\#(w)\#(c)}{\lvert D\rvert}\,\sigma(x)\ \Rightarrow\ e^{x} = \frac{\#(w,c)\,\lvert D\rvert}{\#(w)\,\#(c)}\cdot\frac1k
\ \Rightarrow\ u_c^{\top}v_w = \mathrm{PMI}(w,c) - \log k .
$$

То есть SGNS — это (взвешенная) низкоранговая факторизация **сдвинутой PMI**-матрицы. Отсюда практичная альтернатива: PPMI + SVD
(как в L1-примере), а «тонкости» word2vec — сглаживание контекстов $3/4$, subsampling, динамическое окно — переносятся и на
счётные методы и объясняют большую часть разницы в качестве. Симметричное разбиение $W = U_r\Sigma_r^{1/2}$, $C = V_r\Sigma_r^{1/2}$
обычно лучше, чем $U_r\Sigma_r$.

---

## 6. GloVe: вывод из отношений вероятностей

Пусть $P_{ik} = X_{ik}/X_i$ — вероятность контекста $k$ у слова $i$. Смысл несут **отношения** $P_{ik}/P_{jk}$. Требуем
$F\big((w_i - w_j)^{\top}\tilde w_k\big) = P_{ik}/P_{jk}$. Чтобы разность в аргументе переходила в отношение, $F$ должна быть
гомоморфизмом $(\mathbb R, +)\to(\mathbb R_{>0}, \times)$, то есть $F = \exp$:

$$
w_i^{\top}\tilde w_k = \log P_{ik} = \log X_{ik} - \log X_i .
$$

$\log X_i$ не зависит от $k$ — поглощается смещением $b_i$; для симметрии добавляем $\tilde b_k$:
$w_i^{\top}\tilde w_k + b_i + \tilde b_k = \log X_{ik}$. Взвешенный МНК даёт функционал GloVe с весом $f(X_{ij})$:
$f(0) = 0$ (иначе $\log 0$), $f$ не убывает (редкие пары шумные) и насыщается (частые пары не доминируют). Итоговый вектор — $w + \tilde w$.

---

## 7. Аналогии: 3CosAdd и 3CosMul

Для нормированных векторов поиск $b^{*} = \arg\max_{x}\cos\big(x, b - a + a^{*}\big)$ эквивалентен

$$
\arg\max_x\big[\cos(x, b) - \cos(x, a) + \cos(x, a^{*})\big]\quad\text{(3CosAdd)},
$$

то есть одно сильное сходство может «перекричать» остальные. Устойчивее мультипликативный вариант
$\arg\max_x\frac{\cos(x, b)\cos(x, a^{*})}{\cos(x, a) + \varepsilon}$ (3CosMul, косинусы сдвинуты в $[0, 1]$). Обязательно исключать
$a, b, a^{*}$ из кандидатов: иначе ответом часто оказывается само слово запроса (так работают «красивые» демо с оговорками).

---

## 8. Выравнивание пространств: ортогональная задача Прокруста

Даны пары переводов: строки $X$ (язык 1) и $Y$ (язык 2). Ищем поворот $W$, $W^{\top}W = I$:

$$
\min_{W^{\top}W = I}\lVert XW - Y\rVert_F^2 = \lVert X\rVert_F^2 + \lVert Y\rVert_F^2 - 2\operatorname{tr}\big(W^{\top}X^{\top}Y\big).
$$

Минимизация эквивалентна максимизации $\operatorname{tr}(W^{\top}M)$, $M = X^{\top}Y = U\Sigma V^{\top}$:
$\operatorname{tr}(W^{\top}U\Sigma V^{\top}) = \operatorname{tr}(\Sigma\,V^{\top}W^{\top}U) = \sum_i\sigma_i Z_{ii}$, где $Z = V^{\top}W^{\top}U$ ортогональна,
поэтому $Z_{ii}\le1$, и максимум достигается при $Z = I$:

$$
W^{*} = UV^{\top}.
$$

Ортогональность сохраняет расстояния и косинусы внутри каждого языка — поэтому она лучше произвольной линейной $W$. Тот же
приём выравнивает эмбеддинги разных эпох при изучении семантического сдвига.

---

## 9. Эккарт–Янг и выбор размерности

$\lVert A - A_r\rVert_F^2 = \sum_{i>r}\sigma_i^2$, относительная ошибка $\sqrt{\sum_{i>r}\sigma_i^2/\sum_i\sigma_i^2}$. Для PPMI-матриц спектр
убывает медленно (это не «почти низкоранговая» матрица), поэтому размерность 100–300 — компромисс между шумом и выразительностью,
выбираемый по downstream-задаче, а не по доле объяснённой дисперсии. Усечённое SVD больших разреженных матриц считают
рандомизированно (Halko et al.): случайная проекция на $r + p$ векторов, несколько степенных итераций, QR и SVD маленькой матрицы.

---

## 10. Токенизация как сжатие и как вероятностная модель

**BPE** — жадное сжатие: каждое слияние уменьшает длину корпуса в токенах ровно на число (непересекающихся) вхождений сливаемой пары, то есть BPE жадно
максимизирует сокращение длины при фиксированном размере словаря. Результат зависит от порядка слияний, поэтому при
применении их повторяют в том же порядке.

**WordPiece.** Под униграммной моделью $\log P(\text{корпус}) = \sum_t\log p(t)$. Слияние $a, b\to ab$ меняет правдоподобие примерно на
$\#(ab)\cdot\log\frac{p(ab)}{p(a)\,p(b)}$ (без учёта перенормировки), поэтому критерий — отношение $\frac{\#(ab)}{\#(a)\,\#(b)}$: сливаются
пары, части которых редко встречаются порознь (PMI-подобный критерий), а не просто частые.

**Unigram LM** (Kudo, 2018). Токены независимы, $P(\mathbf x) = \prod_i p(x_i)$; у строки $s$ множество сегментаций $S(s)$:

$$
\mathcal L = \sum_{s}\log\sum_{\mathbf x\in S(s)}P(\mathbf x).
$$

$p$ оцениваются EM-алгоритмом (сумма по сегментациям — динамическим программированием, как forward в HMM). Затем из словаря удаляют
10–20% токенов с наименьшей потерей $\mathcal L$ при удалении (одиночные символы сохраняют) и повторяют до нужного размера.
Сегментация — Витерби; **subword regularization** — сэмплирование сегментаций при обучении как аугментация.

**Плотность токенизации** (fertility): среднее число токенов на слово. Байтовый BPE, обученный на английском, разбивает кириллицу
вплоть до байтов (у буквы 2 байта UTF-8), поэтому русский текст стоит в разы больше токенов: контекст «короче», генерация дороже.

---

## 11. ELMo: функционал biLM

$$
\sum_{k=1}^{N}\Big(\log p(t_k\mid t_1,\ldots,t_{k-1};\,\Theta_x, \overrightarrow\Theta_{\text{LSTM}}, \Theta_s)
+ \log p(t_k\mid t_{k+1},\ldots,t_N;\,\Theta_x, \overleftarrow\Theta_{\text{LSTM}}, \Theta_s)\Big),
$$

$\Theta_x$ (символьная CNN-входная часть) и $\Theta_s$ (softmax) общие у направлений, LSTM — свои. Направления обучаются **независимо**
(каждое — обычная LM), поэтому слово не «видит себя» — в отличие от наивного двунаправленного предсказания. BERT решает ту же
проблему маскированием ([тема 23](../23_transformer_bert/README.md)).

---

## Ссылки

- Mikolov et al. [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781) · [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546), 2013.
- Goldberg, Levy. [word2vec Explained: deriving Mikolov et al.'s negative-sampling](https://arxiv.org/abs/1402.3722), 2014.
- Levy, Goldberg. [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization), 2014 · Levy, Goldberg, Dagan. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/), 2015.
- Levy, Goldberg. [Linguistic Regularities in Sparse and Explicit Word Representations](https://aclanthology.org/W14-1618/) (3CosMul), 2014.
- Pennington, Socher, Manning. [GloVe](https://nlp.stanford.edu/pubs/glove.pdf), 2014 · Bojanowski et al. [fastText](https://arxiv.org/abs/1607.04606), 2017.
- Smith et al. [Offline bilingual word vectors, orthogonal transformations](https://arxiv.org/abs/1702.03859), 2017 · Hamilton et al. [Diachronic Word Embeddings](https://www.aclweb.org/anthology/P16-1141.pdf), 2016.
- Halko, Martinsson, Tropp. [Finding structure with randomness](https://arxiv.org/abs/0909.4061), 2011.
- Sennrich et al. [BPE](https://arxiv.org/abs/1508.07909) · Kudo. [Subword Regularization](https://arxiv.org/abs/1804.10959) · Peters et al. [ELMo](https://arxiv.org/abs/1802.05365).
