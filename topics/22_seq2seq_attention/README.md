# 22. Seq2seq и механизм внимания

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Seq2seq — **условная языковая модель** $p(y\mid x)=\prod_t p(y_t\mid y_{<t},x)$: **энкодер** читает вход, **декодер**
генерирует выход токен за токеном (перевод, суммаризация, диалог). Учится с teacher forcing и кросс-энтропией, генерирует
жадно или **beam search**. Классический seq2seq сжимает весь вход в один вектор — **узкое горло**: длинные предложения
переводятся плохо. **Внимание** даёт декодеру на каждом шаге взвешенную сумму **всех** состояний энкодера: веса = softmax от
сходства запроса (состояние декодера) с ключами (состояния энкодера). Побочно получаем **выравнивание** слов. Отсюда один шаг
до трансформера (тема 23).

| Термин | Определение |
|---|---|
| Энкодер / декодер | RNN (или трансформер), читающая вход / генерирующая выход; параметры разные |
| Вектор контекста | в базовом seq2seq — последнее состояние энкодера; с вниманием — $c_t=\sum_s a_{ts}h_s$, свой на каждом шаге |
| Query / key / value | что ищем (состояние декодера) / с чем сравниваем / что усредняем (состояния энкодера) |
| Score-функция | dot $h^{\top}s$, general (multiplicative) $h^{\top}Ws$, additive (Bahdanau) $v^{\top}\tanh(W_1h+W_2s)$, scaled dot $\frac{q^{\top}k}{\sqrt d}$ |
| Выравнивание (alignment) | матрица весов внимания «слово выхода × слово входа» |
| Greedy / beam search | брать argmax на каждом шаге / держать $k$ лучших префиксов |
| BLEU | геометрическое среднее точностей 1–4-грамм × штраф за краткость |
| Teacher forcing | вход декодера — истинный предыдущий токен (сдвиг цели вправо + `<sos>`) |

$$
p(y\mid x)=\prod_{t=1}^{T_y}p(y_t\mid y_{<t},x),\qquad
a_{ts}=\frac{\exp\operatorname{score}(h_t,s_s)}{\sum_{s'}\exp\operatorname{score}(h_t,s_{s'})},\qquad
c_t=\sum_s a_{ts}\,s_s,\qquad \tilde h_t=\tanh\big(W_c[h_t;c_t]\big)
$$

## Конспект

Разделы раскрываются по клику.

**Часть A. Seq2seq**

<details>
<summary><b>1. Задача и формулировка</b></summary>

Sequence-to-sequence: вход — последовательность $x = (x_1,\ldots,x_{T_x})$, выход — последовательность $y$ другой длины. Главный
пример — машинный перевод; также суммаризация, диалог, распознавание речи, генерация кода, перевод между языками программирования,
исправление опечаток, image captioning (вход — картинка). Порядок слов в переводе меняется («le chat noir» → «the black cat»), длины
не совпадают — поэтому синхронная RNN ([тема 20](../20_rnn/README.md)) не подходит.

Три вопроса (по курсу Лены Войты):
1. **Моделирование:** как устроена $p(y\mid x,\theta)$?
2. **Обучение:** как найти $\theta$?
3. **Вывод (inference):** как найти $y^{*} = \arg\max_y p(y\mid x,\theta)$?

</details>

<details>
<summary><b>2. Encoder–decoder и условная языковая модель</b></summary>

Языковая модель оценивает $p(y)$; seq2seq — **условная** языковая модель (conditional LM):

$$
p(y\mid x) = \prod_{t=1}^{T_y}p(y_t\mid y_{<t}, x).
$$

- **Энкодер** читает вход и строит его представление.
- **Декодер** — языковая модель, обусловленная этим представлением; на каждом шаге выдаёт распределение следующего токена.

Простейший вариант (Sutskever et al., 2014; Cho et al., 2014): энкодер-RNN, его **последнее состояние** — начальное состояние
декодера-RNN. Энкодер и декодер — **разные** сети с разными параметрами (и словарями).

**Sutskever et al., 2014:** 4-слойная LSTM, размерность 1000, входной словарь 160 000, выходной 80 000, обучение ~10 дней, ансамбль
моделей. Неожиданный приём: **разворот порядка входа** заметно улучшал качество — первые слова входа оказываются ближе к первым
словам выхода, путь градиента короче.

Тонкость (лекция 8): если состояние энкодера передаётся только в первый шаг декодера, к концу генерации информация о входе «затирается»;
можно подавать вектор контекста на **каждый** шаг (конкатенацией ко входу).

</details>

<details>
<summary><b>3. Обучение</b></summary>

Кросс-энтропия по всем позициям цели, усреднённая по токенам: $\mathcal L = -\sum_t\log p(y_t\mid y_{<t}, x)$. На входе декодера —
**истинный** предыдущий токен (teacher forcing; цель, сдвинутая вправо, с `<sos>` в начале). Это ровно MLE и позволяет считать все
шаги за один проход. Минус — **exposure bias** (на тесте модель видит свои ошибки); частичные лекарства — смешивание с собственными
прогнозами (`teacher_forcing_ratio` в семинаре, scheduled sampling), подробнее — [тема 20, §14](../20_rnn/README.md).

Технически: `<pad>` в цели игнорируется (`ignore_index`), паддинг во входе — маскируется (packing, маска внимания); `<eos>` — часть
цели, иначе модель не научится останавливаться.

</details>

<details>
<summary><b>4. Вывод: greedy и beam search</b></summary>

Точный $\arg\max_y$ по всем последовательностям недостижим ($\lvert V\rvert^{T}$ вариантов).

- **Greedy:** на каждом шаге argmax. Быстро, но локально оптимальный выбор не гарантирует лучшую последовательность: в L1-примере
  жадный путь «A x» имеет $p = 0.5\cdot0.4 = 0.2$, а «B x» — $0.4\cdot0.9 = 0.36$ («если выбираем лучшего следующего, не обязательно
  максимизируем качество»).
- **Beam search:** храним $k$ лучших префиксов (beam size, обычно 4–10); на шаге расширяем каждый всеми токенами, оставляем $k$ лучших
  по сумме $\log p$; гипотеза с `<eos>` уходит в список готовых.
- **Нормировка на длину:** сумма логарифмов отрицательна, и длинные гипотезы проигрывают — beam search склонен к коротким
  переводам (и при очень большом $k$ — к пустому). Сравнивают $\frac{1}{\lvert y\rvert^{\alpha}}\log p(y\mid x)$, $\alpha\approx0.6$–$1$
  (GNMT: $\mathrm{lp}(y) = \frac{(5+\lvert y\rvert)^{\alpha}}{6^{\alpha}}$).
- Другие эвристики: штраф за повторы n-грамм, coverage penalty (всё ли во входе «покрыто» вниманием).
- Для **генерации текста** (не перевода) beam search даёт скучные, повторяющиеся тексты — там сэмплируют (top-k, top-p, температура; [тема 24](../24_language_models_gpt/README.md)).

</details>

<details>
<summary><b>5. Оценка: BLEU и другие</b></summary>

**BLEU** (Papineni et al., 2002): модифицированные («обрезанные») точности n-грамм $p_n$, $n = 1..4$ — сколько n-грамм гипотезы есть
в референсе, причём каждую n-грамму засчитываем не больше раз, чем она встречается в референсе (иначе «the the the…» получило бы
высокую униграммную точность); геометрическое среднее; **штраф за краткость** (brevity penalty), так как точность не наказывает
за пропуски:

$$
\mathrm{BLEU} = \mathrm{BP}\cdot\exp\Big(\frac14\sum_{n=1}^{4}\log p_n\Big),\qquad \mathrm{BP} = \min\big(1,\ e^{1 - r/c}\big),
$$

$r$ — длина референса, $c$ — длина гипотезы. Считается **на корпусе** (суммы числителей и знаменателей по всем предложениям), на
отдельном предложении нестабилен (нулевая 4-граммная точность → 0; нужен smoothing). Стандартная реализация — `sacrebleu`
(с фиксированной токенизацией, чтобы цифры были сравнимы).

L1-пример: «his dad» вместо «his father» → 0.84 (синоним штрафуется); пропуск «as» → 0.52; обрезанная гипотеза «he is not as tall» → 0.55
только из-за BP.

Другие метрики: **chrF** (F-мера по символьным n-граммам — лучше для морфологически богатых языков), METEOR (синонимы, стемминг),
обученные **COMET**, BLEURT (лучше коррелируют с людьми), человеческая оценка (adequacy/fluency, MQM).

</details>

**Часть B. Механизм внимания**

<details>
<summary><b>6. Узкое горло</b></summary>

Весь вход сжат в **один вектор фиксированного размера** — плохо, особенно для длинных предложений: информация о начале затирается,
качество перевода падает с длиной. L1-пример (разворот цифр): без внимания точность 0.83 на длинах 4–8 и 0 на 13–16; с вниманием —
0.98 и 0.92.

</details>

<details>
<summary><b>7. Идея (Bahdanau, Cho, Bengio, 2015)</b></summary>

Энкодер передаёт декодеру **не одно состояние, а состояния всех токенов**. На каждом шаге декодер сам решает, на какие из них
смотреть:

1. **Оценки (scores):** насколько состояние энкодера $s_k$ соответствует текущему состоянию декодера $h_t$: $e_{tk} = \operatorname{score}(h_t, s_k)$.
2. **Веса:** $a_{tk} = \operatorname{softmax}_k(e_{tk})$ — неотрицательные, сумма 1.
3. **Вектор контекста:** $c_t = \sum_k a_{tk}\,s_k$ — «пулинг по схожести» для последовательности любой длины.
4. Контекст используется в декодере (конкатенируется со входом или с состоянием).

Общая форма: внимание — техника вычисления **взвешенной суммы значений (values) по запросу (query)**; веса зависят от сходства
запроса с ключами (keys). Это способ получить описание фиксированного размера по запросу из множества переменного размера.

</details>

<details>
<summary><b>8. Score-функции</b></summary>

| Название | $\operatorname{score}(h, s)$ | Источник |
|---|---|---|
| dot-product | $h^{\top}s$ | Luong et al., 2015 |
| general (multiplicative, bilinear) | $h^{\top}W s$ | Luong et al., 2015 |
| additive (concat, MLP) | $v^{\top}\tanh(W_1 h + W_2 s)$ | Bahdanau et al., 2015 |
| scaled dot-product | $\dfrac{q^{\top}k}{\sqrt d}$ | Vaswani et al., 2017 |
| content-based (косинус) | $\beta\cos(h, s)$ | Graves et al., 2014 (Neural Turing Machines) |

Dot требует одинаковых размерностей и не имеет параметров; general — одна матрица; additive дороже, но был лучше при больших
размерностях без масштабирования — пока не появилось деление на $\sqrt d$ (почему — см. раздел «3. Зачем делить на корень из d»).

</details>

<details>
<summary><b>9. Bahdanau vs Luong</b></summary>

| | Bahdanau (2015) | Luong (2015) |
|---|---|---|
| Энкодер | **двунаправленный** GRU, $s_k = [\overrightarrow s_k;\overleftarrow s_k]$ | однонаправленный (стек LSTM) |
| Запрос | **предыдущее** состояние декодера $h_{t-1}$ | **текущее** $h_t$ (после шага RNN) |
| Score | additive | dot / general / concat |
| Использование $c_t$ | вход RNN-шага: $h_t = \mathrm{RNN}(h_{t-1}, [y_{t-1}; c_t])$ | после: $\tilde h_t = \tanh(W_c[h_t; c_t])$, $p = \operatorname{softmax}(W_o\tilde h_t)$ |
| Дополнительно | — | input feeding ($\tilde h_{t-1}$ — во вход шага $t$); local attention |

</details>

<details>
<summary><b>10. Виды внимания</b></summary>

- **Soft / global** — ко всему входу, дифференцируемо (стандарт).
- **Hard** (Xu et al., 2015, «Show, Attend and Tell») — выбирается одна позиция (сэмплирование), учится через REINFORCE; дешевле, но шумно.
- **Local** (Luong) — окно вокруг предсказанной позиции $p_t$ с гауссовым весом; компромисс для длинных входов.
- **Self-attention / intra-attention** — к позициям **той же** последовательности: каждое слово собирает контекст из соседей.
  Основа трансформера ([тема 23](../23_transformer_bert/README.md)).
- Внимание к картинке: декодер подписи смотрит на разные области карты признаков CNN.

</details>

<details>
<summary><b>11. Плюсы внимания</b></summary>

- Улучшает качество перевода (и не только).
- Решает проблему узкого горла.
- Решает проблему затухания сигнала / градиента: путь от любого слова входа до выхода — один шаг.
- Интерпретируемость: **выравнивание (alignment)** слов «бесплатно», без разметки — модель сама учится, что «européenne» ↔ «European»,
  а порядок «zone économique européenne» ↔ «European Economic Area» переставляет.

> **Важно.** Веса внимания — не строгое объяснение. Состояния энкодера смешивают контекст: в L1-примере модель выдаёт токен $x_7$, «глядя»
> на позицию 8, потому что $h_8$ однонаправленного энкодера содержит и $x_7$. См. также «Attention is not Explanation» (Jain & Wallace, 2019).

</details>

**Часть C. Реализация (семинар 8)**

<details>
<summary><b>12. Данные</b></summary>

Tatoeba fr–en (`tatoebatools.ParallelCorpus('fra', 'eng')`, 440 683 пары). Нормализация: unicode → ASCII (удаление диакритики через
`unicodedata.normalize('NFD')` и категорию `Mn`), нижний регистр, отделение `!?` пробелом, всё кроме латиницы и `!?` — в пробел.
Фильтр: обе фразы короче `MAX_LENGTH = 10` слов и английская начинается с «i am / he is / she is / you are / we are / they are»
(и их сокращений «i m», «he s»…). Из первых 250 000 пар остаётся 18 757; словари: французский 6341, английский 4457 слов. Класс
`Lang`: `word2index`, `word2count`, `index2word`, `SOS = 0`, `EOS = 1`. Цели и входы дополняются нулями до `MAX_LENGTH`.

</details>

<details>
<summary><b>13. Модели</b></summary>

```python
class EncoderRNN(nn.Module):
    def __init__(self, vocab, H, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab, H)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):                                    # [B, Tx]
        return self.gru(self.dropout(self.embedding(x)))     # outputs [B, Tx, H], hidden [1, B, H]

class AttnDecoderRNN(nn.Module):                             # шаг декодера с вниманием
    def __init__(self, H, vocab, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab, H)
        self.attention = AttentionBlock(H)                   # Q, K, V-проекции + scaled dot-product
        self.gru = nn.GRU(2 * H, H, batch_first=True)
        self.out = nn.Linear(H, vocab)
        self.dropout = nn.Dropout(dropout_p)
    def forward_step(self, y_prev, hidden, enc_out):
        emb = self.dropout(self.embedding(y_prev))           # [B, 1, H]
        query = hidden.permute(1, 0, 2)                      # [B, 1, H] — состояние декодера как запрос (стиль Bahdanau)
        context, weights = self.attention(query, enc_out, enc_out)
        out, hidden = self.gru(torch.cat([emb, context], dim=2), hidden)
        return self.out(out), hidden, weights
```

Цикл декодера: `decoder_input = SOS`, `decoder_hidden = encoder_hidden`; `MAX_LENGTH` шагов; с teacher forcing следующим входом
становится `target[:, i]`, иначе — `topk(1)` предсказания с `.detach()`. Выходы → `log_softmax` → `NLLLoss`. Оптимизаторы AdamW,
lr 0.003, `hidden_size` 256 (без внимания) / 128 (с вниманием), `tf_ratio` 0.6 / 0.7, 30 / 20 эпох.

**Функция внимания из семинара** (матричная форма, для батча запросов):

```python
def attention(queries, keys, values, Q, K, V, mask=None):
    q, k, v = Q(queries), K(keys), V(values)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))       # [B, Tq, Tk]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -torch.inf)
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights
```

**Результаты семинара:** без внимания — «i m ready to go go», «i m glad i saw rain job» (повторы и потеря смысла к концу); с вниманием —
«he is not as tall as his father», «they are teachers» (вместо «professors» — допустимый синоним, но BLEU его накажет). Ошибки
остаются: «i m too tired to drive happy». Визуализация матрицы весов (`matshow`) показывает выравнивание.

</details>

<details>
<summary><b>14. Что улучшить в коде семинара</b></summary>

| Проблема | Последствие | Исправление |
|---|---|---|
| `<pad>` = `SOS` = 0 | модель учится предсказывать `SOS` на паддинге; паддинг неотличим от начала | отдельный `PAD`, `NLLLoss(ignore_index=PAD)` |
| нет маски во внимании | часть веса уходит на паддинг | `mask = (src != PAD)` |
| декодер всегда делает `MAX_LENGTH` шагов | лишние вычисления, «хвосты» после `EOS` | остановка батча, когда все сгенерировали `EOS` |
| teacher forcing решается на весь батч | высокая дисперсия обучения | решать по шагам (scheduled sampling) |
| оценка «на глаз» | нельзя сравнивать модели | BLEU/chrF на отложенной выборке (`sacrebleu`) |
| `set_xticklabels` без `FixedLocator` | предупреждение matplotlib | `ax.set_xticks(range(n))` перед подписями |

</details>

**Часть D. От внимания к трансформеру**

<details>
<summary><b>От внимания к трансформеру</b></summary>

Если внимание даёт прямой доступ к любому состоянию, нужна ли рекуррентность? Трансформер (Vaswani et al., 2017, «Attention Is All
You Need») убирает RNN: **self-attention** в энкодере, **masked self-attention** + **cross-attention** (encoder–decoder attention,
ровно механизм этой темы) в декодере, позиционное кодирование вместо порядка шагов, multi-head. Подробно — [тема 23](../23_transformer_bert/README.md).

</details>

<details>
<summary><b>Чек-лист</b></summary>

- [ ] `<sos>`, `<eos>`, `<pad>` — разные токены; паддинг игнорируется в лоссе и маскируется во внимании.
- [ ] Вход декодера — цель, сдвинутая вправо; не перепутать выравнивание логитов и целей.
- [ ] Inference: beam search (4–10) с нормировкой на длину; лимит длины.
- [ ] Метрика — BLEU/chrF на корпусе через `sacrebleu`, плюс ручной просмотр ошибок.
- [ ] Subword-токенизация (BPE) вместо словаря слов — [тема 21](../21_word_embeddings_tokenization/README.md).

</details>

**Математика и выводы**

<details>
<summary><b>1. Условная языковая модель и функция потерь</b></summary>

По цепному правилу (без всяких допущений) $p(y\mid x) = \prod_{t=1}^{T_y}p(y_t\mid y_{<t}, x)$. Для обучающей пары максимизируем
логарифм правдоподобия, то есть минимизируем

$$
\mathcal L(\theta) = -\sum_{t=1}^{T_y}\log p_\theta(y_t\mid y_{<t}, x) = \sum_{t=1}^{T_y}\mathrm{CE}\big(\text{one-hot}(y_t),\ p_\theta(\cdot\mid y_{<t}, x)\big).
$$

Каждый множитель — softmax над словарём, поэтому лосс — сумма кросс-энтропий по позициям; условие $y_{<t}$ берётся из данных
(teacher forcing). Нормировка: средний лосс на токен $\bar{\mathcal L}$ связан с **перплексией** $\mathrm{PPL} = e^{\bar{\mathcal L}}$ —
«эффективное число равновероятных вариантов» на шаге ([тема 24](../24_language_models_gpt/README.md)).

</details>

<details>
<summary><b>2. Внимание как дифференцируемый поиск по словарю</b></summary>

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

</details>

<details>
<summary><b>3. Зачем делить на корень из d</b></summary>

Пусть компоненты $q$ и $k$ независимы, со средним 0 и дисперсией 1. Тогда

$$
\mathbb E[q^{\top}k] = 0,\qquad \operatorname{Var}(q^{\top}k) = \sum_{i=1}^{d}\operatorname{Var}(q_ik_i) = \sum_{i=1}^{d}\mathbb E[q_i^2]\,\mathbb E[k_i^2] = d .
$$

При $d = 512$ оценки имеют стандартное отклонение $\approx22.6$: разница соседних логитов в десятки единиц делает softmax почти
one-hot, а его якобиан $\operatorname{diag}(a) - aa^{\top}\to0$ — градиенты исчезают, и внимание перестаёт учиться (застревает на
случайном выборе). Деление на $\sqrt d$ возвращает дисперсию к 1 при любой размерности. Это же объясняет наблюдение Vaswani et al.:
без масштабирования additive-внимание выигрывало у dot-product при больших $d$ — $\tanh$ в additive само ограничивает масштаб.

</details>

<details>
<summary><b>4. Additive vs multiplicative: стоимость и выразительность</b></summary>

- **Dot / general:** все оценки — одно матричное произведение $E = HWS^{\top}\in\mathbb R^{T_y\times T_x}$, стоимость $O(T_yT_xd)$ на
  высокооптимизированном GEMM. General — билинейная форма; при $W = I$ — dot.
- **Additive:** $e_{tk} = v^{\top}\tanh(W_1h_t + W_2s_k)$. Проекции $W_1h_t$ и $W_2s_k$ считаются один раз ($O(T_yd^2 + T_xd^2)$), но затем
  нужен тензор $T_y\times T_x\times d'$ с нелинейностью — нельзя свести к одному матричному умножению, больше памяти. Зато это MLP —
  оценка нелинейна по паре $(h, s)$.

Поэтому в трансформере выбрано scaled dot-product: быстро и параллельно, а выразительность добирается обучаемыми проекциями $W_Q, W_K$
и несколькими «головами».

</details>

<details>
<summary><b>5. Модель Bahdanau полностью</b></summary>

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

</details>

<details>
<summary><b>6. Длина пути градиента</b></summary>

Без внимания влияние $x_1$ на выход $y_t$ проходит через $T_x - 1$ шагов энкодера и $t$ шагов декодера — произведение $\sim T_x + t$
якобианов, со всеми проблемами из [темы 20](../20_rnn/README.md). С вниманием есть путь
$x_1\to h_1\to c_t\to y_t$ длины $O(1)$: $\partial c_t/\partial h_1 = \alpha_{t1}I + (\text{через веса})$. Пока вес не нулевой,
градиент доходит без затухания. В L1-примере это видно как разница точностей на длинах 13–16: 0.00 против 0.92.

Ёмкостный аргумент про узкое горло: фиксированный вектор $\mathbb R^{H}$ должен кодировать вход любой длины; энтропия входа растёт
линейно с длиной (для случайных цифр — $T\log_2 10$ бит), а сеть, обученная с ограниченной точностью, хранит в $H$ числах ограниченный
объём. С вниманием «память» — все $T_x$ состояний, она растёт вместе со входом.

</details>

<details>
<summary><b>7. Beam search: что именно ищем</b></summary>

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

</details>

<details>
<summary><b>8. BLEU подробно</b></summary>

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

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Зачем внимание, если есть LSTM?</summary>

узкое горло: весь вход в одном векторе фиксированного размера; внимание даёт прямой путь (и градиент) к любому слову входа.

</details>

<details>
<summary>Чем отличаются Bahdanau и Luong?</summary>

additive vs multiplicative score; Bahdanau: внимание **до** шага RNN по $s_{t-1}$, bidirectional энкодер; Luong: **после**, по $h_t$, и $\tilde h_t=\tanh(W_c[h_t;c_t])$.

</details>

<details>
<summary>Зачем $\sqrt d$ в scaled dot-product?</summary>

дисперсия $q^{\top}k$ растёт как $d$, softmax насыщается и градиенты исчезают.

</details>

<details>
<summary>Почему beam search с большим $k$ выбирает короткие переводы?</summary>

$\log p$ складывается из отрицательных слагаемых — короче = больше; лечат нормировкой на длину $\log p/\lvert y\rvert^{\alpha}$.

</details>

<details>
<summary>Почему не маскировать паддинг во внимании — ошибка?</summary>

часть веса уходит на `<pad>`, контекст «разбавляется» (в семинаре маски нет, а `<pad>` = `SOS` = 0).

</details>

<details>
<summary>Веса внимания = объяснение?</summary>

не совсем: в примере выше модель выдаёт $x_7$, «глядя» на позицию 8 — состояние однонаправленного энкодера $h_8$ содержит и предыдущие токены. Внимание показывает, *какие состояния* смешаны, а не *какие слова* использованы.

</details>

<details>
<summary>Что считает BLEU и чего не видит?</summary>

совпадение n-грамм со ссылкой; синонимы («dad»/«father») и перестановки штрафуются; оценка корпусная, не по предложению.

</details>

<details>
<summary>Почему разворот входа помогал Sutskever?</summary>

первые слова входа оказываются близко к первым словам выхода — короче путь градиента.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- **Sutskever et al. (2014):** 4-слойная LSTM, размер 1000, словари 160k/80k, обучение 10 дней на 8 GPU, ансамбль; разворот входа заметно улучшал BLEU. **Cho et al. (2014):** RNN encoder–decoder с GRU.
- **Виды внимания:** soft/global (ко всему входу, дифференцируемо) и hard/local (к части входа: hard — сэмплирование позиции, учится через REINFORCE; local Luong — окно вокруг предсказанной позиции); self-attention — к позициям той же последовательности.
- **Плюсы внимания:** лучше качество, решает узкое горло, короткий путь градиента, интерпретируемость, выравнивание «бесплатно».
- **Семинар 8 (Tatoeba fr→en, фразы < 10 слов с префиксами «i am», «he is»…, 18 757 пар):** GRU seq2seq, `tf_ratio` 0.6–0.7; без внимания переводы «съезжают» к концу («i m ready to go go»), с вниманием «he is not as tall as his father».
- Метрики кроме BLEU: chrF (символьные n-граммы), METEOR, COMET/BLEURT (обученные), человеческая оценка.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# Игрушечный «перевод»: развернуть последовательность цифр. Seq2seq без внимания vs с вниманием (Luong, general score)
import torch, torch.nn as nn, torch.nn.functional as F
PAD, SOS, EOS, V = 10, 11, 12, 13

def make_batch(n, lo, hi, g=None):
    L = torch.randint(lo, hi + 1, (n,), generator=g); T = int(L.max())
    src, tgt = torch.full((n, T), PAD), torch.full((n, T + 1), PAD)
    for i, l in enumerate(L):
        s = torch.randint(0, 10, (int(l),), generator=g); src[i, :l] = s; tgt[i, :l] = s.flip(0); tgt[i, l] = EOS
    return src, tgt, L

class Seq2Seq(nn.Module):
    def __init__(self, attn, E=32, H=64):
        super().__init__(); self.attn = attn
        self.emb_src, self.emb_tgt = nn.Embedding(V, E, padding_idx=PAD), nn.Embedding(V, E, padding_idx=PAD)
        self.enc, self.dec = nn.GRU(E, H, batch_first=True), nn.GRU(E, H, batch_first=True)
        self.W, self.comb, self.out = nn.Linear(H, H, bias=False), nn.Linear(2 * H, H), nn.Linear(H, V)
    def encode(self, src, L):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb_src(src), L, batch_first=True, enforce_sorted=False)
        out, h = self.enc(packed)
        return nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.shape[1])[0], h
    def decode(self, y_in, h, enc_out, src_mask):
        d, h = self.dec(self.emb_tgt(y_in), h)                                # [B, Ty, H]
        if not self.attn: return self.out(d), h, None                         # только последний h энкодера
        scores = (d @ self.W(enc_out).transpose(1, 2)).masked_fill(~src_mask[:, None, :], -1e9)   # [B, Ty, Tx], паддинг маскируем
        a = scores.softmax(-1); ctx = a @ enc_out                             # веса и вектор контекста
        return self.out(torch.tanh(self.comb(torch.cat([d, ctx], -1)))), h, a
    def forward(self, src, L, tgt):                                           # teacher forcing: вход = <sos> + цель без последнего
        enc_out, h = self.encode(src, L)
        y_in = torch.cat([torch.full((len(src), 1), SOS), tgt[:, :-1]], 1)
        return self.decode(y_in, h, enc_out, src != PAD)[0]
    @torch.no_grad()
    def greedy(self, src, L, max_len=26):
        enc_out, h = self.encode(src, L); y = torch.full((len(src), 1), SOS); outs, atts = [], []
        for _ in range(max_len):
            logit, h, a = self.decode(y, h, enc_out, src != PAD); y = logit.argmax(-1); outs.append(y); atts.append(a)
        return torch.cat(outs, 1), atts

test = make_batch(1000, 4, 24, torch.Generator().manual_seed(1))
models = {}
for attn in (False, True):
    torch.manual_seed(0); m = Seq2Seq(attn); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for step in range(800):                                                   # учим на длинах 4–16
        src, tgt, L = make_batch(64, 4, 16)
        loss = F.cross_entropy(m(src, L, tgt).reshape(-1, V), tgt.reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); models[attn] = m
    pred, _ = m.greedy(test[0], test[2])
    ok = torch.tensor([bool((pred[i, :l + 1] == test[1][i, :l + 1]).all()) for i, l in enumerate(test[2])])
    print(f"{'с вниманием ' if attn else 'без внимания'}: точное совпадение по длинам " +
          "  ".join(f"{a}–{b}: {ok[(test[2] >= a) & (test[2] <= b)].float().mean():.2f}" for a, b in [(4, 8), (9, 12), (13, 16), (17, 20), (21, 24)]))
```

Результат запуска:

```text
без внимания: точное совпадение по длинам 4–8: 0.83  9–12: 0.15  13–16: 0.00  17–20: 0.00  21–24: 0.00
с вниманием : точное совпадение по длинам 4–8: 0.98  9–12: 0.96  13–16: 0.92  17–20: 0.41  21–24: 0.00
```

```python
# Матрица внимания = выравнивание: для разворота видим «антидиагональ» (со сдвигом на шаг — см. ловушки)
src = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]]); L = torch.tensor([8])
pred, atts = models[True].greedy(src, L, max_len=9)
A = torch.cat(atts, 1)[0]                                                     # [Ty, Tx]
print("вход:   ", src[0].tolist(), "\nвыход:  ", pred[0, :8].tolist(), "+ EOS" if pred[0, 8] == EOS else "")
for t in range(8):
    row = A[t]
    print(f"шаг {t}: выдал {pred[0, t].item()} ← смотрит на позицию {row.argmax().item()} (вес {row.max():.2f})  " + "".join("█" if w > 0.5 else "▒" if w > 0.1 else "·" for w in row))
```

Результат запуска:

```text
вход:    [3, 1, 4, 1, 5, 9, 2, 6] 
выход:   [6, 2, 9, 5, 1, 4, 1, 3] + EOS
шаг 0: выдал 6 ← смотрит на позицию 7 (вес 1.00)  ·······█
шаг 1: выдал 2 ← смотрит на позицию 7 (вес 1.00)  ·······█
шаг 2: выдал 9 ← смотрит на позицию 6 (вес 0.99)  ······█·
шаг 3: выдал 5 ← смотрит на позицию 5 (вес 1.00)  ·····█··
шаг 4: выдал 1 ← смотрит на позицию 4 (вес 0.98)  ····█···
шаг 5: выдал 4 ← смотрит на позицию 3 (вес 0.95)  ···█····
шаг 6: выдал 1 ← смотрит на позицию 2 (вес 1.00)  ··█·····
шаг 7: выдал 3 ← смотрит на позицию 1 (вес 0.97)  ·█······
```

```python
# Beam search vs greedy на игрушечной модели и BLEU с нуля
import math
from collections import Counter
P = {(): {"A": 0.5, "B": 0.4, "</s>": 0.1},
     ("A",): {"x": 0.4, "y": 0.3, "</s>": 0.3}, ("B",): {"x": 0.9, "y": 0.05, "</s>": 0.05},
     ("A", "x"): {"</s>": 1.0}, ("A", "y"): {"</s>": 1.0}, ("B", "x"): {"</s>": 1.0}, ("B", "y"): {"</s>": 1.0}}

def beam_search(k, alpha=0.0, max_len=5):
    beams, done = [((), 0.0)], []
    for _ in range(max_len):
        cand = sorted(((seq + (w,), lp + math.log(p)) for seq, lp in beams for w, p in P[seq].items()), key=lambda c: -c[1])
        beams = []
        for seq, lp in cand[:k]: (done if seq[-1] == "</s>" else beams).append((seq, lp))
        if not beams: break
    return max(done, key=lambda c: c[1] / len(c[0]) ** alpha)                # alpha > 0 — нормировка на длину
for k in (1, 2):
    seq, lp = beam_search(k); print(f"beam={k}: {' '.join(seq):9s} p={math.exp(lp):.2f}")   # жадный: 0.5·0.4 < 0.4·0.9

def bleu(hyp, ref, N=4):
    hyp, ref = hyp.split(), ref.split(); logp = 0.0
    for n in range(1, N + 1):
        h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        clipped = sum(min(c, r[g]) for g, c in h.items())                      # «обрезанная» точность n-грамм
        logp += math.log(max(clipped, 1e-9) / max(sum(h.values()), 1)) / N
    return min(1.0, math.exp(1 - len(ref) / len(hyp))) * math.exp(logp)       # штраф за краткость × геом. среднее
ref = "he is not as tall as his father"
for hyp in [ref, "he is not as tall as his dad", "he is not tall as his father", "he is not as tall", "the the the the the the the the"]:
    print(f"BLEU = {bleu(hyp, ref):.3f}   {hyp}")
```

Результат запуска:

```text
beam=1: A x </s>  p=0.20
beam=2: B x </s>  p=0.36
BLEU = 1.000   he is not as tall as his father
BLEU = 0.841   he is not as tall as his dad
BLEU = 0.515   he is not tall as his father
BLEU = 0.549   he is not as tall
BLEU = 0.000   the the the the the the the the
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Лена Войта: [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) (с интерактивными схемами, разделами про beam search, BPE и анализ) ·  [запись лекции «Seq2seq and Attention»](https://disk.yandex.ru/i/Nb3BT79-a4hbWQ) (Яндекс Диск)
- Учебник ML Яндекса: [Нейросети для последовательностей → Seq2seq, внимание](https://education.yandex.ru/handbook/ml/article/nejroseti-dlya-raboty-s-posledovatelnostyami) · репозиторий [yandexdataschool/nlp_course](https://github.com/yandexdataschool/nlp_course) (week03 — Seq2seq and Attention)
- Stanford [CS224n](https://web.stanford.edu/class/cs224n/) — лекции про машинный перевод, seq2seq и внимание · шпаргалка [CS230: Machine translation, Beam search, BLEU, Attention](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/#machine-translation)
- Обзоры: [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) · [D. Batista — Attention in seq2seq](http://www.davidsbatista.net/blog/2020/01/25/Attention-seq2seq/) · [The AI Summer — attention](https://theaisummer.com/attention/) · [PyTorch: NLP from scratch — translation with seq2seq](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- Статьи: [Sutskever 2014](https://arxiv.org/abs/1409.3215) · [Cho 2014](https://www.aclweb.org/anthology/D14-1179/) · [Bahdanau 2015](https://arxiv.org/abs/1409.0473) · [Luong 2015](https://arxiv.org/abs/1508.04025) · [Show, Attend and Tell](http://proceedings.mlr.press/v37/xuc15.pdf) · [Neural Turing Machines](https://arxiv.org/abs/1410.5401) · [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [GNMT (длина, coverage)](https://arxiv.org/abs/1609.08144) · [BLEU](https://aclanthology.org/P02-1040/) · [sacreBLEU](https://arxiv.org/abs/1804.08771) · [chrF](https://aclanthology.org/W15-3049/) · [COMET](https://arxiv.org/abs/2009.09025) · [Attention is not Explanation](https://arxiv.org/abs/1902.10186)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Лекция_08.pptx`, слайды 39–53: seq2seq (Sutskever, Cho), тонкости декодера, внутреннее представление предложений, обобщения, механизм внимания, score-функции, выравнивание, узкое горло, виды внимания (self, global/soft, local/hard), итоги.
- Семинар 8 «Seq2seq. Attention» ([Colab](https://colab.research.google.com/drive/1R8bRUzgymyISMdYAubDVHmoqENDbjtE8)): посимвольная GRU на «hello», корпус Tatoeba fr–en, `Lang`, нормализация, Encoder/Decoder, teacher forcing, функция внимания в матричной форме, `AttnDecoderRNN`, визуализация выравнивания.
- `Повторение.pptx` — обзорные слайды по attention и seq2seq.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Bahdanau, Cho, Bengio. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2015.
- Luong, Pham, Manning. [Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025), 2015.
- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017 (§3.2.1 — масштабирование).
- Wu et al. [Google's NMT System](https://arxiv.org/abs/1609.08144), 2016 (length и coverage penalty).
- Stahlberg, Byrne. [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://arxiv.org/abs/1908.10090), 2019 · Koehn, Knowles. [Six Challenges for NMT](https://arxiv.org/abs/1706.03872), 2017.
- Papineni et al. [BLEU](https://aclanthology.org/P02-1040/), 2002 · Chen, Cherry. [Smoothing techniques for sentence-level BLEU](https://aclanthology.org/W14-3346/), 2014 · Popović. [chrF](https://aclanthology.org/W15-3049/), 2015.
- Bishop, Bishop. Deep Learning: Foundations and Concepts, гл. 12 (внимание как ядерная регрессия) · d2l.ai: [Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html).

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL, лекция 8: представления текста, seq2seq, внимание · DL: повторение (обзор курса)
- **Семинары:** [Семинар 8: seq2seq и внимание](../../materials/seminars/22_Семинар_8_Seq2seq_Attention.ipynb) ([Colab](https://colab.research.google.com/drive/1R8bRUzgymyISMdYAubDVHmoqENDbjtE8))
- **Видео и внешние курсы:** [Лена Войта — Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) · [ Видео: «Seq2seq and Attention»](https://disk.yandex.ru/i/Nb3BT79-a4hbWQ) · [Яндекс: нейросети для последовательностей](https://contest.yandex.ru/tracks/ml/deep-learning-architectures/neural-networks-for-sequences) · [Курс NLP ШАД (yandexdataschool/nlp_course)](https://github.com/yandexdataschool/nlp_course) · [Stanford CS224n — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)

</details>

---

[← 21. Представления текста: от мешка слов до word2vec, BPE и контекстных эмбеддингов](../21_word_embeddings_tokenization/README.md) · [Оглавление курса](../../README.md) · [23. Трансформер и BERT →](../23_transformer_bert/README.md)
