# 22. Seq2seq и механизм внимания

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 21. Представления текста](../21_word_embeddings_tokenization/README.md) · [23. Трансформер и BERT →](../23_transformer_bert/README.md)

---

## Часть A. Seq2seq

### 1. Задача и формулировка

Sequence-to-sequence: вход — последовательность $x = (x_1,\ldots,x_{T_x})$, выход — последовательность $y$ другой длины. Главный
пример — машинный перевод; также суммаризация, диалог, распознавание речи, генерация кода, перевод между языками программирования,
исправление опечаток, image captioning (вход — картинка). Порядок слов в переводе меняется («le chat noir» → «the black cat»), длины
не совпадают — поэтому синхронная RNN ([тема 20](../20_rnn/README.md)) не подходит.

Три вопроса (по курсу Лены Войты):
1. **Моделирование:** как устроена $p(y\mid x,\theta)$?
2. **Обучение:** как найти $\theta$?
3. **Вывод (inference):** как найти $y^{*} = \arg\max_y p(y\mid x,\theta)$?

### 2. Encoder–decoder и условная языковая модель

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

### 3. Обучение

Кросс-энтропия по всем позициям цели, усреднённая по токенам: $\mathcal L = -\sum_t\log p(y_t\mid y_{<t}, x)$. На входе декодера —
**истинный** предыдущий токен (teacher forcing; цель, сдвинутая вправо, с `<sos>` в начале). Это ровно MLE и позволяет считать все
шаги за один проход. Минус — **exposure bias** (на тесте модель видит свои ошибки); частичные лекарства — смешивание с собственными
прогнозами (`teacher_forcing_ratio` в семинаре, scheduled sampling), подробнее — [тема 20, §14](../20_rnn/README.md#14-обучение-генерации-teacher-forcing-и-его-альтернативы).

Технически: `<pad>` в цели игнорируется (`ignore_index`), паддинг во входе — маскируется (packing, маска внимания); `<eos>` — часть
цели, иначе модель не научится останавливаться.

### 4. Вывод: greedy и beam search

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

### 5. Оценка: BLEU и другие

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

---

## Часть B. Механизм внимания

### 6. Узкое горло

Весь вход сжат в **один вектор фиксированного размера** — плохо, особенно для длинных предложений: информация о начале затирается,
качество перевода падает с длиной. L1-пример (разворот цифр): без внимания точность 0.83 на длинах 4–8 и 0 на 13–16; с вниманием —
0.98 и 0.92.

### 7. Идея (Bahdanau, Cho, Bengio, 2015)

Энкодер передаёт декодеру **не одно состояние, а состояния всех токенов**. На каждом шаге декодер сам решает, на какие из них
смотреть:

1. **Оценки (scores):** насколько состояние энкодера $s_k$ соответствует текущему состоянию декодера $h_t$: $e_{tk} = \operatorname{score}(h_t, s_k)$.
2. **Веса:** $a_{tk} = \operatorname{softmax}_k(e_{tk})$ — неотрицательные, сумма 1.
3. **Вектор контекста:** $c_t = \sum_k a_{tk}\,s_k$ — «пулинг по схожести» для последовательности любой длины.
4. Контекст используется в декодере (конкатенируется со входом или с состоянием).

Общая форма: внимание — техника вычисления **взвешенной суммы значений (values) по запросу (query)**; веса зависят от сходства
запроса с ключами (keys). Это способ получить описание фиксированного размера по запросу из множества переменного размера.

### 8. Score-функции

| Название | $\operatorname{score}(h, s)$ | Источник |
|---|---|---|
| dot-product | $h^{\top}s$ | Luong et al., 2015 |
| general (multiplicative, bilinear) | $h^{\top}W s$ | Luong et al., 2015 |
| additive (concat, MLP) | $v^{\top}\tanh(W_1 h + W_2 s)$ | Bahdanau et al., 2015 |
| scaled dot-product | $\dfrac{q^{\top}k}{\sqrt d}$ | Vaswani et al., 2017 |
| content-based (косинус) | $\beta\cos(h, s)$ | Graves et al., 2014 (Neural Turing Machines) |

Dot требует одинаковых размерностей и не имеет параметров; general — одна матрица; additive дороже, но был лучше при больших
размерностях без масштабирования — пока не появилось деление на $\sqrt d$ (почему — [deep.md §3](deep.md#3-зачем-делить-на-корень-из-d)).

### 9. Bahdanau vs Luong

| | Bahdanau (2015) | Luong (2015) |
|---|---|---|
| Энкодер | **двунаправленный** GRU, $s_k = [\overrightarrow s_k;\overleftarrow s_k]$ | однонаправленный (стек LSTM) |
| Запрос | **предыдущее** состояние декодера $h_{t-1}$ | **текущее** $h_t$ (после шага RNN) |
| Score | additive | dot / general / concat |
| Использование $c_t$ | вход RNN-шага: $h_t = \mathrm{RNN}(h_{t-1}, [y_{t-1}; c_t])$ | после: $\tilde h_t = \tanh(W_c[h_t; c_t])$, $p = \operatorname{softmax}(W_o\tilde h_t)$ |
| Дополнительно | — | input feeding ($\tilde h_{t-1}$ — во вход шага $t$); local attention |

### 10. Виды внимания

- **Soft / global** — ко всему входу, дифференцируемо (стандарт).
- **Hard** (Xu et al., 2015, «Show, Attend and Tell») — выбирается одна позиция (сэмплирование), учится через REINFORCE; дешевле, но шумно.
- **Local** (Luong) — окно вокруг предсказанной позиции $p_t$ с гауссовым весом; компромисс для длинных входов.
- **Self-attention / intra-attention** — к позициям **той же** последовательности: каждое слово собирает контекст из соседей.
  Основа трансформера ([тема 23](../23_transformer_bert/README.md)).
- Внимание к картинке: декодер подписи смотрит на разные области карты признаков CNN.

### 11. Плюсы внимания

- Улучшает качество перевода (и не только).
- Решает проблему узкого горла.
- Решает проблему затухания сигнала / градиента: путь от любого слова входа до выхода — один шаг.
- Интерпретируемость: **выравнивание (alignment)** слов «бесплатно», без разметки — модель сама учится, что «européenne» ↔ «European»,
  а порядок «zone économique européenne» ↔ «European Economic Area» переставляет.

> ⚠️ Веса внимания — не строгое объяснение. Состояния энкодера смешивают контекст: в L1-примере модель выдаёт токен $x_7$, «глядя»
> на позицию 8, потому что $h_8$ однонаправленного энкодера содержит и $x_7$. См. также «Attention is not Explanation» (Jain & Wallace, 2019).

---

## Часть C. Реализация (семинар 8)

### 12. Данные

Tatoeba fr–en (`tatoebatools.ParallelCorpus('fra', 'eng')`, 440 683 пары). Нормализация: unicode → ASCII (удаление диакритики через
`unicodedata.normalize('NFD')` и категорию `Mn`), нижний регистр, отделение `!?` пробелом, всё кроме латиницы и `!?` — в пробел.
Фильтр: обе фразы короче `MAX_LENGTH = 10` слов и английская начинается с «i am / he is / she is / you are / we are / they are»
(и их сокращений «i m», «he s»…). Из первых 250 000 пар остаётся 18 757; словари: французский 6341, английский 4457 слов. Класс
`Lang`: `word2index`, `word2count`, `index2word`, `SOS = 0`, `EOS = 1`. Цели и входы дополняются нулями до `MAX_LENGTH`.

### 13. Модели

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

### 14. Что улучшить в коде семинара

| Проблема | Последствие | Исправление |
|---|---|---|
| `<pad>` = `SOS` = 0 | модель учится предсказывать `SOS` на паддинге; паддинг неотличим от начала | отдельный `PAD`, `NLLLoss(ignore_index=PAD)` |
| нет маски во внимании | часть веса уходит на паддинг | `mask = (src != PAD)` |
| декодер всегда делает `MAX_LENGTH` шагов | лишние вычисления, «хвосты» после `EOS` | остановка батча, когда все сгенерировали `EOS` |
| teacher forcing решается на весь батч | высокая дисперсия обучения | решать по шагам (scheduled sampling) |
| оценка «на глаз» | нельзя сравнивать модели | BLEU/chrF на отложенной выборке (`sacrebleu`) |
| `set_xticklabels` без `FixedLocator` | предупреждение matplotlib | `ax.set_xticks(range(n))` перед подписями |

---

## Часть D. От внимания к трансформеру

Если внимание даёт прямой доступ к любому состоянию, нужна ли рекуррентность? Трансформер (Vaswani et al., 2017, «Attention Is All
You Need») убирает RNN: **self-attention** в энкодере, **masked self-attention** + **cross-attention** (encoder–decoder attention,
ровно механизм этой темы) в декодере, позиционное кодирование вместо порядка шагов, multi-head. Подробно — [тема 23](../23_transformer_bert/README.md).

### Чек-лист

- [ ] `<sos>`, `<eos>`, `<pad>` — разные токены; паддинг игнорируется в лоссе и маскируется во внимании.
- [ ] Вход декодера — цель, сдвинутая вправо; не перепутать выравнивание логитов и целей.
- [ ] Inference: beam search (4–10) с нормировкой на длину; лимит длины.
- [ ] Метрика — BLEU/chrF на корпусе через `sacrebleu`, плюс ручной просмотр ошибок.
- [ ] Subword-токенизация (BPE) вместо словаря слов — [тема 21](../21_word_embeddings_tokenization/README.md).

---

## Что почитать / посмотреть

- 📘 Лена Войта: [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) (с интерактивными схемами, разделами про beam search, BPE и анализ) · 🎥 [запись лекции «Seq2seq and Attention»](https://disk.yandex.ru/i/Nb3BT79-a4hbWQ) (Яндекс Диск)
- 📘 Учебник ML Яндекса: [Нейросети для последовательностей → Seq2seq, внимание](https://education.yandex.ru/handbook/ml/article/nejroseti-dlya-raboty-s-posledovatelnostyami) · репозиторий [yandexdataschool/nlp_course](https://github.com/yandexdataschool/nlp_course) (week03 — Seq2seq and Attention)
- 📄 Stanford [CS224n](https://web.stanford.edu/class/cs224n/) — лекции про машинный перевод, seq2seq и внимание · шпаргалка [CS230: Machine translation, Beam search, BLEU, Attention](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/#machine-translation)
- 📄 Обзоры: [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) · [D. Batista — Attention in seq2seq](http://www.davidsbatista.net/blog/2020/01/25/Attention-seq2seq/) · [The AI Summer — attention](https://theaisummer.com/attention/) · [PyTorch: NLP from scratch — translation with seq2seq](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- 📄 Статьи: [Sutskever 2014](https://arxiv.org/abs/1409.3215) · [Cho 2014](https://www.aclweb.org/anthology/D14-1179/) · [Bahdanau 2015](https://arxiv.org/abs/1409.0473) · [Luong 2015](https://arxiv.org/abs/1508.04025) · [Show, Attend and Tell](http://proceedings.mlr.press/v37/xuc15.pdf) · [Neural Turing Machines](https://arxiv.org/abs/1410.5401) · [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [GNMT (длина, coverage)](https://arxiv.org/abs/1609.08144) · [BLEU](https://aclanthology.org/P02-1040/) · [sacreBLEU](https://arxiv.org/abs/1804.08771) · [chrF](https://aclanthology.org/W15-3049/) · [COMET](https://arxiv.org/abs/2009.09025) · [Attention is not Explanation](https://arxiv.org/abs/1902.10186)

## Источники курса

- `Лекция_08.pptx`, слайды 39–53: seq2seq (Sutskever, Cho), тонкости декодера, внутреннее представление предложений, обобщения, механизм внимания, score-функции, выравнивание, узкое горло, виды внимания (self, global/soft, local/hard), итоги.
- Семинар 8 «Seq2seq. Attention» ([Colab](https://colab.research.google.com/drive/1R8bRUzgymyISMdYAubDVHmoqENDbjtE8)): посимвольная GRU на «hello», корпус Tatoeba fr–en, `Lang`, нормализация, Encoder/Decoder, teacher forcing, функция внимания в матричной форме, `AttnDecoderRNN`, визуализация выравнивания.
- `Повторение.pptx` — обзорные слайды по attention и seq2seq.


<!-- materials:start -->
**Файлы курса в репозитории:**

- DL, лекция 8: представления текста, seq2seq, внимание (лекция курса, файл в репозиторий не включён)
- DL: повторение (обзор курса) (лекция курса, файл в репозиторий не включён)
- [Семинар 8: seq2seq и внимание](../../materials/seminars/22_Семинар_8_Seq2seq_Attention.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1R8bRUzgymyISMdYAubDVHmoqENDbjtE8)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [23. Трансформер и BERT →](../23_transformer_bert/README.md)
