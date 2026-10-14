# 20. Рекуррентные сети: RNN, LSTM, GRU, предобработка текста, временные ряды

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 19. Contrastive learning](../19_contrastive_embeddings/README.md) · [21. Эмбеддинги слов и токенизация →](../21_word_embeddings_tokenization/README.md)

---

## Часть A. Идея и архитектуры

### 1. Зачем отдельная архитектура для последовательностей

Текст, речь, временные ряды, видео, траектории имеют **переменную длину** и **порядок**. MLP требует вход фиксированного размера и не делит
веса между позициями; свёртка видит только локальное окно. Рекуррентная сеть (RNN) применяет **один и тот же блок** к каждому элементу и
передаёт дальше **скрытое состояние** $h_t$ — сжатую «память» о прочитанном. Главная идея — **разделение параметров** (parameter sharing)
по времени, как свёртка делит веса по пространству: одна модель для последовательности любой длины, число параметров от длины не зависит.

| Режим | $T_x$ / $T_y$ | Пример |
|---|---|---|
| one-to-one | 1 / 1 | обычная сеть |
| one-to-many | 1 / >1 | подпись к изображению, генерация музыки |
| many-to-one | >1 / 1 | тональность текста, классификация видео |
| many-to-many синхронный | $T_x=T_y$ | теги частей речи (PoS), NER, покадровая разметка |
| many-to-many через encoder–decoder | $T_x\ne T_y$ | машинный перевод, суммаризация ([тема 22](../22_seq2seq_attention/README.md)) |

Где применяли (и где RNN-идеи живут до сих пор): языковые модели, перевод, распознавание речи и рукописного текста, подписи к картинкам,
visual question answering, генерация рукописного текста, прогноз временных рядов.

### 2. Vanilla RNN

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h),\qquad y_t = W_{hy}h_t + b_y ,\qquad h_0 = 0 .
$$

Это обычный однослойный перцептрон, на вход которому подали конкатенацию $[x_t, h_{t-1}]$. «Развернув» сеть во времени, получаем глубокую
сеть из $T$ слоёв **с общими весами**. Потенциальные проблемы видны сразу: (1) **забывание** — чтобы ответить в конце, надо помнить начало;
(2) **градиенты** через $T$ одинаковых слоёв затухают или взрываются.

Параметров: $H(D+H) + H$ (+ $H$ ещё раз в PyTorch, где два вектора bias `bias_ih` и `bias_hh`).

### 3. Обучение: BPTT

**Backpropagation through time** — обычный backprop по развёрнутому графу: проход вперёд по всей последовательности, потом назад. Потери
суммируются по шагам: $\mathcal L = \sum_t \mathcal L_t(\hat y_t, y_t)$, и градиент по общим весам — сумма вкладов всех шагов.

- Память: нужно хранить активации всех $T$ шагов — $O(T\cdot H)$ на объект.
- **Truncated BPTT:** длинную последовательность режут на куски по $k$ шагов, состояние передают дальше, а граф отсекают:
  `h = h.detach()`. Градиент не видит зависимостей длиннее $k$, зато обучение возможно на бесконечном потоке.
- Время: шаги **последовательные** — $T$ шагов нельзя распараллелить (главный минус против трансформера).

### 4. Затухающие и взрывающиеся градиенты — главная проблема RNN

$$
\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \operatorname{diag}\big(1 - h_t^2\big)\,W_{hh} .
$$

В обычной сети перемножаются **разные** матрицы, в RNN — **одна и та же** $W_{hh}$, то есть, по сути, возводится в степень. Собственные
значения больше 1 — градиенты **взрываются**, меньше 1 — **затухают** экспоненциально (вывод — [deep.md §2](deep.md#2-произведение-якобианов-затухание-и-взрыв)).
Затухание опаснее: оно молчаливое — loss просто не учит дальние зависимости.

| Против взрыва | Против затухания |
|---|---|
| **клиппинг градиента** по норме (стандарт, порог ≈ 1–5) | **гейты**: LSTM, GRU (аддитивная память) |
| регуляризация весов | инициализация: ортогональная $W_{hh}$ (все $\lvert\lambda\rvert=1$), bias forget-гейта ≈ 1 |
| truncated BPTT | leaky units — «прокидывание» состояния, аналог шортката |
| teacher forcing (короче цепочки ошибок) | специальные регуляризаторы градиента; методы второго порядка |
| эхо-сети (ESN): рекуррентные веса не учим | ReLU-RNN с инициализацией $W_{hh}=I$ (IRNN) |

**Резервуарные вычисления** (Echo State Networks, Liquid State Machines): рекуррентную часть («резервуар») задают случайно с
спектральным радиусом < 1, чтобы история «отзывалась эхом» и затухала, и **учат только выходной линейный слой** — градиентов через время нет.

### 5. LSTM (Hochreiter & Schmidhuber, 1997)

Два состояния: **cell state** $c_t$ — «конвейер памяти», и **hidden state** $h_t$ — выход ячейки. Ключевая идея: $c_t$ проходит через все
шаги почти без преобразований — **только поэлементное умножение и сложение**, как шорткат в ResNet. Информация, которая должна «слабо
меняться», переносится свободно, градиент — тоже.

$$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f) && \text{forget: что оставить из старой памяти}\\
i_t &= \sigma(W_i[h_{t-1}, x_t] + b_i) && \text{input: сколько записать нового}\\
\tilde c_t &= \tanh(W_c[h_{t-1}, x_t] + b_c) && \text{кандидат: что записать}\\
c_t &= f_t\odot c_{t-1} + i_t\odot\tilde c_t && \text{обновление памяти}\\
o_t &= \sigma(W_o[h_{t-1}, x_t] + b_o) && \text{output: что показать наружу}\\
h_t &= o_t\odot\tanh(c_t)
\end{aligned}
$$

- $f=1$ — память передаётся целиком, $f=0$ — забыта (строгих равенств не бывает: это сигмоида).
- **Bias forget-гейта ≈ 1** (Gers 2000; Jozefowicz 2015): при $b_f=0$ гейт $\approx0.5$, и память всё равно тает как $0.5^k$. Код инициализации из лекции — ниже.
- **Peephole-связи** (Gers & Schmidhuber): гейты видят ещё и $c_{t-1}$ — «чтобы решить, что забыть, неплохо бы видеть, что забываем».
- Параметров: $4\cdot[H(D+H)+H]$ — в 4 раза больше vanilla RNN (в PyTorch веса уложены блоками в порядке **i, f, g, o**: `weight_ih_l0` размера $4H\times D$).

### 6. GRU (Cho et al., 2014)

Упрощение LSTM: нет отдельной $c_t$, **update-гейт** $z$ объединяет forget и input, **reset-гейт** $r$ решает, сколько прошлого
использовать при расчёте кандидата.

$$
z_t = \sigma(W_z[h_{t-1}, x_t]),\quad r_t = \sigma(W_r[h_{t-1}, x_t]),\quad
\tilde h_t = \tanh\big(W_h[r_t\odot h_{t-1},\,x_t]\big),\quad
h_t = (1 - z_t)\odot h_{t-1} + z_t\odot\tilde h_t .
$$

> ⚠️ Конвенции разные: в статье Cho и в PyTorch $h_t = z\odot h_{t-1} + (1-z)\odot\tilde h_t$ (то есть $z\to1$ = «помнить»), у Olah и в
> семинаре — наоборот. Кроме того, в PyTorch $r$ умножается **после** матрицы: $\tanh(W_{in}x + r\odot(W_{hn}h + b_{hn}))$.

**LSTM или GRU?** Большие сравнения («LSTM: A Search Space Odyssey», Greff 2015; Jozefowicz 2015) не выявили явного победителя: GRU на
четверть легче и быстрее, LSTM чуть мощнее на некоторых задачах; самое важное в LSTM — forget-гейт и его bias. Оптимальные
гиперпараметры у них разные. На семинаре 7 GRU ≈ LSTM.

### 7. Двунаправленные, глубокие и другие

- **Bidirectional RNN:** два независимых прохода → и ←, на каждом шаге конкатенация $[\overrightarrow h_t;\overleftarrow h_t]$. Нужна
  **вся последовательность заранее** — подходит для разметки, классификации, распознавания речи/рукописного текста, биоинформатики;
  не подходит для генерации и онлайн-прогноза.
- **Deep (stacked) RNN:** выход слоя $l$ — вход слоя $l+1$ (`num_layers`). Обычно 2–4 слоя; больше — трудно учить без шорткатов.
- **Многомерные RNN** (2D-рекуррентность по изображению), **Hierarchical Subsampling Networks** (Graves) — сжатие последовательности по
  уровням. Исторически важны, сейчас редки.

### 8. Как классифицировать последовательность (many-to-one)

Нужен выход фиксированной длины. Варианты: (1) $h$ **последнего реального** токена → классификатор (возможно, многослойный);
(2) пулинг (mean/max) по всем $h_t$ с маской паддинга; (3) для bidirectional — конкатенация последнего $\overrightarrow h$ и первого
$\overleftarrow h$; (4) attention-пулинг (тема 22).

---

## Часть B. RNN в PyTorch

### 9. API и формы

```python
lstm = nn.LSTM(input_size=D, hidden_size=H, num_layers=2, bias=True,
               batch_first=True, dropout=0.2, bidirectional=True)
out, (h_n, c_n) = lstm(x)          # x: [B, T, D] при batch_first=True, иначе [T, B, D] (по умолчанию!)
# out: [B, T, 2H]  — h_t ВЕРХНЕГО слоя на всех шагах (у LSTM — h, не c)
# h_n, c_n: [num_layers * 2, B, H] — последние состояния ВСЕХ слоёв; h_n[-2], h_n[-1] — верхний слой →/←
```

- `nn.RNN` (по умолчанию `tanh`), `nn.GRU` возвращают `out, h_n`; `nn.LSTM` — `out, (h_n, c_n)`.
- `dropout` действует **только между слоями** — при `num_layers=1` он ничего не делает (и PyTorch предупреждает).
- Вопрос с лекции «почему выход — это первое из скрытых состояний?»: `out` — это $h_t$, а $c_t$ — внутренняя память, наружу не отдаётся.
- Вопрос «почему две матрицы 12×3 и два вектора длины 12 при `hidden_size=3`?»: четыре блока (i, f, g, o) × 3 = 12; `weight_ih` и `weight_hh`, `bias_ih` и `bias_hh`.
- Шаговый режим: `out, hidden = lstm(x_t.view(1, 1, -1), hidden)` в цикле — так делают при генерации.

### 10. Паддинг, длины, packing

Последовательности в батче разной длины: дополняем `<pad>` до максимальной (`pad_sequence`) и **передаём длины**. Без этого RNN
пройдёт по паддингу, и состояние «испортится». Варианты:

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

emb = self.embedding(tokens)                                    # [B, T, E]; padding_idx=0 → вектор <pad> нулевой и не учится
packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
out_packed, h_n = self.gru(packed)                              # h_n — у последнего РЕАЛЬНОГО токена (и для ← тоже)
out, _ = pad_packed_sequence(out_packed, batch_first=True)      # если нужны все шаги

# Альтернатива семинара (только для однонаправленной сети): взять out у последнего реального токена
last = out[torch.arange(B), lengths - 1]                        # при batch_first=True
```

Для эффективности батчи собирают из последовательностей **похожей длины** (bucketing): меньше паддинга — меньше пустых вычислений.

### 11. Инициализация (из лекции, в стиле Keras)

```python
for name, p in model.named_parameters():
    if 'weight_ih' in name: nn.init.xavier_uniform_(p)
    elif 'weight_hh' in name: nn.init.orthogonal_(p)         # собственные значения по модулю 1
    elif 'bias' in name:
        nn.init.zeros_(p)
        if 'bias_ih' in name:                                # forget-гейт = второй блок из четырёх
            n = p.size(0); p.data[n // 4:n // 2].fill_(1.0)
```

### 12. Пример: тегирование частей речи (many-to-many синхронный, из лекции)

```python
class LSTMTagger(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def forward(self, sentence):                         # sentence: [T] индексов
        out, _ = self.lstm(self.emb(sentence).view(len(sentence), 1, -1))
        return F.log_softmax(self.hidden2tag(out.view(len(sentence), -1)), dim=1)   # [T, n_tags] → NLLLoss
```

### 13. Пример: классификатор тональности (many-to-one, семинар 7)

```python
class RNNClassifier(nn.Module):
    def __init__(self, emb_dim, hidden, n_classes, vocab_size, rec_layer=nn.RNN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = rec_layer(emb_dim, hidden)            # формат [T, B, E]
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, tokens, lens):                     # tokens: [T, B]
        out, _ = self.rnn(self.emb(tokens))              # [T, B, H]
        last = out[lens - 1, torch.arange(tokens.shape[1])]
        return self.fc(last)

# обучение: CrossEntropyLoss, Adam(lr=1e-2), clip_grad_norm_(model.parameters(), 1.0) после backward
```

Результаты семинара (твиты, 3 класса: neutral / negative / positive, словарь 5000, `max_len=200`, 5–10 эпох, macro-F1 на тесте):

| Модель | без чистки текста | с чисткой (нижний регистр, только буквы, без стоп-слов) |
|---|---|---|
| RNN | 0.41 | 0.63 |
| LSTM | 0.67 | 0.70 |
| GRU | 0.67 | 0.68 |

Выводы: vanilla RNN учится нестабильно (accuracy скачет по эпохам), гейтовые модели заметно лучше; чистка текста сильнее всего помогает
слабой модели (меньше словарь → меньше `<unk>` и редких токенов). LSTM с 3-й эпохи переобучается (train 0.88, test 0.67) — нужны
регуляризация и ранняя остановка.

---

## Часть C. Приёмы обучения

### 14. Обучение генерации: teacher forcing и его альтернативы

Языковая модель на RNN: на вход — предыдущий токен, на выходе — распределение следующего.

- **Teacher forcing:** на обучении вместо выхода модели на прошлом шаге подаём **истинный** токен. Плюсы: быстро сходится, шаги можно
  считать параллельно по слоям, это ровно максимизация правдоподобия. Минус — **exposure bias**: на тесте модель кормит себя своими
  прогнозами, видит ситуации, которых не было на обучении, и ошибки накапливаются. Применяется и в трансформерах.
- **Scheduled sampling** (Bengio et al., 2015): с вероятностью $\varepsilon_i$ подаём истинный токен, иначе — свой; $\varepsilon_i$
  постепенно уменьшается (линейно, экспоненциально, обратной сигмоидой).
- **Professor forcing** (Goyal et al., 2016): режимы «с учителем» и «свободной генерации» обучаются одновременно, а дискриминатор
  требует, чтобы скрытые состояния в обоих режимах были неразличимы.
- Код из лекции (PyTorch seq2seq tutorial): `use_teacher_forcing = random.random() < teacher_forcing_ratio` (0.5), и в ветке без учителя
  `decoder_input = topi.squeeze().detach()`.

**Когда остановить генерацию?** Спецтокен `<eos>` в словаре (модель сама предсказывает конец) или отдельный выход «вероятность
окончания» — так можно генерировать и последовательности чисел. Ещё защита — лимит длины.

### 15. Регуляризация RNN

| Метод | Идея |
|---|---|
| Dropout только на нерекуррентных связях (Zaremba 2014) | маска на входах/выходах слоёв, связи $h_{t-1}\to h_t$ не трогаем |
| Вариационный dropout (Gal & Ghahramani 2016) | **одна маска на все шаги** (и на рекуррентных связях) — теоретически обоснован как байесовское приближение |
| Recurrent dropout без потери памяти (Semeniuta 2016) | дропаем только **добавку** $\tilde c_t$, а не саму память: состояние искажается слабо |
| Zoneout | случайно оставляем часть нейронов в прежнем состоянии $h_t=h_{t-1}$ |
| Curriculum / Fraternal dropout | интенсивность растёт со временем / две маски + штраф за расхождение предсказаний |
| BatchNorm (Laurent 2015; Cooijmans 2016) | только на «вертикальных» связях или со статистиками по шагам; проблемы с разной длиной |
| **LayerNorm** (Ba et al. 2016) | нормировка по признакам внутри объекта — не зависит от батча и длины; стандарт для RNN |
| Weight tying | общие веса у входного эмбеддинга и выходного слоя языковой модели |

### 16. Интерпретация

- Karpathy, Johnson, Fei-Fei (2016): в символьной LSTM нашлись нейроны-«счётчики» позиции в строке, индикаторы «внутри кавычек»,
  «внутри комментария», глубины вложенности скобок.
- Li et al. (2015): визуализация вкладов слов в тональность на Stanford Sentiment Treebank.
- **Sentiment neuron** (OpenAI, Radford et al., 2017): mLSTM, обученная просто предсказывать следующий символ отзывов Amazon, получила
  один нейрон, линейно кодирующий тональность; фиксируя его значение, можно генерировать позитивные или негативные тексты.

### 17. Практические советы (из лекции)

1. Брать GRU или LSTM, не vanilla RNN.
2. Ортогональная инициализация $W_{hh}$, bias forget-гейта ≈ 1, Adam, клиппинг градиента ≈ 1.
3. Начинать с простой модели и маленького датасета — добиться переобучения; потом усложнять и регуляризовать.
4. Смотреть на статистику данных и ответов модели (длины, классы, ошибки).

---

## Часть D. Предобработка текста (семинар 7, часть 1)

Подробнее о векторизации и токенизации — в [теме 21](../21_word_embeddings_tokenization/README.md).

### 18. Конвейер

1. **Нормализация:** `text.lower()`, удаление/замена символов регулярками.
2. **Токенизация:** `text.split()`, `re.split`, `nltk.word_tokenize`; в современных моделях — subword (BPE/WordPiece, тема 21).
3. **Стемминг / лемматизация:** стемминг отрезает окончания по правилам (Snowball); лемматизация приводит к словарной форме
   (`pymorphy3`: «люблю» → «любить», «году» → «год»). Для русского лемматизация намного полезнее английского.
4. **Стоп-слова:** `nltk.corpus.stopwords.words('russian')` (151 слово) — частые слова без смысла.
5. **Векторизация:** словарь → индексы → эмбеддинги.

```python
import re, pymorphy3
from nltk.corpus import stopwords

morph = pymorphy3.MorphAnalyzer()
stop = set(stopwords.words('russian'))

def preprocess(text):
    text = re.sub(r'[^а-яёa-z\s]', ' ', text.lower())                       # оставить буквы и пробелы
    return [w for w in (morph.parse(t)[0].normal_form for t in text.split()) if w not in stop]
```

### 19. Шпаргалка по регулярным выражениям (`re`)

| Функция | Что делает |
|---|---|
| `re.search(p, s)` | первое совпадение → `Match` (`m[0]`, `m.span()`) или `None` |
| `re.findall(p, s)` | список всех непересекающихся совпадений |
| `re.split(p, s)` | разбить строку по шаблону (`re.split('!', 'a!!b')` → `['a', '', 'b']`) |
| `re.sub(p, r, s)` | заменить все совпадения |

| Элемент | Значение |
|---|---|
| `.` `*` `+` `?` | любой символ (кроме `\n`) / 0+ / 1+ / 0–1 повторов |
| `{n}` `{n,}` `{n,m}` | ровно $n$ / от $n$ / от $n$ до $m$ повторов |
| `[abc]` `[a-z]` `[^abc]` | множество / диапазон / отрицание |
| `^` `$` | начало / конец строки |
| `\d` `\D` `\w` `\W` `\s` `\S` | цифра / не цифра / буква-цифра-`_` / не она / пробельный / не пробельный |
| `( )` `a\|b` | группа / «или» |

Шаблоны пишите **raw-строками** `r'...'` — иначе `'\s'` даёт `SyntaxWarning` (в семинаре так и было).

### 20. Словарь, специальные токены, Dataset и collate

- Считаем частоты токенов на **train**, сортируем по убыванию, берём `max_size` (в семинаре 5000) самых частых.
- Специальные токены: `<pad>` (индекс 0, паддинг), `<unk>` (индекс 1, слово вне словаря), `<sos>`/`<eos>` (начало/конец — для генерации).
- `stoi` (string-to-index, словарь) и `itos` (index-to-string, список) — взаимно обратные.
- Dataset хранит индексы и **длины**; `max_len` обрезает длинные тексты.
- `collate_fn` собирает батч: `pad_sequence(tokens, batch_first=False, padding_value=stoi['<pad>'])` → `[T_max, B]` + длины + метки.

```python
def collate_fn(batch, pad_id=0):
    tokens = pad_sequence([b['tokens'] for b in batch], padding_value=pad_id)    # [T_max, B]
    lens = torch.stack([b['tokens_len'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    return {'tokens': tokens, 'tokens_lens': lens, 'labels': labels}

loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
```

### 21. Ловушки предобработки (найдены в семинаре)

- `re.sub(r'[^\sa-zа-я]', '', text)` удаляет **«ё»** (она вне диапазона `а-я` в Unicode) и цифры — «всё» становится «вс».
- Стоп-слова NLTK для английского содержат `not`, `no`, `nor`, `don't`: для **тональности** их удаление превращает «not good» в «good».
- Удаление эмодзи и пунктуации («!!!», «:)))») выбрасывает сильные сигналы тональности — проверяйте на валидации, а не по умолчанию.
- Словарь строится только по train; на тесте неизвестные слова → `<unk>`.
- `pd.Series.replace({'neutral': 0, ...})` даёт `FutureWarning` о downcasting — надёжнее `.map(...)`.

---

## Часть E. Временные ряды (ноутбук `rnn-international-airline-passengers`)

Месячные перевозки авиапассажиров 1949–1960 (144 точки, тыс. человек): **тренд** + **сезонность** с растущей амплитудой
(мультипликативная). Ноутбук строит окна `window=10`, прогноз на `gap=2` шага вперёд, 5 слоёв `SimpleRNN` (Keras, ReLU, dropout 0.2),
250 эпох. Итог в ноутбуке — Train MSE 481 и Test MSE 62 352: модель не работает. Причины — хорошая коллекция типичных ошибок:

| Ошибка в ноутбуке | Почему плохо | Как правильно |
|---|---|---|
| `prepare(dataset, ...)` берёт окна из глобального `train`, а не из аргумента | тестовые входы на самом деле из train — оценка бессмысленна | использовать аргумент функции; тесты на формы и значения |
| `MinMaxScaler.fit_transform` на **всём** ряде | утечка: масштаб знает о будущих максимумах | `fit` только на train |
| тренд: тест выходит за диапазон train (max 467 → 622) | модель с насыщающимися активациями не экстраполирует | log + разности (сезонная $\log y_t - \log y_{t-12}$) |
| 5 рекуррентных слоёв на 73 обучающих окна | переобучение | маленькая модель, регуляризация |
| вывод «MAE», а считается MSE | путаница в метриках | называть метрики правильно; RMSE/MAE в исходных единицах |
| нет бейзлайнов | непонятно, хороша ли модель | наивный $y_{t-1}$, сезонный наивный $y_{t-12}$ |

Шпаргалочный пример (L1) показывает: на сыром ряде GRU хуже сезонного наивного прогноза (RMSE ≈ 74 против 42), а на сезонной
разности логарифма — 15.6 против 17.7 у простой формулы «сезонный × годовой рост». **Подготовка ряда важнее архитектуры.** Сплит —
только по времени, без перемешивания; валидация — скользящим окном (`TimeSeriesSplit`, [тема 05](../05_validation/README.md)).
Для честности гиперпараметры (`H`, число эпох) нужно выбирать на отдельном валидационном отрезке, а не на тесте.

---

## Часть F. Место RNN сегодня

- Трансформер (темы 23–24) вытеснил RNN в NLP: обучение параллельно по позициям, прямой доступ к любому токену (путь длины 1 вместо $T$).
- RNN выигрывают в **инференсе**: память $O(1)$ на шаг, без растущего KV-кэша. Поэтому идеи вернулись в линейных RNN и
  state-space-моделях (S4, Mamba, RWKV), которые обучаются параллельно, а генерируют рекуррентно.
- На маленьких данных и для потоковых сигналов (сенсоры, онлайн-прогноз) GRU/LSTM — по-прежнему разумный бейзлайн.

### Чек-лист

- [ ] GRU/LSTM, а не vanilla RNN; `batch_first` указан явно.
- [ ] Паддинг + длины; `pack_padded_sequence` для bidirectional или берём $h$ последнего реального токена.
- [ ] Клиппинг градиента; ортогональная $W_{hh}$; bias forget-гейта ≈ 1.
- [ ] Dropout между слоями / вариационный; LayerNorm; ранняя остановка.
- [ ] Словарь и статистики нормировки — только по train.
- [ ] Для рядов: сплит по времени, стационаризация, наивные бейзлайны.

---

## Что почитать / посмотреть

- 📘 Учебник ML Яндекса: [Нейросети для работы с последовательностями](https://education.yandex.ru/handbook/ml/article/nejroseti-dlya-raboty-s-posledovatelnostyami) (то же в треке [contest.yandex.ru](https://contest.yandex.ru/tracks/ml/deep-learning-architectures/neural-networks-for-sequences))
- 📄 Шпаргалка Stanford CS230: [Recurrent Neural Networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/)
- 📄 Блоги: [C. Olah — Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) · [A. Karpathy — The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) · [Smerity — Orthogonal init](https://smerity.com/articles/2016/orthogonal_init.html) · [Vanishing and exploding gradients (jefkine)](http://www.jefkine.com/general/2018/05/21/2018-05-21-vanishing-and-exploding-gradient-problems/) · [Обзор dropout в RNN](https://adriangcoder.medium.com/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b) · [OpenAI — Sentiment neuron](https://openai.com/blog/unsupervised-sentiment-neuron/)
- 📄 Статьи: [Hochreiter & Schmidhuber 1997 (LSTM)](https://www.bioinf.jku.at/publications/older/2604.pdf) · [Cho et al. 2014 (GRU)](https://arxiv.org/abs/1406.1078) · [Pascanu et al. 2013 — On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063) · [Greff et al. — LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069) · [Jozefowicz et al. 2015](http://proceedings.mlr.press/v37/jozefowicz15.pdf) · [Zaremba 2014](https://arxiv.org/abs/1409.2329) · [Gal & Ghahramani 2016](https://arxiv.org/abs/1512.05287) · [Semeniuta 2016](https://arxiv.org/abs/1603.05118) · [Curriculum Dropout](https://arxiv.org/abs/1703.06229) · [Fraternal Dropout](https://arxiv.org/abs/1711.00066) · [Laurent 2015 (BN-RNN)](https://arxiv.org/abs/1510.01378) · [Cooijmans 2016 (Recurrent BN)](https://arxiv.org/abs/1603.09025) · [Scheduled Sampling](https://arxiv.org/abs/1506.03099) · [Professor Forcing](https://arxiv.org/abs/1610.09038) · [Karpathy 2015 — Visualizing RNNs](https://arxiv.org/abs/1506.02078) · [Li 2015 — Visualizing neural models in NLP](https://arxiv.org/abs/1506.01066) · [Salehinejad — Recent Advances in RNNs (обзор)](https://arxiv.org/abs/1801.01078) · [Lukoševičius & Jaeger — Reservoir computing approaches](https://doi.org/10.1016/j.cosrev.2009.03.005) · [A. Graves — Supervised Sequence Labelling with RNNs](https://www.cs.toronto.edu/~graves/preprint.pdf) · [J. Guo — BPTT notes](http://ir.hit.edu.cn/~jguo/docs/notes/bptt.pdf)
- 📄 Документация: [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) · [Sequence models tutorial (LSTMTagger)](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) · [Seq2seq translation tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- 📘 Книги: [Deep Learning Book, гл. 10](https://www.deeplearningbook.org/) · 📄 [Distill.pub](https://distill.pub/) · [G. Sapunov — Multidimensional RNN](https://www.slideshare.net/grigorysapunov/multidimensional-rnn)

## Источники курса

- `RNN_лекция_оформленная_презентация.pptx` (DEEP LEARNING, день 7, 66 слайдов): идея и вычислительный граф, BPTT, LSTM/GRU, минутки кода, teacher forcing, двунаправленные и глубокие RNN, затухание/взрыв градиентов, резервуарные вычисления, dropout и batchnorm в RNN, интерпретация, советы.
- Семинар 7 «Рекуррентные нейросети (LSTM, внимание). Предобработка текста» ([Colab](https://colab.research.google.com/drive/1D_XIvuqGltfgfkJ2sDcwh5ivuPFKW6ue)): нормализация, регулярки, лемматизация, стоп-слова, словарь, Dataset/collate, RNN/LSTM/GRU для тональности, влияние предобработки.
- `rnn-international-airline-passengers.ipynb` + `international-airline-passengers.csv`: прогноз временного ряда на Keras `SimpleRNN`.


<!-- materials:start -->
**Файлы курса в репозитории:**

- DL, лекция 7: рекуррентные сети (лекция курса, файл в репозиторий не включён)
- DL: повторение (обзор курса) (лекция курса, файл в репозиторий не включён)
- [RNN для временного ряда авиапассажиров](../../materials/seminars/20_rnn_airline_passengers.ipynb)
- [Данные: авиапассажиры 1949–1960](../../materials/data/international-airline-passengers.csv)
- [Семинар 7: RNN, LSTM, предобработка текста](../../materials/seminars/20_Семинар_7_RNN_предобработка_текста.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1D_XIvuqGltfgfkJ2sDcwh5ivuPFKW6ue)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [21. Эмбеддинги слов и токенизация →](../21_word_embeddings_tokenization/README.md)
