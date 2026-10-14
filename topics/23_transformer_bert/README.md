# 23. Трансформер и BERT

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 22. Seq2seq и внимание](../22_seq2seq_attention/README.md) · [24. Языковые модели и GPT →](../24_language_models_gpt/README.md)

---

## Часть A. Архитектура

### 1. Идея: «parallelized attention»

Vaswani et al. (2017), «Attention Is All You Need»: оставить от seq2seq с вниманием ([тема 22](../22_seq2seq_attention/README.md))
только внимание. Вместо RNN — **self-attention**: представление каждого токена уточняется с учётом **всех** токенов сразу.
Следствия: обучение параллельно по позициям (нет последовательных шагов), путь между любыми двумя словами — один слой,
масштабируемость на огромные данные. Цена — $O(T^2)$ по длине.

Общая схема — снова «кодировщик–декодировщик», 6 + 6 одинаковых по архитектуре блоков **с разными весами**. На входе — эмбеддинги
токенов (после subword-токенизации, [тема 21](../21_word_embeddings_tokenization/README.md)) размерности $d_{\text{model}} = 512$; каждый
блок превращает векторы в векторы **той же размерности** — «представления следующего уровня».

### 2. Query, key, value — матричная запись

Для входа $X\in\mathbb R^{T\times d}$ три обучаемые матрицы дают

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\qquad
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)V .
$$

Строка $i$ матрицы весов — распределение внимания токена $i$ по всем токенам; выход — взвешенная сумма значений. В оригинале
$d_k = d_v = 64$. Пример из лекции: в «The animal didn't cross the street because **it** was too tired» представление «it»
на следующем слое должно впитать «animal»; к чему относится «making» в другой фразе — тоже видно по весам (визуализатор exBERT).
Деление на $\sqrt{d_k}$ — [deep тема 22, §3](../22_seq2seq_attention/deep.md#3-зачем-делить-на-корень-из-d).

Self-attention можно рассматривать как **полный граф** на токенах с **пулингом по релевантности**: веса рёбер вычисляются из данных.

```python
def attention(query, key, value, mask=None, dropout=None):        # «The Annotated Transformer»
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn @ value, p_attn
```

### 3. Multi-head attention

$h$ голов, каждая — внимание в своём подпространстве размерности $d_k = d/h$; результаты конкатенируются и проецируются:

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\,W_O,\qquad
\mathrm{head}_i = \operatorname{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)}).
$$

Каждая головка «обозревает свой аспект»: синтаксические связи, кореференцию, соседние слова, разделители. Стоимость та же, что у одной
головы полной размерности. Реализация — одна линейная проекция $d\to d$ и `view(B, T, h, d_k).transpose(1, 2)`:

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k, self.h = d_model // h, h
        self.linears = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(4))   # W_Q, W_K, W_V, W_O
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)                                  # одна маска на все головы
        B = query.size(0)
        q, k, v = [l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

### 4. Позиционное кодирование

Self-attention **эквивариантен к перестановкам**: без информации о позиции «собака кусает человека» и «человек кусает собаку» дают
одинаковый набор векторов. Решение — прибавить к эмбеддингу вектор позиции. Требования: уникален для позиции, не зависит от длины
предложения, ограничен, позволяет выражать относительные сдвиги. Синусоиды:

$$
\mathrm{PE}_{(p, 2i)} = \sin\frac{p}{10000^{2i/d}},\qquad \mathrm{PE}_{(p, 2i+1)} = \cos\frac{p}{10000^{2i/d}} .
$$

Длины волн — геометрическая прогрессия от $2\pi$ до $10000\cdot2\pi$: младшие координаты меняются быстро, старшие — медленно (как
разряды двоичного счётчика, но гладко). Сдвиг на $k$ — **линейное** преобразование (поворот в каждой паре координат, [deep §2](deep.md#2-синусоидальные-позиции-сдвиг--поворот)).
В статье обучаемые позиции дали то же качество; синусоиды выбраны в надежде на экстраполяцию за длину обучения. BERT и GPT-2 учат
позиции; современные LLM — **RoPE** (поворот $q$ и $k$, скалярное произведение зависит только от $i - j$) или ALiBi (штраф за расстояние).

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))   # в лог-пространстве
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                   # буфер, не параметр
    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])
```

Эмбеддинги умножают на $\sqrt{d}$, чтобы позиционный сигнал не доминировал над смысловым.

### 5. Residual + LayerNorm, FFN

Каждый подблок (внимание, FFN) обёрнут в **прокидывание связи** и **LayerNorm**. Оригинал — **post-LN**: $\mathrm{LN}(x + \mathrm{Dropout}(\mathrm{Sublayer}(x)))$.
«Annotated Transformer» и большинство современных моделей — **pre-LN**: $x + \mathrm{Dropout}(\mathrm{Sublayer}(\mathrm{LN}(x)))$ — шорткат
остаётся «чистым», глубокие модели учатся стабильнее, меньше нужен warm-up ([deep §4](deep.md#4-pre-ln-vs-post-ln)). LayerNorm нормирует по
признакам каждого токена (не по батчу) — подходит для последовательностей разной длины; в LLM часто RMSNorm (без центрирования).

**FFN** — двухслойный перцептрон, **одинаковый для всех позиций** (разделение параметров), «inverted bottleneck» $d\to4d\to d$:

$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)\,W_2 + b_2\quad(\text{ReLU; в BERT/GPT — GELU, в LLaMA — SwiGLU}).
$$

Внимание смешивает информацию **между** токенами, FFN преобразует каждый токен **отдельно** — в нём сосредоточено ~2/3 параметров
и, по ряду работ, «знания» модели (FFN как память ключ–значение).

### 6. Декодер: маски и cross-attention

Три вида внимания:

| Где | Вид | Q | K, V | Маска |
|---|---|---|---|---|
| энкодер | MHSA (self-attention) | энкодер | энкодер | padding |
| декодер, 1-й подблок | **Masked** MHSA | декодер | декодер | padding + **causal** (будущее скрыто) |
| декодер, 2-й подблок | MHCA (**cross-attention**) | декодер | **выход энкодера** | padding источника |

Логика: в энкодере смотрим на всё предложение, чтобы понять смысл каждого токена; в декодере смотрим на энкодер (чтобы переводить, надо
видеть оригинал); при генерации нельзя знать будущее. Маски:
- **padding mask** — во всех вниманиях: не смотреть на `<pad>` (дополнение коротких предложений);
- **sequence (causal) mask** — в self-attention декодера: позиция $i$ видит $j\le i$ (нижнетреугольная матрица, `-inf` выше диагонали).

```python
def subsequent_mask(size):
    return torch.triu(torch.ones(1, size, size), diagonal=1) == 0                   # True = можно смотреть

# альтернатива (Peter Bloem): dot[:, indices[0], indices[1]] = float('-inf') с indices = torch.triu_indices(t, t, offset=1)
```

**Обучение нерекуррентное, вывод — рекуррентный.** Благодаря causal-маске декодер на обучении за один проход предсказывает все позиции
(teacher forcing: вход — цель, сдвинутая вправо; `Batch` в семинаре делит `tgt` на `tgt[:, :-1]` и `tgt_y = tgt[:, 1:]`). На инференсе длина
ответа неизвестна — генерируем по одному токену и подаём обратно. Выход декодера → линейный слой (часто с весами, общими с
эмбеддингами) → softmax по словарю → cross-entropy.

### 7. Итог по архитектуре (из лекции)

- Трансформер — стекинг трансформер-блоков, послойное уточнение представлений токенов.
- Self-attention — граф + пулинг по релевантности.
- Размерность представлений токенов не меняется по слоям (не путать с внутренними $d_k, d_v$ и размерностью входа).
- Нерекуррентная модель для последовательностей; глубокая, с блоками внимания; изначально для машинного перевода.
- Последний слой — softmax + cross-entropy.

Готовые блоки в PyTorch: `nn.Transformer`, `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer` (`norm_first=True` для pre-LN,
`batch_first=True`), `F.scaled_dot_product_attention` (сам выбирает FlashAttention / memory-efficient ядро).

---

## Часть B. Обучение трансформера

### 8. Рецепт оригинала

| Что | Значение |
|---|---|
| Данные | WMT14 en-de, ~4.5 млн пар, общий BPE-словарь ~37k; en-fr — 36 млн пар, word-piece 32k |
| Батчи | по приблизительной длине; ~25 000 токенов источника + 25 000 цели |
| Железо | 8 × P100; base: 100k шагов (0.4 с/шаг, 12 ч); big: 300k (1 с/шаг, 3.5 дня) |
| Оптимизатор | Adam, $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$ |
| LR («Noam») | $\mathrm{lr} = d^{-0.5}\min(s^{-0.5},\ s\cdot w^{-1.5})$, $w = 4000$: линейный рост, затем $\propto s^{-1/2}$ |
| Регуляризация | dropout 0.1 (на выходах подблоков и на суммах эмбеддингов с PE), **label smoothing** $\epsilon_{ls} = 0.1$ |
| Вывод | beam search $k = 4$, штраф за длину $\alpha = 0.6$ |

```python
def rate(step, model_size, factor, warmup):           # расписание из семинара 10, с LambdaLR
    step = max(step, 1)
    return factor * model_size ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: rate(s, 512, 1.0, 4000))
```

### 9. Warm-up: зачем

Лекция (по Liu et al., 2019, RAdam): в Adam оценка второго момента по первым шагам ненадёжна — дисперсия адаптивного lr огромна,
bias correction её увеличивает; плюс LayerNorm (особенно post-LN) даёт большие градиенты у последних слоёв. Без прогрева первые
шаги «выбрасывают» модель в плохую область. Сейчас обычно: линейный warm-up 1–5% шагов, затем **cosine** или линейное убывание
(`get_cosine_schedule_with_warmup` в 🤗 Transformers). Советы Popel & Bojar (2018): батч побольше (насколько позволяет память), lr
повыше, но в пределах устойчивости; качество продолжает расти долго — не останавливать рано; усреднение чекпойнтов.

### 10. Label smoothing

Вместо one-hot цели — $q = (1 - \epsilon)\,\text{one-hot} + \epsilon/(K - 1)$ на остальные классы (в семинаре — на $K - 2$, исключая
`<pad>`), лосс — KL-дивергенция (`nn.KLDivLoss`). Модель перестаёт быть «самоуверенной»: перплексия хуже, но accuracy и BLEU лучше;
калибровка лучше. Подробно — [deep §5](deep.md#5-label-smoothing).

---

## Часть C. Три семейства и BERT

### 11. Encoder-only, decoder-only, encoder–decoder

| Семейство | Внимание | Предобучение | Задачи | Модели |
|---|---|---|---|---|
| encoder-only | двунаправленное | MLM (+NSP) | классификация, NER, поиск, эмбеддинги | BERT, RoBERTa, DeBERTa, ruBERT |
| decoder-only | causal | следующий токен (LM) | генерация, всё через промпт | GPT, LLaMA, Qwen, YandexGPT ([тема 24](../24_language_models_gpt/README.md)) |
| encoder–decoder | энкодер полное, декодер causal + cross | denoising (span corruption) | перевод, суммаризация | оригинал, T5, BART, mT5 |

### 12. BERT (Devlin et al., 2018)

**Bidirectional Encoder Representations from Transformers.** Идея из GPT — предобучить языковую модель на большом корпусе и
дообучать под задачи (transfer learning), — но с **двусторонним** контекстом. Отличия от ELMo ([тема 21](../21_word_embeddings_tokenization/README.md#11-elmo-peters-et-al-2018)):
вместо двух независимых LSTM — один трансформер с полным обзором; отличия от GPT — нет causal-маски.

**Главный подводный камень двусторонности:** если просто учить предсказывать слово по всем остальным, в глубокой сети слово «увидит
себя» через другие позиции. Решение — **Masked LM**:
- выбираем **15%** токенов и предсказываем только их;
- из выбранных **80%** заменяем на `[MASK]` («went to the store» → «went to the [MASK]»), **10%** — на случайное слово («went to the running»),
  **10%** оставляем как есть.

Зачем 80/10/10: `[MASK]` не встречается при дообучении (рассогласование); случайные и нетронутые токены не дают модели расслабиться —
она должна строить хорошее представление **каждого** токена, ведь любой может оказаться «подменённым».

**Next Sentence Prediction:** вход — пара предложений A, B; в 50% случаев B действительно следует за A, в 50% — случайное. Классификация
по `[CLS]`. Задумано для QA и NLI; RoBERTa показала, что NSP не нужен (лучше — длинные непрерывные куски текста).

**Вход:** `[CLS] A [SEP] B [SEP]`; каждый токен = сумма **трёх** эмбеддингов: токен (WordPiece, 30 000) + сегмент (A/B) + позиция (обучаемая,
до 512).

| | BERT-Base | BERT-Large |
|---|---|---|
| слоёв / скрытый размер / голов | 12 / 768 / 12 | 24 / 1024 / 16 |
| параметров | 110M | 340M |

Данные: Wikipedia (2.5 млрд слов) + BookCorpus (800 млн). Батч 256 × 512 = 131 072 токена (сначала 90% шагов с длиной 128), 1M шагов
(~40 эпох), AdamW, lr 1e-4 с прогревом 10 000 шагов и линейным убыванием, dropout 0.1, GELU; 4 дня на 16/64 TPU-чипах (Base/Large).

### 13. Дообучение (fine-tuning)

- **Классификация предложения/пары:** финальное состояние `[CLS]` × обучаемая матрица → softmax.
- **Разметка токенов (NER, POS):** линейный слой на каждый токен.
- **Extractive QA (SQuAD):** два обучаемых вектора $S$, $E$; $p_{\text{start}}(i) = \operatorname{softmax}_i(S^{\top}h_i)$, аналогично конец; ответ — отрезок с
  максимальным $S^{\top}h_i + E^{\top}h_j$, $j\ge i$.
- Гиперпараметры дообучения из статьи: lr 2e-5 – 5e-5, батч 16–32, 2–4 эпохи.

Эффекты (лекция): для одних задач важнее MLM, для других — NSP; модели «слева направо» плохи на SQuAD; **размер имеет значение** даже
на небольших датасетах, и кривая не вышла на асимптоту. Задачи GLUE: CoLA (грамматичность), QQP (парафразы), MNLI (логический вывод),
SST-2 (тональность) и др.

**Признаки из BERT без дообучения** — как в ELMo, комбинировать слои: для NER конкатенация последних четырёх слоёв почти не уступает
дообучению. Вектор слова зависит от контекста: в L1-примере «bank» в финансовых предложениях — косинус 0.90, в «речных» — 0.84,
между группами — 0.63.

### 14. Семинар 9: NER с Hugging Face

Экосистема 🤗: **Transformers** (модели), **Datasets** (`load_dataset`), **Tokenizers** (быстрые токенизаторы), **Hub** (модели и датасеты),
**Spaces** (демо). Задача — классификация токенов на WikiANN-ru (`unimelb-nlp/wikiann`, `"ru"`: 20 000 / 10 000 / 10 000): теги
`O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC` (схема **BIO**: B — начало сущности, I — продолжение).

```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
                          TrainingArguments, Trainer, pipeline)

raw = load_dataset("unimelb-nlp/wikiann", "ru")
label_names = raw["train"].features["ner_tags"].feature.names
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def align_labels_with_tokens(labels, word_ids):          # слово → несколько subword-токенов
    new, prev = [], None
    for w in word_ids:
        if w is None: new.append(-100)                   # [CLS], [SEP] — не учитываются в лоссе
        elif w != prev: new.append(labels[w])            # первый подтокен — метка слова
        else:
            l = labels[w]; new.append(l + 1 if l % 2 == 1 else l)   # B-X → I-X для продолжения слова
        prev = w
    return new

def tokenize_and_align(ex):
    enc = tokenizer(ex["tokens"], truncation=True, is_split_into_words=True)
    enc["labels"] = [align_labels_with_tokens(l, enc.word_ids(i)) for i, l in enumerate(ex["ner_tags"])]
    return enc

ds = raw.map(tokenize_and_align, batched=True, remove_columns=raw["train"].column_names)
model = AutoModelForTokenClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased", id2label=dict(enumerate(label_names)),
    label2id={l: i for i, l in enumerate(label_names)})
args = TrainingArguments("rubert-finetuned-ner", eval_strategy="steps", eval_steps=0.25, learning_rate=1e-4,
                         num_train_epochs=2, per_device_train_batch_size=16, weight_decay=0.01,
                         warmup_steps=0.1, lr_scheduler_type="cosine")
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                  data_collator=DataCollatorForTokenClassification(tokenizer),   # паддинг labels значением -100
                  compute_metrics=compute_metrics, processing_class=tokenizer)
trainer.train()
ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
ner("Иван Иванович Иванов родился в Ставрополе и работает землекопом в ОАО РедПис.")
# → PER «Иван Иванович Иванов» 0.98, LOC «Ставрополе» 0.99, ORG «ОАО РедПис» 0.81
```

- `"Илизаров, Гавриил Абрамович"` → `['[CLS]', 'Или', '##зар', '##ов', ',', 'Гавриил', 'Абрамович', '[SEP]']` — `word_ids()` связывает токены со словами.
- **seqeval** считает precision/recall/F1 **по сущностям** целиком: один неверный тег внутри «Илизаров, Гавриил Абрамович» → сущность не найдена
  (F1 = 0), хотя token accuracy = 0.75.
- Предупреждение при загрузке: веса `cls.*` (MLM-голова) и `pooler` — UNEXPECTED, `classifier.*` — MISSING (новая голова, её и учим) — это нормально.
- ⚠️ В семинаре токен Hugging Face записан прямо в коде (`login(token="hf_...")`) — так делать нельзя: используйте `notebook_login()` или
  переменную окружения / секреты Colab, а утёкший токен отзовите.

### 15. Семинар 10: трансформер с нуля

По «The Annotated Transformer» (Harvard NLP): `EncoderDecoder`, `Encoder`/`Decoder` из $N = 6$ клонов слоя, `SublayerConnection`
(pre-LN + residual + dropout), `MultiHeadedAttention`, `PositionwiseFeedForward`, `Embeddings` (× $\sqrt d$), `PositionalEncoding`,
`Generator` (linear + log_softmax), `make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)` с
инициализацией Xavier. Обучение: класс `Batch` (маски, `ntokens` для нормировки лосса), `run_epoch` с аккумуляцией градиентов,
расписание Noam, `LabelSmoothing` через KL, **copy-task** (выучить копирование случайных последовательностей) как проверка, что всё
работает; жадное декодирование, beam search, визуализация матриц внимания по слоям и головам. Вторая часть семинара (pretrained vs
instruct, chat template, LoRA, PEFT) — [тема 25](../25_llm_finetuning_peft/README.md).

### 16. Тренды (итоги лекции)

Почти все лучшие решения основаны на трансформере; обучение на огромных корпусах; подстройка под задачи; «упрощение» — меньше
параметров и слоёв (DistilBERT, ALBERT, TinyBERT). Центральная идея современного NLP — **transfer learning**: предобучаемся на задаче
с большими данными и дешёвой разметкой, переносим модель и дообучаем.

### Чек-лист

- [ ] Маски: padding во всех вниманиях, causal в декодере; формы масок согласованы с `[B, h, Tq, Tk]`.
- [ ] Позиции: синусоиды/обучаемые/RoPE; длина инференса ≤ длины обучения (или RoPE-скейлинг).
- [ ] Pre-LN, warm-up + cosine, AdamW, клиппинг 1.0, label smoothing 0.1 для seq2seq.
- [ ] Для понимания текста — дообучать BERT-подобную модель (lr 2e-5…5e-5, 2–4 эпохи); для NER — выравнивание меток по `word_ids`.
- [ ] Секреты (токены HF, wandb) — не в коде.

---

## Что почитать / посмотреть

- 📘 [J. Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) · [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) ([версия 2018](http://nlp.seas.harvard.edu/2018/04/03/attention.html)) · [P. Bloem — Transformers from scratch](http://peterbloem.nl/blog/transformers) · [Kazemnejad — Positional encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- 📘 Лена Войта: [Seq2seq and Attention → Transformer](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#transformer_intro) · [Transfer Learning](https://lena-voita.github.io/nlp_course/transfer_learning.html) · [yandexdataschool/nlp_course](https://github.com/yandexdataschool/nlp_course) (week03–04)
- 📘 Учебник ML Яндекса: [Трансформеры](https://education.yandex.ru/handbook/ml/article/transformery) (то же в треке [contest.yandex.ru](https://contest.yandex.ru/tracks/ml/deep-learning-architectures/transformers))
- 📄 [DL2 HSE, неделя 3: The Evolution of Transformers](https://github.com/thecrazymage/DL2_HSE/tree/main/week_03) · Stanford [CS224n](https://web.stanford.edu/class/cs224n/) (лекции Transformers, Pretraining; [плейлист](https://www.youtube.com/watch?v=LWMzyfvuehA&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D&index=9)) · [CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/)
- 📄 🤗: [курс Hugging Face, гл. 7 — Token classification](https://huggingface.co/learn/nlp-course/chapter7/2) · [exBERT](https://huggingface.co/exbert/) · [расписания lr](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) · [Google AI Blog — Transformer](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- 📄 Статьи: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [BERT](https://arxiv.org/abs/1810.04805) · [RoBERTa](https://arxiv.org/abs/1907.11692) · [ALBERT](https://arxiv.org/abs/1909.11942) · [DistilBERT](https://arxiv.org/abs/1910.01108) · [ELECTRA](https://arxiv.org/abs/2003.10555) · [T5](https://arxiv.org/abs/1910.10683) · [Training Tips for the Transformer (Popel & Bojar)](https://arxiv.org/abs/1804.00247) · [RAdam / warm-up (Liu et al.)](https://arxiv.org/abs/1908.03265) · [On Layer Normalization in the Transformer (pre-LN)](https://arxiv.org/abs/2002.04745) · [Label smoothing (Szegedy et al.)](https://arxiv.org/abs/1512.00567) · [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) · [Что выучивают позиционные эмбеддинги](https://arxiv.org/abs/2010.04903) · [FFN как память ключ–значение](https://arxiv.org/abs/2012.14913) · [FlashAttention](https://arxiv.org/abs/2205.14135)

## Источники курса

- `Лекция 9.pptx` (DEEP LEARNING, день 9, 63 слайда): attention / self-attention в матричной записи, код attention и MHA, архитектура (encoder, self-attention, multi-head, positional encoding, residual + LayerNorm, FFN, decoder, виды внимания, маски), особенности обучения (warm-up), BERT (MLM, NSP, вход, обучение, дообучение, эффекты, представления слов), итоги.
- Семинар 9 «Transformer, Attention, Hugging Face» ([Colab](https://colab.research.google.com/drive/1kJBn03m4qCQH-Atgp-QoLOKoopMqVQM_)): функция внимания, экосистема 🤗, NER на WikiANN-ru с ruBERT, `Trainer`, `seqeval`, `pipeline`.
- Семинар 10 «Transformer → PEFT» ([Colab](https://colab.research.google.com/drive/1yQ4b8tCat38RpLbeQIIIMqYpz4QIpoRE)), первая часть: трансформер с нуля, Noam, label smoothing, copy-task, декодирование, визуализация внимания.
- `Повторение.pptx` — обзорные слайды по трансформеру и BERT.


<!-- materials:start -->
**Файлы курса в репозитории:**

- DL, лекция 9: трансформер и BERT (лекция курса, файл в репозиторий не включён)
- DL: повторение (обзор курса) (лекция курса, файл в репозиторий не включён)
- [Семинар 9: трансформер, Hugging Face, NER](../../materials/seminars/23_Семинар_9_Transformer_HF_NER.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1kJBn03m4qCQH-Atgp-QoLOKoopMqVQM_)
- [Семинар 10: от трансформера к PEFT и SFT](../../materials/seminars/23-25_Семинар_10_Transformer_PEFT_SFT.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1yQ4b8tCat38RpLbeQIIIMqYpz4QIpoRE)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [24. Языковые модели и GPT →](../24_language_models_gpt/README.md)
