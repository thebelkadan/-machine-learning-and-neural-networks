# 23. Трансформер и BERT

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Трансформер (Vaswani et al., 2017) — seq2seq **без рекуррентности**: стопка блоков «**multi-head self-attention** → FFN»,
каждый обёрнут в **residual + LayerNorm**. Self-attention: каждый токен строит query, key, value и собирает контекст со всех
позиций сразу — путь между любыми словами длины 1, обучение параллельно по позициям. Порядок слов вносит **позиционное
кодирование** (без него модель инвариантна к перестановкам). Декодер использует **маску будущего** и **cross-attention** на энкодер.
Три семейства: **encoder-only** (BERT — понимание текста), **decoder-only** (GPT — генерация, тема 24), **encoder–decoder** (T5, перевод).
BERT предобучают на **MLM** (угадай 15% замаскированных токенов) и NSP, потом дообучают под задачу: так в NLP пришёл transfer learning.

| Термин | Определение |
|---|---|
| Self-attention | $Q=XW_Q,\ K=XW_K,\ V=XW_V$ из одной последовательности; $\operatorname{softmax}(QK^{\top}/\sqrt{d_k})V$ |
| Multi-head | $h$ голов в подпространствах $d_k=d/h$, конкатенация и $W_O$; головы смотрят на разные «аспекты» |
| Маски | padding mask (не смотреть на `<pad>`) и causal mask (не смотреть в будущее, в декодере) |
| Cross-attention | Q — из декодера, K и V — из выхода энкодера |
| Позиционное кодирование | синусоиды разных частот (оригинал) или обучаемые векторы (BERT); RoPE в современных LLM |
| FFN | $\max(0,xW_1+b_1)W_2+b_2$ (или GELU), $d\to4d\to d$, применяется к каждому токену отдельно |
| Pre-LN / Post-LN | $x+F(\mathrm{LN}(x))$ / $\mathrm{LN}(x+F(x))$; pre-LN стабильнее для глубоких сетей |
| Warm-up | линейный рост lr в начале обучения (Adam + LN без него расходятся) |
| MLM / NSP | маскированное LM: 15% токенов (80% `[MASK]`, 10% случайный, 10% как есть) / «B идёт после A?» |
| `[CLS]`, `[SEP]` | токен «для классификации» в начале / разделитель сегментов |

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}+M\Big)V,\qquad
\mathrm{PE}_{(p,2i)}=\sin\frac{p}{10000^{2i/d}},\ \ \mathrm{PE}_{(p,2i+1)}=\cos\frac{p}{10000^{2i/d}},\qquad
\mathrm{lr}=d^{-0.5}\min\big(s^{-0.5},\,s\cdot w^{-1.5}\big)
$$

## Конспект

Разделы раскрываются по клику.

**Часть A. Архитектура**

<details>
<summary><b>1. Идея: «parallelized attention»</b></summary>

Vaswani et al. (2017), «Attention Is All You Need»: оставить от seq2seq с вниманием ([тема 22](../22_seq2seq_attention/README.md))
только внимание. Вместо RNN — **self-attention**: представление каждого токена уточняется с учётом **всех** токенов сразу.
Следствия: обучение параллельно по позициям (нет последовательных шагов), путь между любыми двумя словами — один слой,
масштабируемость на огромные данные. Цена — $O(T^2)$ по длине.

Общая схема — снова «кодировщик–декодировщик», 6 + 6 одинаковых по архитектуре блоков **с разными весами**. На входе — эмбеддинги
токенов (после subword-токенизации, [тема 21](../21_word_embeddings_tokenization/README.md)) размерности $d_{\text{model}} = 512$; каждый
блок превращает векторы в векторы **той же размерности** — «представления следующего уровня».

</details>

<details>
<summary><b>2. Query, key, value — матричная запись</b></summary>

Для входа $X\in\mathbb R^{T\times d}$ три обучаемые матрицы дают

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\qquad
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)V .
$$

Строка $i$ матрицы весов — распределение внимания токена $i$ по всем токенам; выход — взвешенная сумма значений. В оригинале
$d_k = d_v = 64$. Пример из лекции: в «The animal didn't cross the street because **it** was too tired» представление «it»
на следующем слое должно впитать «animal»; к чему относится «making» в другой фразе — тоже видно по весам (визуализатор exBERT).
Деление на $\sqrt{d_k}$ — [тема 22, раздел «3. Зачем делить на корень из d»](../22_seq2seq_attention/README.md).

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

</details>

<details>
<summary><b>3. Multi-head attention</b></summary>

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

</details>

<details>
<summary><b>4. Позиционное кодирование</b></summary>

Self-attention **эквивариантен к перестановкам**: без информации о позиции «собака кусает человека» и «человек кусает собаку» дают
одинаковый набор векторов. Решение — прибавить к эмбеддингу вектор позиции. Требования: уникален для позиции, не зависит от длины
предложения, ограничен, позволяет выражать относительные сдвиги. Синусоиды:

$$
\mathrm{PE}_{(p, 2i)} = \sin\frac{p}{10000^{2i/d}},\qquad \mathrm{PE}_{(p, 2i+1)} = \cos\frac{p}{10000^{2i/d}} .
$$

Длины волн — геометрическая прогрессия от $2\pi$ до $10000\cdot2\pi$: младшие координаты меняются быстро, старшие — медленно (как
разряды двоичного счётчика, но гладко). Сдвиг на $k$ — **линейное** преобразование (поворот в каждой паре координат, см. раздел «2. Синусоидальные позиции: сдвиг = поворот»).
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

</details>

<details>
<summary><b>5. Residual + LayerNorm, FFN</b></summary>

Каждый подблок (внимание, FFN) обёрнут в **прокидывание связи** и **LayerNorm**. Оригинал — **post-LN**: $\mathrm{LN}(x + \mathrm{Dropout}(\mathrm{Sublayer}(x)))$.
«Annotated Transformer» и большинство современных моделей — **pre-LN**: $x + \mathrm{Dropout}(\mathrm{Sublayer}(\mathrm{LN}(x)))$ — шорткат
остаётся «чистым», глубокие модели учатся стабильнее, меньше нужен warm-up (см. раздел «4. Pre-LN vs Post-LN»). LayerNorm нормирует по
признакам каждого токена (не по батчу) — подходит для последовательностей разной длины; в LLM часто RMSNorm (без центрирования).

**FFN** — двухслойный перцептрон, **одинаковый для всех позиций** (разделение параметров), «inverted bottleneck» $d\to4d\to d$:

$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)\,W_2 + b_2\quad(\text{ReLU; в BERT/GPT — GELU, в LLaMA — SwiGLU}).
$$

Внимание смешивает информацию **между** токенами, FFN преобразует каждый токен **отдельно** — в нём сосредоточено ~2/3 параметров
и, по ряду работ, «знания» модели (FFN как память ключ–значение).

</details>

<details>
<summary><b>6. Декодер: маски и cross-attention</b></summary>

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

</details>

<details>
<summary><b>7. Итог по архитектуре (из лекции)</b></summary>

- Трансформер — стекинг трансформер-блоков, послойное уточнение представлений токенов.
- Self-attention — граф + пулинг по релевантности.
- Размерность представлений токенов не меняется по слоям (не путать с внутренними $d_k, d_v$ и размерностью входа).
- Нерекуррентная модель для последовательностей; глубокая, с блоками внимания; изначально для машинного перевода.
- Последний слой — softmax + cross-entropy.

Готовые блоки в PyTorch: `nn.Transformer`, `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer` (`norm_first=True` для pre-LN,
`batch_first=True`), `F.scaled_dot_product_attention` (сам выбирает FlashAttention / memory-efficient ядро).

</details>

**Часть B. Обучение трансформера**

<details>
<summary><b>8. Рецепт оригинала</b></summary>

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

</details>

<details>
<summary><b>9. Warm-up: зачем</b></summary>

Лекция (по Liu et al., 2019, RAdam): в Adam оценка второго момента по первым шагам ненадёжна — дисперсия адаптивного lr огромна,
bias correction её увеличивает; плюс LayerNorm (особенно post-LN) даёт большие градиенты у последних слоёв. Без прогрева первые
шаги «выбрасывают» модель в плохую область. Сейчас обычно: линейный warm-up 1–5% шагов, затем **cosine** или линейное убывание
(`get_cosine_schedule_with_warmup` в Hugging Face Transformers). Советы Popel & Bojar (2018): батч побольше (насколько позволяет память), lr
повыше, но в пределах устойчивости; качество продолжает расти долго — не останавливать рано; усреднение чекпойнтов.

</details>

<details>
<summary><b>10. Label smoothing</b></summary>

Вместо one-hot цели — $q = (1 - \epsilon)\,\text{one-hot} + \epsilon/(K - 1)$ на остальные классы (в семинаре — на $K - 2$, исключая
`<pad>`), лосс — KL-дивергенция (`nn.KLDivLoss`). Модель перестаёт быть «самоуверенной»: перплексия хуже, но accuracy и BLEU лучше;
калибровка лучше. Подробно — см. раздел «5. Label smoothing».

</details>

**Часть C. Три семейства и BERT**

<details>
<summary><b>11. Encoder-only, decoder-only, encoder–decoder</b></summary>

| Семейство | Внимание | Предобучение | Задачи | Модели |
|---|---|---|---|---|
| encoder-only | двунаправленное | MLM (+NSP) | классификация, NER, поиск, эмбеддинги | BERT, RoBERTa, DeBERTa, ruBERT |
| decoder-only | causal | следующий токен (LM) | генерация, всё через промпт | GPT, LLaMA, Qwen, YandexGPT ([тема 24](../24_language_models_gpt/README.md)) |
| encoder–decoder | энкодер полное, декодер causal + cross | denoising (span corruption) | перевод, суммаризация | оригинал, T5, BART, mT5 |

</details>

<details>
<summary><b>12. BERT (Devlin et al., 2018)</b></summary>

**Bidirectional Encoder Representations from Transformers.** Идея из GPT — предобучить языковую модель на большом корпусе и
дообучать под задачи (transfer learning), — но с **двусторонним** контекстом. Отличия от ELMo ([тема 21](../21_word_embeddings_tokenization/README.md)):
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

</details>

<details>
<summary><b>13. Дообучение (fine-tuning)</b></summary>

- **Классификация предложения/пары:** финальное состояние `[CLS]` × обучаемая матрица → softmax.
- **Разметка токенов (NER, POS):** линейный слой на каждый токен.
- **Extractive QA (SQuAD):** два обучаемых вектора $S$, $E$; $p_{\text{start}}(i) = \operatorname{softmax}_i(S^{\top}h_i)$, аналогично конец; ответ — отрезок с
  максимальным $S^{\top}h_i + E^{\top}h_j$, $j\ge i$.
- Гиперпараметры дообучения из статьи: lr 2e-5 – 5e-5, батч 16–32, 2–4 эпохи.

Эффекты (лекция): для одних задач важнее MLM, для других — NSP; модели «слева направо» плохи на SQuAD; **размер имеет значение** даже
на небольших датасетах, и кривая не вышла на асимптоту. Задачи GLUE: CoLA (грамматичность), QQP (парафразы), MNLI (логический вывод),
SST-2 (тональность) и др.

**Признаки из BERT без дообучения** — как в ELMo, комбинировать слои: для NER конкатенация последних четырёх слоёв почти не уступает
дообучению. Вектор слова зависит от контекста: в примере кода этой темы «bank» в финансовых предложениях — косинус 0.90, в «речных» — 0.84,
между группами — 0.63.

</details>

<details>
<summary><b>14. Семинар 9: NER с Hugging Face</b></summary>

Экосистема Hugging Face: **Transformers** (модели), **Datasets** (`load_dataset`), **Tokenizers** (быстрые токенизаторы), **Hub** (модели и датасеты),
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
- Важно: В семинаре токен Hugging Face записан прямо в коде (`login(token="hf_...")`) — так делать нельзя: используйте `notebook_login()` или
  переменную окружения / секреты Colab, а утёкший токен отзовите.

</details>

<details>
<summary><b>15. Семинар 10: трансформер с нуля</b></summary>

По «The Annotated Transformer» (Harvard NLP): `EncoderDecoder`, `Encoder`/`Decoder` из $N = 6$ клонов слоя, `SublayerConnection`
(pre-LN + residual + dropout), `MultiHeadedAttention`, `PositionwiseFeedForward`, `Embeddings` (× $\sqrt d$), `PositionalEncoding`,
`Generator` (linear + log_softmax), `make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)` с
инициализацией Xavier. Обучение: класс `Batch` (маски, `ntokens` для нормировки лосса), `run_epoch` с аккумуляцией градиентов,
расписание Noam, `LabelSmoothing` через KL, **copy-task** (выучить копирование случайных последовательностей) как проверка, что всё
работает; жадное декодирование, beam search, визуализация матриц внимания по слоям и головам. Вторая часть семинара (pretrained vs
instruct, chat template, LoRA, PEFT) — [тема 25](../25_llm_finetuning_peft/README.md).

</details>

<details>
<summary><b>16. Тренды (итоги лекции)</b></summary>

Почти все лучшие решения основаны на трансформере; обучение на огромных корпусах; подстройка под задачи; «упрощение» — меньше
параметров и слоёв (DistilBERT, ALBERT, TinyBERT). Центральная идея современного NLP — **transfer learning**: предобучаемся на задаче
с большими данными и дешёвой разметкой, переносим модель и дообучаем.

</details>

<details>
<summary><b>Чек-лист</b></summary>

- [] Маски: padding во всех вниманиях, causal в декодере; формы масок согласованы с `[B, h, Tq, Tk]`.
- [] Позиции: синусоиды/обучаемые/RoPE; длина инференса ≤ длины обучения (или RoPE-скейлинг).
- [] Pre-LN, warm-up + cosine, AdamW, клиппинг 1.0, label smoothing 0.1 для seq2seq.
- [] Для понимания текста — дообучать BERT-подобную модель (lr 2e-5…5e-5, 2–4 эпохи); для NER — выравнивание меток по `word_ids`.
- [] Секреты (токены HF, wandb) — не в коде.

</details>

**Математика и выводы**

<details>
<summary><b>1. Self-attention эквивариантен к перестановкам</b></summary>

Пусть $P$ — матрица перестановки строк ($P^{\top}P = I$). Для входа $PX$: $Q' = PQ$, $K' = PK$, $V' = PV$, и

$$
\operatorname{softmax}\Big(\frac{PQK^{\top}P^{\top}}{\sqrt{d_k}}\Big)PV = P\,\operatorname{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)P^{\top}P\,V = P\cdot\operatorname{Attention}(Q, K, V),
$$

так как построчный softmax коммутирует с перестановкой строк и столбцов. FFN, LayerNorm и residual действуют на каждый токен отдельно —
тоже эквивариантны. Значит, весь энкодер без позиций — функция **множества** токенов: переставили вход — так же переставился выход.
В примере кода этой темы задача «сколько раз встречается токен» (не зависит от порядка) решается без PE (0.993), а «какой токен слева» — нет (0.239).

</details>

<details>
<summary><b>2. Синусоидальные позиции: сдвиг = поворот</b></summary>

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

</details>

<details>
<summary><b>3. Параметры и вычисления</b></summary>

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

</details>

<details>
<summary><b>4. Pre-LN vs Post-LN</b></summary>

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

</details>

<details>
<summary><b>5. Label smoothing</b></summary>

Цель $q_k = (1 - \epsilon)[k = y] + \frac{\epsilon}{K - 1}[k\ne y]$ (вариант с $\frac{\epsilon}{K}$ на все классы тоже распространён). Лосс

$$
\mathrm{CE}(q, p) = -\sum_kq_k\log p_k = (1 - \epsilon)\,\big(-\log p_y\big) + \frac{\epsilon}{K - 1}\sum_{k\ne y}\big(-\log p_k\big).
$$

Минимум по $p$ — при $p = q$ (неравенство Гиббса), то есть оптимальные логиты **конечны**: $z_y - z_k = \ln\frac{(1 - \epsilon)(K - 1)}{\epsilon}$
(для $\epsilon = 0.1$, $K = 37000$ — около 12.7). Без сглаживания оптимум $z_y - z_k\to\infty$: модель бесконечно наращивает уверенность,
веса растут, калибровка портится. Минимальное значение $\mathrm{CE} = H(q) > 0$, поэтому удобнее KL: $\mathrm{KL}(q\,\|\,p) = \mathrm{CE}(q, p) - H(q)$
равна 0 в оптимуме (как в `nn.KLDivLoss` семинара). Перплексия на one-hot целях ухудшается (модель намеренно оставляет массу другим
токенам), но BLEU и accuracy растут. В PyTorch: `nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=pad)`.

</details>

<details>
<summary><b>6. Расписание Noam и зачем прогрев</b></summary>

$\mathrm{lr}(s) = d^{-0.5}\min(s^{-0.5}, s\,w^{-1.5})$: две ветви равны при $s = w$, пик $\mathrm{lr}_{\max} = d^{-0.5}w^{-0.5}$ (для $d = 512$,
$w = 4000$ — $\approx7\cdot10^{-4}$); до пика рост линейный, после — убывание $\propto1/\sqrt s$. Множитель $d^{-0.5}$ уменьшает lr у широких моделей.

**Почему Adam нужен прогрев** (Liu et al., 2019). Адаптивный шаг $\propto1/\sqrt{\hat v_t}$, где $\hat v_t$ — экспоненциальное среднее $g^2$.
На первых шагах оно оценено по считанным градиентам: если приблизить $g\sim\mathcal N(0, \sigma^2)$, то $1/\sqrt{\hat v_t}$ имеет огромную
(при $t\to1$ — неограниченную) дисперсию. Отдельные координаты получают гигантские шаги. Прогрев гасит это, пока статистики не
накопятся; RAdam делает то же аналитически — выключает адаптивность, пока оценка дисперсии ненадёжна, и затем вводит поправку
$r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}{(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}}$, $\rho_\infty = \frac{2}{1 - \beta_2} - 1$.

</details>

<details>
<summary><b>7. MLM как задача и её цена</b></summary>

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

</details>

<details>
<summary><b>8. BERT для QA: span-предсказание</b></summary>

Выходы $h_i\in\mathbb R^{d}$ для токенов абзаца, два обучаемых вектора $S, E$:

$$
p^{\text{start}}_i = \frac{e^{S^{\top}h_i}}{\sum_je^{S^{\top}h_j}},\qquad p^{\text{end}}_i = \frac{e^{E^{\top}h_i}}{\sum_je^{E^{\top}h_j}},\qquad
\mathcal L = -\log p^{\text{start}}_{s^{*}} - \log p^{\text{end}}_{e^{*}} .
$$

На выводе ищется $\arg\max_{i\le j\le i+L_{\max}}\big(S^{\top}h_i + E^{\top}h_j\big)$ за $O(T\cdot L_{\max})$; для вопросов без ответа (SQuAD 2.0)
сравнивают со скором $S^{\top}h_{\mathrm{CLS}} + E^{\top}h_{\mathrm{CLS}}$ и порогом.

</details>

<details>
<summary><b>9. NER: метрика по сущностям</b></summary>

seqeval собирает из BIO-тегов **спаны** (тип, начало, конец) и считает

$$
P = \frac{\lvert\text{предсказанные}\cap\text{истинные}\rvert}{\lvert\text{предсказанные}\rvert},\quad
R = \frac{\lvert\text{предсказанные}\cap\text{истинные}\rvert}{\lvert\text{истинные}\rvert},\quad F_1 = \frac{2PR}{P + R},
$$

где совпадение — точное (тип и обе границы). Пример семинара с логитами $[[7.12, -0.81], [0.23, 8.1], [5.3, 0.5]]$ и целью $[0, 1, 1]$:
предсказание $[0, 1, 0]$ → token accuracy $2/3$; сущностей предсказано 1 и она верна ($P = 1$), истинных 2 ($R = 0.5$), $F_1 = 2/3$.

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Сложность self-attention по длине?</summary>

$O(T^2d)$ времени и $O(T^2)$ памяти на матрицу весов — главное ограничение на длину контекста (FlashAttention экономит память, но не меняет $O(T^2)$ вычислений).

</details>

<details>
<summary>Почему без позиционного кодирования трансформер — «мешок слов»?</summary>

self-attention эквивариантен к перестановкам: переставили входы — так же переставились выходы (в примере: задача «левый сосед» 0.24 без PE против 0.80 с PE).

</details>

<details>
<summary>Зачем несколько голов, если можно одну большую?</summary>

разные подпространства и разные распределения внимания (синтаксис, кореференция, соседние токены) при той же стоимости.

</details>

<details>
<summary>Зачем warm-up?</summary>

в начале статистики Adam (второй момент) неточны, а post-LN даёт большие градиенты у выходных слоёв — без warm-up обучение расходится.

</details>

<details>
<summary>Почему BERT не может просто учиться «слева-направо + справа-налево» в одной модели?</summary>

при полном внимании слово «видит себя» через другие слои; отсюда MLM.

</details>

<details>
<summary>Почему 80/10/10 при маскировании?</summary>

на дообучении `[MASK]` не встречается; случайные и неизменённые токены заставляют модель строить представление **каждого** токена.

</details>

<details>
<summary>Что брать как вектор предложения из BERT?</summary>

`[CLS]` после дообучения; без него — mean-pooling (сырой `[CLS]` предобученного BERT — плохой эмбеддинг; для поиска — Sentence-BERT).

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- **Оригинал:** 6 + 6 слоёв, $d=512$, 8 голов по 64, FFN 2048, dropout 0.1, label smoothing 0.1, Adam $(0.9, 0.98, 10^{-9})$, warm-up 4000 шагов; WMT14 en-de 4.5 млн пар, BPE 37k; 8×P100, base — 100k шагов (12 ч), big — 300k (3.5 дня).
- **BERT:** Wikipedia (2.5 млрд слов) + BookCorpus (800 млн); WordPiece 30k; base 12 слоёв / 768 / 12 голов (110M), large 24 / 1024 / 16 (340M); 1M шагов, батч 256×512, AdamW 1e-4; вход = токен + сегмент + позиция (обучаемая), до 512 токенов.
- **Дообучение:** классификация — линейный слой на `[CLS]`; разметка токенов (NER) — слой на каждый токен; QA (SQuAD) — два вектора «начало/конец ответа», скалярное произведение с каждым токеном.
- **Наследники:** RoBERTa (без NSP, больше данных, динамическое маскирование), ALBERT (общие веса слоёв), DistilBERT (дистилляция, 6 слоёв), ELECTRA (детектор подменённых токенов), DeBERTa; для русского — ruBERT (DeepPavlov), ai-forever/ruBERT, rubert-tiny2.
- **Семинар 9:** NER на WikiANN-ru (O, B/I-PER, ORG, LOC) с `DeepPavlov/rubert-base-cased`, `DataCollatorForTokenClassification`, `seqeval`, `Trainer` (lr 1e-4, 2 эпохи, cosine). **Семинар 10:** трансформер с нуля по «The Annotated Transformer» (copy-task, label smoothing через KL, Noam-расписание lr).

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# Блок трансформера с нуля: multi-head attention, маска, pre-LN, FFN, синусоидальные позиции
import math, torch, torch.nn as nn, torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h, self.dk = h, d // h
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
    def forward(self, x, mask=None):                                        # mask: [T, T], True = можно смотреть
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)   # 3 × [B, h, T, dk]
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if mask is not None: s = s.masked_fill(~mask, float("-inf"))
        a = s.softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, T, d)), a           # головы склеиваем и проецируем

class Block(nn.Module):                                                     # pre-LN: x + MHA(LN x), x + FFN(LN x)
    def __init__(self, d, h):
        super().__init__(); self.ln1, self.ln2, self.attn = nn.LayerNorm(d), nn.LayerNorm(d), MHA(d, h)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)[0]
        return x + self.ffn(self.ln2(x))

def sinusoidal(T, d):
    pos, i = torch.arange(T)[:, None], torch.arange(0, d, 2)
    pe = torch.zeros(T, d); pe[:, 0::2] = torch.sin(pos / 10000 ** (i / d)); pe[:, 1::2] = torch.cos(pos / 10000 ** (i / d))
    return pe

torch.manual_seed(0)
m, x = MHA(32, 4), torch.randn(2, 5, 32)
causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))                     # декодер: только прошлое и текущее
out, a = m(x, causal)
q, k, v = m.qkv(x).view(2, 5, 3, 4, 8).permute(2, 0, 3, 1, 4)
ref = m.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 5, 32))
print("своя MHA == F.scaled_dot_product_attention:", torch.allclose(out, ref, atol=1e-6))
print("веса внимания головы 0 (causal):\n", a[0, 0].detach().numpy().round(2))
print("параметров в блоке d=512, h=8:", sum(p.numel() for p in Block(512, 8).parameters()), "≈ 12·d²")
```

Результат запуска:

```text
своя MHA == F.scaled_dot_product_attention: True
веса внимания головы 0 (causal):
 [[1.   0.   0.   0.   0.  ]
 [0.66 0.34 0.   0.   0.  ]
 [0.14 0.4  0.46 0.   0.  ]
 [0.4  0.23 0.16 0.21 0.  ]
 [0.13 0.23 0.18 0.24 0.22]]
параметров в блоке d=512, h=8: 3152384 ≈ 12·d²
```

```python
# Зачем позиционное кодирование: self-attention без PE «не видит» порядка
class Encoder(nn.Module):
    def __init__(self, V, n_out, d=64, h=4, L=2, use_pe=True, T=16):
        super().__init__(); self.emb, self.use_pe = nn.Embedding(V, d), use_pe
        self.register_buffer("pe", sinusoidal(T, d)); self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, n_out)
    def forward(self, x):
        z = self.emb(x) * math.sqrt(self.emb.embedding_dim) + (self.pe[: x.shape[1]] if self.use_pe else 0)
        for b in self.blocks: z = b(z)
        return self.head(self.ln(z))

T = 12
tasks = {"сколько ещё раз встречается токен": lambda x: (x[:, :, None] == x[:, None, :]).sum(-1) - 1,     # порядок не важен
         "какой токен стоит слева":           lambda x: torch.cat([torch.full_like(x[:, :1], 10), x[:, :-1]], 1)}  # важен
for tname, f in tasks.items():
    for use_pe in (False, True):
        torch.manual_seed(0); model = Encoder(10, 11, use_pe=use_pe); opt = torch.optim.AdamW(model.parameters(), 1e-3)
        for step in range(600):
            x = torch.randint(0, 10, (128, T))
            loss = F.cross_entropy(model(x).reshape(-1, 11), f(x).reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        x = torch.randint(0, 10, (1000, T))
        with torch.no_grad(): acc = (model(x).argmax(-1) == f(x)).float().mean().item()
        print(f"{tname:34s} PE={'да ' if use_pe else 'нет'}: accuracy {acc:.3f}")
```

Результат запуска:

```text
сколько ещё раз встречается токен  PE=нет: accuracy 0.993
сколько ещё раз встречается токен  PE=да : accuracy 0.995
какой токен стоит слева            PE=нет: accuracy 0.239
какой токен стоит слева            PE=да : accuracy 0.801
```

```python
# BERT из коробки: маскированное LM и контекстные эмбеддинги (DistilBERT, ~270 МБ при первом запуске)
import warnings; warnings.filterwarnings("ignore")
try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    name = "distilbert-base-uncased"
    tok, mlm = AutoTokenizer.from_pretrained(name), AutoModelForMaskedLM.from_pretrained(name).eval()
    for s in ["The capital of France is [MASK].", "The cat is drinking [MASK].", "She sat on the [MASK] of the river."]:
        enc = tok(s, return_tensors="pt")
        with torch.no_grad(): p = mlm(**enc).logits[0, (enc.input_ids[0] == tok.mask_token_id).nonzero()[0, 0]].softmax(-1)
        top = p.topk(5); print(f"{s:38s}", [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])

    sents = ["I opened a savings account at the bank.", "The bank approved my loan.",
             "We had a picnic on the bank of the river.", "The boat drifted toward the muddy bank."]
    def vec(sent, word="bank"):                                             # вектор слова = скрытое состояние его токена
        enc = tok(sent, return_tensors="pt")
        with torch.no_grad(): h = mlm.base_model(**enc).last_hidden_state[0]
        return h[enc.input_ids[0].tolist().index(tok.convert_tokens_to_ids(word))]
    V_ = torch.stack([vec(s) for s in sents]); C = F.cosine_similarity(V_[:, None], V_[None], dim=-1)
    print("косинусы «bank»: деньги-деньги %.2f | река-река %.2f | деньги-река %.2f" % (C[0, 1], C[2, 3], C[:2, 2:].mean()))
except Exception as e:
    print("не удалось загрузить с Hugging Face Hub (нужен интернет):", type(e).__name__, str(e)[:120])
```

Результат запуска:

```text
The capital of France is [MASK].       [('marseille', 0.14), ('nantes', 0.09), ('toulouse', 0.09), ('paris', 0.09), ('lyon', 0.08)]
The cat is drinking [MASK].            [('water', 0.13), ('tea', 0.11), ('beer', 0.1), ('milk', 0.05), ('coffee', 0.04)]
She sat on the [MASK] of the river.    [('bank', 0.27), ('edge', 0.25), ('banks', 0.21), ('side', 0.08), ('shore', 0.02)]
косинусы «bank»: деньги-деньги 0.90 | река-река 0.84 | деньги-река 0.63
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- [J. Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) · [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) ([версия 2018](http://nlp.seas.harvard.edu/2018/04/03/attention.html)) · [P. Bloem — Transformers from scratch](http://peterbloem.nl/blog/transformers) · [Kazemnejad — Positional encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- Лена Войта: [Seq2seq and Attention → Transformer](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#transformer_intro) · [Transfer Learning](https://lena-voita.github.io/nlp_course/transfer_learning.html) · [yandexdataschool/nlp_course](https://github.com/yandexdataschool/nlp_course) (week03–04)
- Учебник ML Яндекса: [Трансформеры](https://education.yandex.ru/handbook/ml/article/transformery) (то же в треке [contest.yandex.ru](https://contest.yandex.ru/tracks/ml/deep-learning-architectures/transformers))
- [DL2 HSE, неделя 3: The Evolution of Transformers](https://github.com/thecrazymage/DL2_HSE/tree/main/week_03) · Stanford [CS224n](https://web.stanford.edu/class/cs224n/) (лекции Transformers, Pretraining; [плейлист](https://www.youtube.com/watch?v=LWMzyfvuehA&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D&index=9)) · [CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/)
- Hugging Face: [курс Hugging Face, гл. 7 — Token classification](https://huggingface.co/learn/nlp-course/chapter7/2) · [exBERT](https://huggingface.co/exbert/) · [расписания lr](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) · [Google AI Blog — Transformer](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- Статьи: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [BERT](https://arxiv.org/abs/1810.04805) · [RoBERTa](https://arxiv.org/abs/1907.11692) · [ALBERT](https://arxiv.org/abs/1909.11942) · [DistilBERT](https://arxiv.org/abs/1910.01108) · [ELECTRA](https://arxiv.org/abs/2003.10555) · [T5](https://arxiv.org/abs/1910.10683) · [Training Tips for the Transformer (Popel & Bojar)](https://arxiv.org/abs/1804.00247) · [RAdam / warm-up (Liu et al.)](https://arxiv.org/abs/1908.03265) · [On Layer Normalization in the Transformer (pre-LN)](https://arxiv.org/abs/2002.04745) · [Label smoothing (Szegedy et al.)](https://arxiv.org/abs/1512.00567) · [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) · [Что выучивают позиционные эмбеддинги](https://arxiv.org/abs/2010.04903) · [FFN как память ключ–значение](https://arxiv.org/abs/2012.14913) · [FlashAttention](https://arxiv.org/abs/2205.14135)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Лекция 9.pptx` (DEEP LEARNING, день 9, 63 слайда): attention / self-attention в матричной записи, код attention и MHA, архитектура (encoder, self-attention, multi-head, positional encoding, residual + LayerNorm, FFN, decoder, виды внимания, маски), особенности обучения (warm-up), BERT (MLM, NSP, вход, обучение, дообучение, эффекты, представления слов), итоги.
- Семинар 9 «Transformer, Attention, Hugging Face» ([Colab](https://colab.research.google.com/drive/1kJBn03m4qCQH-Atgp-QoLOKoopMqVQM_)): функция внимания, экосистема Hugging Face, NER на WikiANN-ru с ruBERT, `Trainer`, `seqeval`, `pipeline`.
- Семинар 10 «Transformer → PEFT» ([Colab](https://colab.research.google.com/drive/1yQ4b8tCat38RpLbeQIIIMqYpz4QIpoRE)), первая часть: трансформер с нуля, Noam, label smoothing, copy-task, декодирование, визуализация внимания.
- `Повторение.pptx` — обзорные слайды по трансформеру и BERT.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017 · Rush. [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/).
- Xiong et al. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), 2020 · Zhang, Sennrich. [RMSNorm](https://arxiv.org/abs/1910.07467), 2019.
- Liu et al. [On the Variance of the Adaptive Learning Rate and Beyond (RAdam)](https://arxiv.org/abs/1908.03265), 2019.
- Su et al. [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021 · Press et al. [ALiBi](https://arxiv.org/abs/2108.12409), 2021.
- Bhojanapalli et al. [Low-Rank Bottleneck in Multi-head Attention](https://arxiv.org/abs/2002.07028), 2020 · Dao et al. [FlashAttention](https://arxiv.org/abs/2205.14135), 2022 · Kaplan et al. [Scaling Laws](https://arxiv.org/abs/2001.08361) (оценка $6N$ FLOPs/токен), 2020.
- Müller, Kornblith, Hinton. [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629), 2019.
- Devlin et al. [BERT](https://arxiv.org/abs/1810.04805), 2018 · Clark et al. [ELECTRA](https://arxiv.org/abs/2003.10555), 2020.

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL, лекция 9: трансформер и BERT · DL: повторение (обзор курса)
- **Семинары:** [Семинар 9: трансформер, Hugging Face, NER](../../materials/seminars/23_Семинар_9_Transformer_HF_NER.ipynb) ([Colab](https://colab.research.google.com/drive/1kJBn03m4qCQH-Atgp-QoLOKoopMqVQM_)) · [Семинар 10: от трансформера к PEFT и SFT](../../materials/seminars/23-25_Семинар_10_Transformer_PEFT_SFT.ipynb) ([Colab](https://colab.research.google.com/drive/1yQ4b8tCat38RpLbeQIIIMqYpz4QIpoRE))
- **Видео и внешние курсы:** [Яндекс: трансформеры](https://contest.yandex.ru/tracks/ml/deep-learning-architectures/transformers) · [CS224n, лекция из плейлиста (YouTube)](https://www.youtube.com/watch?v=LWMzyfvuehA&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D&index=9) · [Лена Войта — Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) · [DL2 HSE (thecrazymage/DL2_HSE)](https://github.com/thecrazymage/DL2_HSE) · [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/)

</details>

---

[← 22. Seq2seq и механизм внимания](../22_seq2seq_attention/README.md) · [Оглавление курса](../../README.md) · [24. Языковые модели, декодирование, GPT →](../24_language_models_gpt/README.md)
