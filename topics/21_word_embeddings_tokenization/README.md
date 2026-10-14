# 21. Представления текста: мешок слов, TF-IDF, word2vec, GloVe, fastText, BPE, контекстные эмбеддинги

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 20. RNN](../20_rnn/README.md) · [22. Seq2seq и внимание →](../22_seq2seq_attention/README.md)

---

## Часть A. Что такое «слово» для модели

### 1. Немного лингвистики (лекция Сириуса)

**Языковой знак** — единство **означающего** (форма: буквы, звуки) и **означаемого** (смысл), по Соссюру. В означаемом различают
**денотационный** смысл (прямое, словарное значение) и **прагматический** (зависит от говорящего, слушающего, ситуации), плюс
**коннотации** — сопутствующие оттенки. Словарями всё это не покрыть, поэтому смысл извлекают из употребления — из **корпусов**.

Задача векторизации: превратить текст во что-то числовое. Два пути:
1. вектор на **весь текст** (мешок слов, TF-IDF, sentence-эмбеддинги);
2. вектор на **каждый токен** → последовательность векторов для RNN/CNN/трансформера.

### 2. One-hot и хэширование

**One-hot:** словарь размера $\lvert V\rvert$, слову $i$ соответствует единичный вектор $e_i$. Проблемы: размерность = размер словаря
(десятки–сотни тысяч), **все слова одинаково далеки** друг от друга ($e_i^{\top}e_j = 0$), нет обобщения на незнакомые слова.

**Hashing trick:** индекс слова = `hash(word) % n` — фиксированная размерность без хранения словаря, ценой коллизий
(`HashingVectorizer` в sklearn, n-граммы в fastText).

### 3. Мешок слов и TF-IDF

**Bag-of-Words:** текст → вектор счётчиков слов (кроме стоп-слов). Порядок теряется: «Я не люблю ML» и «Я люблю не ML» неразличимы.
Частично спасают **n-граммы** (`ngram_range=(1, 2)`: «не понравился» — один признак).

**TF-IDF** взвешивает счётчик редкостью слова в коллекции:

$$
\text{tf}(w,d) = \frac{n_{wd}}{\sum_{w'}n_{w'd}},\qquad
\text{idf}(w) = \log\frac{N}{\mathrm{df}(w)},\qquad
\text{tf-idf}(w,d) = \text{tf}(w,d)\cdot\text{idf}(w),
$$

где $\mathrm{df}(w)$ — число документов со словом $w$. Слова «везде» (предлоги) получают вес ≈ 0, специфичные — большой.
В sklearn по умолчанию `smooth_idf=True`: $\text{idf} = \ln\frac{1+N}{1+\mathrm{df}} + 1$, и строки нормируются по L2.
`sublinear_tf=True` заменяет $n$ на $1+\ln n$ — полезно для длинных текстов.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True),
    LogisticRegression(C=5, max_iter=2000, class_weight="balanced"),
)
# analyzer="char_wb", ngram_range=(2, 5) — символьные n-граммы: устойчивы к опечаткам и морфологии (но медленнее)
```

**Ноутбук `simple_sentiment_analysis`** (2000 рецензий IMDB из `nltk.corpus.movie_reviews`, 5-fold accuracy) — опыты с бейзлайном:

| Вариант | Accuracy |
|---|---|
| CountVectorizer + LogisticRegression | 0.841 ± 0.017 |
| TfidfVectorizer + LogisticRegression | 0.821 ± 0.004 |
| `min_df=10` / `min_df=50` | 0.839 / 0.813 |
| LinearSVC / SGDClassifier | 0.833 / 0.755 ± 0.112 |
| стоп-слова NLTK / sklearn | 0.842 / 0.839 |
| **униграммы + биграммы** | **0.853** |
| символьные 3–5-граммы (`char_wb`) | 0.82 |

Выводы: биграммы помогают (отрицания), стоп-слова почти ничего не дают, слишком большой `min_df` выбрасывает полезные редкие
слова, SGD без настройки нестабилен. Самые весомые слова у LR — «bad», «worst», «waste», «awful», «boring», «great», «fun». Код
ноутбука на Python 2 (`print` без скобок, `sklearn.cross_validation`) — в современном sklearn это `sklearn.model_selection`.

**Kinopoisk (L1-пример, 36 591 рецензия, 3 класса, macro-F1 ≈ 0.66):** класс Neutral распознаётся плохо (F1 ≈ 0.34) — нейтральные
рецензии смешивают хвалу и критику. Найдена **утечка**: 68% рецензий содержат «N из 10», и до очистки топ-признаками были «из 10» и
«10 из»; в нейтральных встречается само слово «нейтральная». Удаление оценок снижает macro-F1 лишь на ≈ 0.01, но модель становится честной.

---

## Часть B. Дистрибутивная семантика

### 4. Гипотеза и матрица совстречаемости

> «You shall know a word by the company it keeps» — J. R. Firth, 1957. Значение слова определяется словами, которые часто стоят рядом.

Пример из лекции: «Полосатая маленькая \*\*\*\*\* мурлычит и пьёт молоко» — пропуск угадывается по контексту. Контекст — слова в окне
$\pm h$ вокруг. Матрица совстречаемости $X_{ij}$ — сколько раз слово $j$ встретилось в окне слова $i$. Строки уже отражают
семантику, но **разреженные** и размерности $\lvert V\rvert$. Сырые счётчики забиты частыми словами, поэтому их переводят в
**PMI**:

$$
\mathrm{PMI}(w,c) = \log\frac{p(w,c)}{p(w)\,p(c)},\qquad \mathrm{PPMI} = \max(0, \mathrm{PMI}) .
$$

### 5. SVD и LSA

**Сингулярное разложение:** любая матрица $A = U\Sigma V^{\top}$, $\sigma_1\ge\sigma_2\ge\ldots\ge0$. **Теорема Эккарта–Янга:** лучшее
по норме Фробениуса приближение ранга $r$ — усечённое SVD:

$$
\min_{\operatorname{rank}B = r}\lVert A - B\rVert_F = \lVert A - U_r\Sigma_rV_r^{\top}\rVert_F = \sqrt{\sigma_{r+1}^2 + \ldots + \sigma_{\min(n,m)}^2} .
$$

Пример лекции: матрица Гильберта-подобная $10\times10$ приближается рангом 2 с относительной ошибкой 2.6%. Вектор слова — строка
$U_r\Sigma_r^{1/2}$ (или $U_r$). На матрице «слово × документ» это **латентный семантический анализ (LSA)**; тот же приём —
коллаборативная фильтрация ([тема 13](../13_recommender_systems/README.md)), сжатие данных, PCA ([тема 11](../11_clustering_dim_reduction/README.md)).
Тематическое моделирование развивает идею вероятностно (PLSA, LDA; библиотека BigARTM, курс К. В. Воронцова).

**L1-пример** (PPMI + SVD-200 на 10.7 млн токенов рецензий, окно ±4) даёт соседей: «отличный» → великолепный, потрясающий, замечательный;
«режиссёр» → сценарист, Нолан, продюсер, Спилберг; «музыка» → саундтрек, мелодия, Циммер. И сразу видна слабость: «смешно» рядом со
«скучно» — **антонимы делят контексты**. Аналогия «мужчина : женщина = король : ?» → «королева» удалась, «он : она = актёр : ?» — нет.

### 6. word2vec (Mikolov et al., 2013)

Вместо подсчёта — **предсказание**. Трюк самообучения: учим модель решать вспомогательную задачу на неразмеченном корпусе, а
используем её **внутренние представления** (принцип трансферного обучения).

- **Skip-gram:** по центральному слову предсказать слова контекста. Лучше для редких слов и небольших корпусов (окно ~10).
- **CBOW** (Continuous Bag of Words): по сумме/среднему контекста предсказать центральное слово. Быстрее, для больших корпусов (окно ~5).

У каждого слова **два вектора**: $v_w$ (центральное) и $u_w$ (контекст). Функционал skip-gram и вероятность:

$$
J(\theta) = \frac1T\sum_{t=1}^{T}\sum_{-h\le j\le h,\,j\ne0}\log p(w_{t+j}\mid w_t),\qquad
p(o\mid c) = \frac{\exp(u_o^{\top}v_c)}{\sum_{w\in V}\exp(u_w^{\top}v_c)} .
$$

Градиент (вывод — [deep.md §2](deep.md#2-skip-gram-градиент-полного-softmax)):
$\partial\log p(o\mid c)/\partial v_c = u_o - \sum_w p(w\mid c)\,u_w$ — «тянем к настоящему контексту, отталкиваем от ожидаемого».
Знаменатель — сумма по всему словарю на каждую пару, это дорого. Решения:

- **Negative sampling (SGNS):** бинарная классификация «реальная пара или случайная»:
  $J_t = \log\sigma(u_o^{\top}v_c) + \sum_{i=1}^{k}\mathbb E_{n_i\sim P_n}\log\sigma(-u_{n_i}^{\top}v_c)$, $k = 5$–$20$, $P_n(w)\propto \mathrm{count}(w)^{3/4}$.
- **Hierarchical softmax:** словарь — дерево Хаффмана, $p(w\mid c)$ — произведение $\log_2\lvert V\rvert$ бинарных решений.
- **Subsampling частых слов:** слово выкидывается с вероятностью $1-\sqrt{t/f(w)}$ ($t\approx10^{-5}$) — «the» почти ничего не говорит о соседях.
- **Фразы:** частые словосочетания склеиваются в один токен (`New_York`, `machine_learning`).

Реализация в коде отличается от статьи — тонкости есть у McCormick и Lilian Weng. На выходе берут $v_w$ (или $v_w + u_w$).

```python
from gensim.models import Word2Vec          # pip install gensim
w2v = Word2Vec(sentences=docs, vector_size=200, window=5, min_count=5, sg=1, negative=10, sample=1e-5, workers=4, epochs=5)
w2v.wv.most_similar("отличный", topn=5)
w2v.wv.most_similar(positive=["король", "женщина"], negative=["мужчина"])
```

Готовые русские модели — RusVectōrēs (токены с частью речи: `слово_NOUN`), Navec (natasha).

### 7. GloVe (Pennington et al., 2014)

**Global Vectors**: «сначала посчитать, потом выучить». Регрессия на логарифм совстречаемости с весом:

$$
J = \sum_{i,j}f(X_{ij})\,\big(w_i^{\top}\tilde w_j + b_i + \tilde b_j - \log X_{ij}\big)^2,\qquad
f(x) = \min\big(1, (x/x_{\max})^{3/4}\big),\ x_{\max} = 100 .
$$

$f(0) = 0$ — нулевые пары не учитываются; частые пары не доминируют. Мотив: **отношения** вероятностей
$P(k\mid\text{ice})/P(k\mid\text{steam})$ кодируют смысл (велико для «solid», мало для «gas», ≈ 1 для «water»), и разности векторов
должны их воспроизводить. Ближайшие соседи «frog» в GloVe: frogs, toad, litoria, leptodactylidae, rana, lizard.

### 8. fastText (Bojanowski et al., 2017)

word2vec + **символьные n-граммы** ($n = 3..6$): `where` → `<wh, whe, her, ere, re>` + само слово `<where>`. Вектор слова — **сумма
векторов его n-грамм** (n-граммы хэшируются в фиксированное число корзин). Плюсы: (1) **нет OOV** — вектор есть у любого слова,
в том числе с опечаткой; (2) учитывается **морфология** — «сеть», «сетевой», «сетью» получают похожие векторы. Особенно важно
для русского. Сайт fasttext.cc — готовые векторы на 157 языков и быстрый классификатор.

### 9. Как проверять эмбеддинги

1. **Визуально:** проекция (t-SNE/UMAP/PCA), кластеры слов.
2. **Ближайшие соседи** и задача «найди лишнее слово».
3. **Аналогии:** $v_{\text{король}} - v_{\text{мужчина}} + v_{\text{женщина}}\approx v_{\text{королева}}$ (бенчмарк аналогий Mikolov et al.); исключайте сами слова запроса из ответа!
4. **Бенчмарки сходства** (корреляция косинусов с оценками людей: SimLex, WordSim).
5. **Extrinsic:** качество в целевой задаче (например, тональность фиксированным классификатором) — главный критерий.

**Наложение пространств.** Линейные отношения похожи в разных языках → эмбеддинги двух языков связывает линейное (лучше —
ортогональное) отображение $W$, обученное по небольшому словарю пар (Mikolov et al., 2013, «Exploiting similarities among languages»).
Тот же приём для **семантического сдвига** (Hamilton et al., 2016): учим векторы на текстах разных эпох, выравниваем поворотом и
смотрим, какие слова «уехали» (gay, broadcast).

### 10. Ограничение: один вектор на слово

«Рискую всем **банком**» / «В **банке** не работал кондиционер» / «На **банке** сидела муха»; «the **bank** will not accept cash» /
«the river overflowed the **bank**». Неконтекстные (context-free) методы учитывают контекст при обучении, но при использовании
у слова фиксированный вектор — смесь всех значений. Выход — **языковые модели**: CoVe, TagLM, **ELMo**, Flair, затем BERT и GPT.

---

## Часть C. Контекстные представления

### 11. ELMo (Peters et al., 2018)

Двунаправленная языковая модель (biLM: LSTM слева направо + LSTM справа налево, вход — символьная CNN) предобучается на большом
корпусе. Новое предложение прогоняется через biLM, и представление токена — **обучаемая линейная комбинация всех слоёв**:

$$
\mathrm{ELMo}_k = \gamma\sum_{j=0}^{L}s_j\,h_{k,j},\qquad s = \operatorname{softmax}(w),
$$

включая слой эмбеддингов $j = 0$. Свойства: зависит от всего предложения; глубокое (разные слои — разный уровень: нижние ≈ части речи
и синтаксис, верхние ≈ смысл, полезны для QA); веса $s_j$ подстраиваются под задачу. Разные представления можно конкатенировать
(GloVe + ELMo) или использовать одни как инициализацию других. Дальше — BERT (трансформер-энкодер, [тема 23](../23_transformer_bert/README.md)).

### 12. Представления предложений и текстов

- Простое: среднее (или TF-IDF-взвешенное) эмбеддингов слов — удивительно сильный бейзлайн (SIF: ещё и вычесть первую главную компоненту).
- Universal Sentence Encoder, InferSent; сейчас — **Sentence-BERT** (сиамский BERT, обученный на парах; библиотека `sentence-transformers`)
  и современные эмбеддеры для поиска/RAG ([тема 26](../26_llm_rag_agents/README.md)). Обучаются contrastive-лоссами из [темы 19](../19_contrastive_embeddings/README.md).
- **BERTScore** — сходство двух текстов через попарные косинусы контекстных эмбеддингов токенов (жадное выравнивание) — метрика генерации.

---

## Часть D. Токенизация

### 13. Почему не слова и не символы

Словарь слов не решает OOV и не справляется с «Yeeees! Goooood!», «айфонизация», опечатками, а для русского морфология раздувает
словарь в разы. Символы — нет OOV, но последовательности длинные, и сети приходится заново «собирать» слова. **Subword-токенизация**
— компромисс: частые слова целиком, редкие — из осмысленных кусков («subword» → «sub» + «word»). Токен — элемент текста для сети,
словарь — множество допустимых токенов. Есть и гибридный подход: известное слово — его вектор, иначе посимвольная модель.

### 14. Byte Pair Encoding (Sennrich et al., 2016; алгоритм сжатия Gage, 1994)

**Обучение:**
1. Слово = последовательность символов + маркер конца слова (`</w>`). Словарь = все символы.
2. Посчитать частоты всех пар соседних токенов в корпусе (с учётом частот слов).
3. Самую частую пару объявить новым токеном, заменить во всём корпусе, записать слияние.
4. Повторять, пока словарь не достигнет заданного размера.

Пример лекции: `AABABCABBAABAC` — пара AB встречается 4 раза → D: `ADDCDBADAC`; затем AD (2 раза) → E: `EDCDBEAC`.

**Применение:** к новому слову применяем выученные слияния **в том же порядке**. Различают токен в начале/конце слова и внутри
(`</w>` в BPE, `##` в WordPiece, `Ġ`/пробел-префикс в GPT-2).

**Byte-level BPE (GPT-2):** алфавит — 256 байтов UTF-8, поэтому `<unk>` невозможен; запрещены слияния через категории символов
(буквы с пунктуацией), иначе в словаре были бы «dog», «dog!», «dog?» по отдельности. Платформа OpenAI tokenizer показывает,
что 110 слов ≈ 162 токена английского текста. Для кириллицы английский byte-level BPE катастрофичен (L1: 7 слов → 90 токенов у GPT-2,
19 у ruGPT-3).

### 15. WordPiece, Unigram, SentencePiece

- **WordPiece** (Schuster & Nakajima, 2012; BERT): как BPE, но сливается пара, сильнее всего повышающая **правдоподобие** корпуса
  под униграммной моделью: score $= \frac{p(ab)}{p(a)\,p(b)}$ вместо частоты $p(ab)$. В RoBERTa показали, что BPE работает не хуже.
- **Unigram LM** (Kudo, 2018): начинаем с большого словаря и **удаляем** токены, меньше всего снижающие правдоподобие; сегментация —
  наиболее вероятная (Витерби), можно сэмплировать разные сегментации (subword regularization).
- **SentencePiece:** библиотека, работающая с сырым текстом без предварительного деления на слова (пробел — символ `▁`); реализует BPE и Unigram. Используется в T5, LLaMA.

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=8000, special_tokens=["[UNK]", "[PAD]"]))
tok.encode("Айфонизация кинематографа").tokens   # ['А', 'й', 'ф', 'они', 'за', 'ция', 'кинематографа']

from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
t("Привет, мир!", return_offsets_mapping=True)  # input_ids, attention_mask, позиции в исходной строке
```

**Последствия токенизации:** длина контекста и цена API считаются в токенах; арифметика и посимвольные задачи («сколько букв r в
strawberry») страдают, потому что модель не видит символов; разные токенизаторы несовместимы — эмбеддинги и модель идут в паре.

---

## Часть E. Модели для текстов до трансформеров

### 16. Свёрточная сеть для текста (Kim, 2014)

Эмбеддинги слов → свёртки с окнами 3, 4, 5 слов (много фильтров) → **max-pooling по времени** (фильтр отвечает «есть ли где-то
такая n-грамма») → dropout → softmax. В статье два канала эмбеддингов: замороженный word2vec и дообучаемый. Быстро, хорошо для
классификации; CNN можно сочетать с RNN.

```python
class TextCNN(nn.Module):
    def __init__(self, vocab, emb=128, n_filters=100, sizes=(3, 4, 5), n_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, n_filters, k) for k in sizes)
        self.fc = nn.Linear(n_filters * len(sizes), n_classes)
    def forward(self, x):                                   # x: [B, T]
        e = self.emb(x).transpose(1, 2)                     # [B, E, T]
        return self.fc(torch.cat([F.relu(c(e)).amax(dim=2) for c in self.convs], dim=1))
```

Seq2seq, attention и их место — в [теме 22](../22_seq2seq_attention/README.md); RNN — в [теме 20](../20_rnn/README.md).

### Итоги лекции

Есть классические способы представления; лучше учитывать контекст; стандарт — subword-токенизация; можно получать представления
целых текстов; представления можно обучать вместе с моделью (`nn.Embedding` — просто обучаемая таблица, её можно инициализировать
word2vec/fastText и заморозить или дообучать).

### Чек-лист

- [ ] Бейзлайн: TF-IDF (1–2-граммы) + логистическая регрессия; посмотрите топ-признаки — нет ли утечки.
- [ ] Предобработка проверяется на валидации (стоп-слова и пунктуация могут быть сигналом).
- [ ] Для русского: лемматизация или fastText/subword; токенизатор под язык.
- [ ] Предобученные эмбеддинги — инициализация `nn.Embedding`; заморозить при малом датасете.
- [ ] Контекстные модели (BERT-семейство) — если есть ресурсы; эмбеддинги текстов — Sentence-BERT.

---

## Что почитать / посмотреть

- 📘 Курс Лены Войты: [Word Embeddings](https://lena-voita.github.io/nlp_course/word_embeddings.html) · [NLP Course For You](https://lena-voita.github.io/nlp_course.html) · репозиторий [yandexdataschool/nlp_course](https://github.com/yandexdataschool/nlp_course) (week01 — эмбеддинги)
- 📘 Учебник ML Яндекса: [Нейросети для работы с последовательностями → Word Embeddings](https://education.yandex.ru/handbook/ml/article/nejroseti-dlya-raboty-s-posledovatelnostyami)
- 🎥 Запись лекции «Word Embedding» ([Яндекс Диск](https://disk.yandex.ru/i/PbqlnI7XyNPwrg))
- 📄 Шпаргалка CS230: [раздел Learning word representation](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/#word-representation) · Stanford [CS224n](https://web.stanford.edu/class/cs224n/) (лекции 1–2)
- 📄 Туториалы: [McCormick — word2vec skip-gram](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) · [McCormick — negative sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) · [Lilian Weng — Learning word embedding](https://lilianweng.github.io/posts/2017-10-15-word-embedding/) · [awesome-sentence-embedding](https://github.com/Separius/awesome-sentence-embedding) · [Universal Sentence Encoder (amitness)](https://amitness.com/2020/06/universal-sentence-encoder/) · [OpenAI tokenizer](https://platform.openai.com/tokenizer) · [fasttext.cc](https://fasttext.cc) · [GloVe](https://nlp.stanford.edu/projects/glove/) · [sbert.net](https://www.sbert.net/) · [RusVectōrēs](https://rusvectores.org/)
- 📄 Статьи: [Mikolov 2013a — word2vec](https://arxiv.org/abs/1301.3781) · [Mikolov 2013b — negative sampling, фразы](https://arxiv.org/abs/1310.4546) · [Mikolov 2013 — языки](https://arxiv.org/abs/1309.4168) · [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) · [fastText](https://arxiv.org/abs/1607.04606) · [Levy & Goldberg 2014 — SGNS как разложение PMI](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) · [Hamilton 2016 — семантический сдвиг](https://www.aclweb.org/anthology/P16-1141.pdf) · [ELMo](https://arxiv.org/abs/1802.05365) · [Sentence-BERT](https://arxiv.org/abs/1908.10084) · [BERTScore](https://arxiv.org/abs/1904.09675) · [BPE (Sennrich)](https://arxiv.org/abs/1508.07909) · [WordPiece (Schuster & Nakajima)](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf) · [Unigram LM (Kudo)](https://arxiv.org/abs/1804.10959) · [SentencePiece](https://arxiv.org/abs/1808.06226) · [Kim 2014 — CNN for sentence classification](https://arxiv.org/abs/1408.5882) · [Морфология в эмбеддингах (Luong 2013)](https://nlp.stanford.edu/~lmthang/data/papers/conll13_morpho.pdf)
- 📘 Кобозева И. М. «Лингвистическая семантика»; Соссюр Ф. «Курс общей лингвистики» (из лекции Сириуса).

## Источники курса

- `Лекция_08.pptx` (DEEP LEARNING, день 8, 55 слайдов), слайды 1–38: OHE и counts, word2vec (CBOW, skip-gram, тонкости), fastText, GloVe, проверка представлений, семантический сдвиг, наложение языков, контекстные представления (ELMo), представления текстов (SentenceBERT, BERTScore), BPE, BPE в GPT-2, WordPiece, CNN для текстов. Слайды 39–53 — [тема 22](../22_seq2seq_attention/README.md).
- `NLP лекция_Сириус_Мамаев.html`: семиотика, one-hot, хэширование, дистрибутивная семантика, матрица совстречаемости, SVD и теорема Эккарта–Янга, LSA, word2vec с выводом градиента, negative sampling, примеры RusVectōrēs; языковые модели — [тема 24](../24_language_models_gpt/README.md).
- `simple_sentiment_analysis.ipynb`: тональность рецензий IMDB, CountVectorizer/TF-IDF + LR/SVM/SGD, `min_df`, стоп-слова, n-граммы.
- `0000.parquet` — рецензии Кинопоиска ([blinoff/kinopoisk](https://huggingface.co/datasets/blinoff/kinopoisk), 70 МБ, в репозиторий не копируется): данные L1-примеров.


<!-- materials:start -->
**Файлы курса в репозитории:**

- DL, лекция 8: представления текста, seq2seq, внимание (лекция курса, файл в репозиторий не включён)
- NLP-лекция (Сириус): дистрибутивная семантика, word2vec, LM (лекция курса, файл в репозиторий не включён)
- [Тональность: мешок слов и TF-IDF](../../materials/seminars/21_simple_sentiment_analysis.ipynb)
- [Семинар 7: RNN, LSTM, предобработка текста](../../materials/seminars/20_Семинар_7_RNN_предобработка_текста.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1D_XIvuqGltfgfkJ2sDcwh5ivuPFKW6ue)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [22. Seq2seq и внимание →](../22_seq2seq_attention/README.md)
