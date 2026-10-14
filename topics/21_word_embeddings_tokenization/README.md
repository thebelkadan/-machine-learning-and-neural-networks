# 21. Представления текста: от мешка слов до word2vec, BPE и контекстных эмбеддингов

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Сеть принимает числа, поэтому текст надо **токенизировать** (слова → subword-токены BPE/WordPiece) и **векторизовать**.
Классика: **мешок слов / TF-IDF** — разреженный вектор на весь текст, порядок слов теряется; с линейной моделью это до сих пор
сильный бейзлайн. **Дистрибутивная гипотеза** («слово определяется соседями») даёт **плотные эмбеддинги**: SVD матрицы
совстречаемости (LSA), **word2vec** (skip-gram/CBOW + negative sampling), **GloVe**, **fastText** (+ символьные n-граммы, нет OOV).
Они **неконтекстные**: у «банка» один вектор. **Контекстные** (ELMo → BERT/GPT) считают вектор слова по всему предложению.

| Термин | Определение |
|---|---|
| Токен / словарь | единица входа сети / множество допустимых токенов; вне словаря — `<unk>` (OOV) |
| One-hot / BoW / TF-IDF | вектор размера словаря / счётчики слов текста / счётчики × редкость слова в коллекции |
| Дистрибутивная гипотеза | «You shall know a word by the company it keeps» (Firth, 1957) |
| PMI / PPMI | $\log\frac{p(w,c)}{p(w)p(c)}$ / $\max(0,\mathrm{PMI})$ |
| Skip-gram / CBOW | по слову предсказать контекст / по контексту — слово |
| Negative sampling | вместо softmax по словарю — логрегрессия «реальная пара vs $k$ случайных» |
| fastText | вектор слова = сумма векторов его символьных n-грамм (`<wh, whe, her, ere, re>`) |
| BPE / WordPiece | словарь subword-токенов слиянием самой частой пары / пары, сильнее всего повышающей правдоподобие |
| Byte-level BPE | BPE над байтами UTF-8 (GPT-2): нет `<unk>`, но кириллица дробится на байты |
| Контекстный эмбеддинг | вектор токена зависит от предложения (ELMo, BERT); стандарт сейчас |

$$
\text{tf-idf}(w,d)=\frac{n_{wd}}{\sum_{w'}n_{w'd}}\cdot\log\frac{N}{\mathrm{df}(w)},\qquad
p(o\mid c)=\frac{e^{u_o^{\top}v_c}}{\sum_{w\in V}e^{u_w^{\top}v_c}},\qquad
J_{\text{SGNS}}=\log\sigma(u_o^{\top}v_c)+\sum_{i=1}^{k}\log\sigma(-u_{n_i}^{\top}v_c)
$$

## Конспект

Разделы раскрываются по клику.

**Часть A. Что такое «слово» для модели**

<details>
<summary><b>1. Немного лингвистики (лекция Сириуса)</b></summary>

**Языковой знак** — единство **означающего** (форма: буквы, звуки) и **означаемого** (смысл), по Соссюру. В означаемом различают
**денотационный** смысл (прямое, словарное значение) и **прагматический** (зависит от говорящего, слушающего, ситуации), плюс
**коннотации** — сопутствующие оттенки. Словарями всё это не покрыть, поэтому смысл извлекают из употребления — из **корпусов**.

Задача векторизации: превратить текст во что-то числовое. Два пути:
1. вектор на **весь текст** (мешок слов, TF-IDF, sentence-эмбеддинги);
2. вектор на **каждый токен** → последовательность векторов для RNN/CNN/трансформера.

</details>

<details>
<summary><b>2. One-hot и хэширование</b></summary>

**One-hot:** словарь размера $\lvert V\rvert$, слову $i$ соответствует единичный вектор $e_i$. Проблемы: размерность = размер словаря
(десятки–сотни тысяч), **все слова одинаково далеки** друг от друга ($e_i^{\top}e_j = 0$), нет обобщения на незнакомые слова.

**Hashing trick:** индекс слова = `hash(word) % n` — фиксированная размерность без хранения словаря, ценой коллизий
(`HashingVectorizer` в sklearn, n-граммы в fastText).

</details>

<details>
<summary><b>3. Мешок слов и TF-IDF</b></summary>

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

</details>

**Часть B. Дистрибутивная семантика**

<details>
<summary><b>4. Гипотеза и матрица совстречаемости</b></summary>

> «You shall know a word by the company it keeps» — J. R. Firth, 1957. Значение слова определяется словами, которые часто стоят рядом.

Пример из лекции: «Полосатая маленькая \*\*\*\*\* мурлычит и пьёт молоко» — пропуск угадывается по контексту. Контекст — слова в окне
$\pm h$ вокруг. Матрица совстречаемости $X_{ij}$ — сколько раз слово $j$ встретилось в окне слова $i$. Строки уже отражают
семантику, но **разреженные** и размерности $\lvert V\rvert$. Сырые счётчики забиты частыми словами, поэтому их переводят в
**PMI**:

$$
\mathrm{PMI}(w,c) = \log\frac{p(w,c)}{p(w)\,p(c)},\qquad \mathrm{PPMI} = \max(0, \mathrm{PMI}) .
$$

</details>

<details>
<summary><b>5. SVD и LSA</b></summary>

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

</details>

<details>
<summary><b>6. word2vec (Mikolov et al., 2013)</b></summary>

Вместо подсчёта — **предсказание**. Трюк самообучения: учим модель решать вспомогательную задачу на неразмеченном корпусе, а
используем её **внутренние представления** (принцип трансферного обучения).

- **Skip-gram:** по центральному слову предсказать слова контекста. Лучше для редких слов и небольших корпусов (окно ~10).
- **CBOW** (Continuous Bag of Words): по сумме/среднему контекста предсказать центральное слово. Быстрее, для больших корпусов (окно ~5).

У каждого слова **два вектора**: $v_w$ (центральное) и $u_w$ (контекст). Функционал skip-gram и вероятность:

$$
J(\theta) = \frac1T\sum_{t=1}^{T}\sum_{-h\le j\le h,\,j\ne0}\log p(w_{t+j}\mid w_t),\qquad
p(o\mid c) = \frac{\exp(u_o^{\top}v_c)}{\sum_{w\in V}\exp(u_w^{\top}v_c)} .
$$

Градиент (вывод — см. раздел «2. Skip-gram: градиент полного softmax»):
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

</details>

<details>
<summary><b>7. GloVe (Pennington et al., 2014)</b></summary>

**Global Vectors**: «сначала посчитать, потом выучить». Регрессия на логарифм совстречаемости с весом:

$$
J = \sum_{i,j}f(X_{ij})\,\big(w_i^{\top}\tilde w_j + b_i + \tilde b_j - \log X_{ij}\big)^2,\qquad
f(x) = \min\big(1, (x/x_{\max})^{3/4}\big),\ x_{\max} = 100 .
$$

$f(0) = 0$ — нулевые пары не учитываются; частые пары не доминируют. Мотив: **отношения** вероятностей
$P(k\mid\text{ice})/P(k\mid\text{steam})$ кодируют смысл (велико для «solid», мало для «gas», ≈ 1 для «water»), и разности векторов
должны их воспроизводить. Ближайшие соседи «frog» в GloVe: frogs, toad, litoria, leptodactylidae, rana, lizard.

</details>

<details>
<summary><b>8. fastText (Bojanowski et al., 2017)</b></summary>

word2vec + **символьные n-граммы** ($n = 3..6$): `where` → `<wh, whe, her, ere, re>` + само слово `<where>`. Вектор слова — **сумма
векторов его n-грамм** (n-граммы хэшируются в фиксированное число корзин). Плюсы: (1) **нет OOV** — вектор есть у любого слова,
в том числе с опечаткой; (2) учитывается **морфология** — «сеть», «сетевой», «сетью» получают похожие векторы. Особенно важно
для русского. Сайт fasttext.cc — готовые векторы на 157 языков и быстрый классификатор.

</details>

<details>
<summary><b>9. Как проверять эмбеддинги</b></summary>

1. **Визуально:** проекция (t-SNE/UMAP/PCA), кластеры слов.
2. **Ближайшие соседи** и задача «найди лишнее слово».
3. **Аналогии:** $v_{\text{король}} - v_{\text{мужчина}} + v_{\text{женщина}}\approx v_{\text{королева}}$ (бенчмарк аналогий Mikolov et al.); исключайте сами слова запроса из ответа!
4. **Бенчмарки сходства** (корреляция косинусов с оценками людей: SimLex, WordSim).
5. **Extrinsic:** качество в целевой задаче (например, тональность фиксированным классификатором) — главный критерий.

**Наложение пространств.** Линейные отношения похожи в разных языках → эмбеддинги двух языков связывает линейное (лучше —
ортогональное) отображение $W$, обученное по небольшому словарю пар (Mikolov et al., 2013, «Exploiting similarities among languages»).
Тот же приём для **семантического сдвига** (Hamilton et al., 2016): учим векторы на текстах разных эпох, выравниваем поворотом и
смотрим, какие слова «уехали» (gay, broadcast).

</details>

<details>
<summary><b>10. Ограничение: один вектор на слово</b></summary>

«Рискую всем **банком**» / «В **банке** не работал кондиционер» / «На **банке** сидела муха»; «the **bank** will not accept cash» /
«the river overflowed the **bank**». Неконтекстные (context-free) методы учитывают контекст при обучении, но при использовании
у слова фиксированный вектор — смесь всех значений. Выход — **языковые модели**: CoVe, TagLM, **ELMo**, Flair, затем BERT и GPT.

</details>

**Часть C. Контекстные представления**

<details>
<summary><b>11. ELMo (Peters et al., 2018)</b></summary>

Двунаправленная языковая модель (biLM: LSTM слева направо + LSTM справа налево, вход — символьная CNN) предобучается на большом
корпусе. Новое предложение прогоняется через biLM, и представление токена — **обучаемая линейная комбинация всех слоёв**:

$$
\mathrm{ELMo}_k = \gamma\sum_{j=0}^{L}s_j\,h_{k,j},\qquad s = \operatorname{softmax}(w),
$$

включая слой эмбеддингов $j = 0$. Свойства: зависит от всего предложения; глубокое (разные слои — разный уровень: нижние ≈ части речи
и синтаксис, верхние ≈ смысл, полезны для QA); веса $s_j$ подстраиваются под задачу. Разные представления можно конкатенировать
(GloVe + ELMo) или использовать одни как инициализацию других. Дальше — BERT (трансформер-энкодер, [тема 23](../23_transformer_bert/README.md)).

</details>

<details>
<summary><b>12. Представления предложений и текстов</b></summary>

- Простое: среднее (или TF-IDF-взвешенное) эмбеддингов слов — удивительно сильный бейзлайн (SIF: ещё и вычесть первую главную компоненту).
- Universal Sentence Encoder, InferSent; сейчас — **Sentence-BERT** (сиамский BERT, обученный на парах; библиотека `sentence-transformers`)
  и современные эмбеддеры для поиска/RAG ([тема 26](../26_llm_rag_agents/README.md)). Обучаются contrastive-лоссами из [темы 19](../19_contrastive_embeddings/README.md).
- **BERTScore** — сходство двух текстов через попарные косинусы контекстных эмбеддингов токенов (жадное выравнивание) — метрика генерации.

</details>

**Часть D. Токенизация**

<details>
<summary><b>13. Почему не слова и не символы</b></summary>

Словарь слов не решает OOV и не справляется с «Yeeees! Goooood!», «айфонизация», опечатками, а для русского морфология раздувает
словарь в разы. Символы — нет OOV, но последовательности длинные, и сети приходится заново «собирать» слова. **Subword-токенизация**
— компромисс: частые слова целиком, редкие — из осмысленных кусков («subword» → «sub» + «word»). Токен — элемент текста для сети,
словарь — множество допустимых токенов. Есть и гибридный подход: известное слово — его вектор, иначе посимвольная модель.

</details>

<details>
<summary><b>14. Byte Pair Encoding (Sennrich et al., 2016; алгоритм сжатия Gage, 1994)</b></summary>

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

</details>

<details>
<summary><b>15. WordPiece, Unigram, SentencePiece</b></summary>

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

</details>

**Часть E. Модели для текстов до трансформеров**

<details>
<summary><b>16. Свёрточная сеть для текста (Kim, 2014)</b></summary>

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

</details>

<details>
<summary><b>Итоги лекции</b></summary>

Есть классические способы представления; лучше учитывать контекст; стандарт — subword-токенизация; можно получать представления
целых текстов; представления можно обучать вместе с моделью (`nn.Embedding` — просто обучаемая таблица, её можно инициализировать
word2vec/fastText и заморозить или дообучать).

</details>

<details>
<summary><b>Чек-лист</b></summary>

- [ ] Бейзлайн: TF-IDF (1–2-граммы) + логистическая регрессия; посмотрите топ-признаки — нет ли утечки.
- [ ] Предобработка проверяется на валидации (стоп-слова и пунктуация могут быть сигналом).
- [ ] Для русского: лемматизация или fastText/subword; токенизатор под язык.
- [ ] Предобученные эмбеддинги — инициализация `nn.Embedding`; заморозить при малом датасете.
- [ ] Контекстные модели (BERT-семейство) — если есть ресурсы; эмбеддинги текстов — Sentence-BERT.

</details>

**Математика и выводы**

<details>
<summary><b>1. TF-IDF: откуда логарифм</b></summary>

Если документ выбирается случайно, $p(w\in d) = \mathrm{df}(w)/N$, и $\text{idf}(w) = \log\frac{N}{\mathrm{df}(w)} = -\log p(w\in d)$ —
**собственная информация** события «слово есть в документе»: редкое слово сообщает много, слово из каждого документа — ноль.
TF-IDF — «частота × информативность». Варианты sklearn:

$$
\text{idf}_{\text{smooth}}(w) = \ln\frac{1+N}{1+\mathrm{df}(w)} + 1,\qquad \text{tf}_{\text{sublinear}} = 1 + \ln n_{wd},\qquad x_d \leftarrow \frac{x_d}{\lVert x_d\rVert_2} .
$$

«+1» в числителе и знаменателе — как будто есть документ со всеми словами (нет деления на 0); «+1» снаружи — слово из каждого
документа не обнуляется. L2-нормировка делает скалярное произведение косинусом и убирает зависимость от длины текста.

</details>

<details>
<summary><b>2. Skip-gram: градиент полного softmax</b></summary>

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

</details>

<details>
<summary><b>3. Hierarchical softmax</b></summary>

Слова — листья бинарного дерева (Хаффмана: частые слова ближе к корню). Каждому внутреннему узлу $n$ — вектор $u_n$; путь к слову
$w$ — узлы $n_1,\ldots,n_L$ с направлениями $d_j\in\lbrace +1, -1\rbrace$:

$$
p(w\mid c) = \prod_{j=1}^{L}\sigma\big(d_j\,u_{n_j}^{\top}v_c\big).
$$

Так как $\sigma(x) + \sigma(-x) = 1$, в каждом узле вероятность делится между ветками, и $\sum_w p(w\mid c) = 1$ автоматически.
Стоимость — $O(\log\lvert V\rvert\cdot d)$, для частых слов путь ещё короче.

</details>

<details>
<summary><b>4. Negative sampling</b></summary>

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

</details>

<details>
<summary><b>5. SGNS неявно раскладывает матрицу PMI (Levy & Goldberg, 2014)</b></summary>

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

</details>

<details>
<summary><b>6. GloVe: вывод из отношений вероятностей</b></summary>

Пусть $P_{ik} = X_{ik}/X_i$ — вероятность контекста $k$ у слова $i$. Смысл несут **отношения** $P_{ik}/P_{jk}$. Требуем
$F\big((w_i - w_j)^{\top}\tilde w_k\big) = P_{ik}/P_{jk}$. Чтобы разность в аргументе переходила в отношение, $F$ должна быть
гомоморфизмом $(\mathbb R, +)\to(\mathbb R_{>0}, \times)$, то есть $F = \exp$:

$$
w_i^{\top}\tilde w_k = \log P_{ik} = \log X_{ik} - \log X_i .
$$

$\log X_i$ не зависит от $k$ — поглощается смещением $b_i$; для симметрии добавляем $\tilde b_k$:
$w_i^{\top}\tilde w_k + b_i + \tilde b_k = \log X_{ik}$. Взвешенный МНК даёт функционал GloVe с весом $f(X_{ij})$:
$f(0) = 0$ (иначе $\log 0$), $f$ не убывает (редкие пары шумные) и насыщается (частые пары не доминируют). Итоговый вектор — $w + \tilde w$.

</details>

<details>
<summary><b>7. Аналогии: 3CosAdd и 3CosMul</b></summary>

Для нормированных векторов поиск $b^{*} = \arg\max_{x}\cos\big(x, b - a + a^{*}\big)$ эквивалентен

$$
\arg\max_x\big[\cos(x, b) - \cos(x, a) + \cos(x, a^{*})\big]\quad\text{(3CosAdd)},
$$

то есть одно сильное сходство может «перекричать» остальные. Устойчивее мультипликативный вариант
$\arg\max_x\frac{\cos(x, b)\cos(x, a^{*})}{\cos(x, a) + \varepsilon}$ (3CosMul, косинусы сдвинуты в $[0, 1]$). Обязательно исключать
$a, b, a^{*}$ из кандидатов: иначе ответом часто оказывается само слово запроса (так работают «красивые» демо с оговорками).

</details>

<details>
<summary><b>8. Выравнивание пространств: ортогональная задача Прокруста</b></summary>

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

</details>

<details>
<summary><b>9. Эккарт–Янг и выбор размерности</b></summary>

$\lVert A - A_r\rVert_F^2 = \sum_{i>r}\sigma_i^2$, относительная ошибка $\sqrt{\sum_{i>r}\sigma_i^2/\sum_i\sigma_i^2}$. Для PPMI-матриц спектр
убывает медленно (это не «почти низкоранговая» матрица), поэтому размерность 100–300 — компромисс между шумом и выразительностью,
выбираемый по downstream-задаче, а не по доле объяснённой дисперсии. Усечённое SVD больших разреженных матриц считают
рандомизированно (Halko et al.): случайная проекция на $r + p$ векторов, несколько степенных итераций, QR и SVD маленькой матрицы.

</details>

<details>
<summary><b>10. Токенизация как сжатие и как вероятностная модель</b></summary>

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

</details>

<details>
<summary><b>11. ELMo: функционал biLM</b></summary>

$$
\sum_{k=1}^{N}\Big(\log p(t_k\mid t_1,\ldots,t_{k-1};\,\Theta_x, \overrightarrow\Theta_{\text{LSTM}}, \Theta_s)
+ \log p(t_k\mid t_{k+1},\ldots,t_N;\,\Theta_x, \overleftarrow\Theta_{\text{LSTM}}, \Theta_s)\Big),
$$

$\Theta_x$ (символьная CNN-входная часть) и $\Theta_s$ (softmax) общие у направлений, LSTM — свои. Направления обучаются **независимо**
(каждое — обычная LM), поэтому слово не «видит себя» — в отличие от наивного двунаправленного предсказания. BERT решает ту же
проблему маскированием ([тема 23](../23_transformer_bert/README.md)).

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему у «хорошо» и «плохо» близкие word2vec-векторы?</summary>

одинаковые контексты: дистрибутивная близость ≠ синонимия, антонимы рядом («смешно» ~ «скучно»).

</details>

<details>
<summary>Зачем в word2vec два вектора на слово ($v$ — центральное, $u$ — контекст)?</summary>

иначе $u_w^{\top}v_w=\lVert v_w\rVert^2$ — слово «тянет» себя в свой контекст; на выходе берут $v$ или $v+u$.

</details>

<details>
<summary>Зачем negative sampling?</summary>

softmax по словарю стоит $O(\lvert V\rvert)$ на пару; с $k$ негативами — $O(k)$. Негативы — из униграмм в степени 3/4.

</details>

<details>
<summary>Как fastText получает вектор незнакомого слова?</summary>

сумма векторов его символьных n-грамм (хэшированных).

</details>

<details>
<summary>Почему BPE, а не слова?</summary>

нет OOV, словарь фиксированного размера (30–100k), редкие слова — из частей. Но больше токенов на текст.

</details>

<details>
<summary>Почему GPT-2 режет русскую фразу из 7 слов на 90 токенов, а ruGPT-3 — на 19?</summary>

byte-level BPE учили на английском: кириллица распадается на байты. Токенизатор выбирают под язык — от этого зависят длина контекста и цена.

</details>

<details>
<summary>Когда TF-IDF + линейная модель лучше нейросети?</summary>

мало данных, длинные тексты, нужна скорость и интерпретируемость; всегда как бейзлайн.

</details>

<details>
<summary>Утечка в тексте: 68% рецензий содержат «N из 10», а часть — слово «нейтральная»</summary>

модель учит подсказки, а не язык. Проверяйте топ-признаки!

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- **SGNS ≈ разложение матрицы PMI:** Levy & Goldberg (2014) показали, что оптимум skip-gram с $k$ негативами — $u_c^{\top}v_w=\mathrm{PMI}(w,c)-\log k$. Поэтому PPMI+SVD из примера — близкий родственник word2vec.
- **GloVe:** $\sum_{ij}f(X_{ij})\,(w_i^{\top}\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2$ — регрессия на логарифм совстречаемости с весом $f$, обрезающим частые пары.
- **Проверка эмбеддингов:** ближайшие соседи, «найди лишнее», аналогии ($\text{king}-\text{man}+\text{woman}\approx\text{queen}$; в примере выше одна из двух удалась), бенчмарки сходства, качество в downstream-задаче. Линейные отношения похожи в разных языках → пространства выравнивают ортогональным поворотом (перевод слов, семантический сдвиг во времени).
- **Контекстные:** ELMo — двунаправленная LSTM-LM, вектор слова = обучаемая взвешенная сумма слоёв (низкие ≈ синтаксис, высокие ≈ семантика); дальше BERT (тема 23). Текст целиком — Sentence-BERT, mean-pooling.
- **Текстовая CNN** (Kim, 2014): свёртки 3/4/5 слов над эмбеддингами + max-pool по времени — быстрый сильный классификатор.
- **Ноутбук `simple_sentiment_analysis` (рецензии IMDB из NLTK, 5-fold accuracy):** CountVectorizer+LR 0.841, TF-IDF+LR 0.821, `min_df=50` 0.813, LinearSVC 0.833, SGD 0.755 ± 0.11, биграммы 0.853, символьные 3–5-граммы 0.82.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# BPE с нуля: словарь растёт слиянием самой частой пары соседних токенов
from collections import Counter

def merge_word(w, pair):
    out, i = [], 0
    while i < len(w):
        if i + 1 < len(w) and (w[i], w[i + 1]) == pair: out.append(w[i] + w[i + 1]); i += 2
        else: out.append(w[i]); i += 1
    return tuple(out)

def bpe_train(word_freq, n_merges):
    vocab = {tuple(w) + ("</w>",): f for w, f in word_freq.items()}          # </w> — конец слова
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, f in vocab.items():
            for p in zip(w, w[1:]): pairs[p] += f
        if not pairs: break
        best = max(pairs, key=pairs.get); merges.append(best)
        vocab = {merge_word(w, best): f for w, f in vocab.items()}
    return merges

def bpe_encode(word, merges):                                               # применяем слияния в порядке обучения
    w = tuple(word) + ("</w>",)
    for pair in merges: w = merge_word(w, pair)
    return w

merges = bpe_train({"низкий": 5, "низкая": 2, "новейший": 6, "широчайший": 3, "новый": 4, "низший": 2}, 12)
print("слияния:", ["+".join(m) for m in merges])
for w in ["новейший", "низкий", "широкий", "новизна"]:
    print(f"{w:10s} → {bpe_encode(w, merges)}")                               # незнакомые слова — из кусочков, без <unk>
```

Результат запуска:

```text
слияния: ['й+</w>', 'и+й</w>', 'ш+ий</w>', 'н+о', 'но+в', 'н+и', 'ни+з', 'й+ший</w>', 'низ+к', 'нов+е', 'нове+йший</w>', 'низк+ий</w>']
новейший   → ('новейший</w>',)
низкий     → ('низкий</w>',)
широкий    → ('ш', 'и', 'р', 'о', 'к', 'ий</w>')
новизна    → ('нов', 'и', 'з', 'н', 'а', '</w>')
```

```python
# Данные курса: 36 591 рецензия Кинопоиска (blinoff/kinopoisk на Hugging Face), метка grade3 = Good / Neutral / Bad
import os, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
paths = ["0000.parquet", "materials/data/0000.parquet", "../0000.parquet"]
src = next((p for p in paths if os.path.exists(p)),
           "https://huggingface.co/datasets/blinoff/kinopoisk/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")  # ~70 МБ
df = pd.read_parquet(src, columns=["grade3", "content"])
df["text"] = df.content.str.replace("\xa0", " ")
print(df.shape, df.grade3.value_counts(normalize=True).round(3).to_dict())

# BPE на корпусе (библиотека tokenizers) и сравнение с готовыми токенизаторами
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(models.BPE(unk_token="[UNK]")); tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(df.text, trainers.BpeTrainer(vocab_size=8000, special_tokens=["[UNK]", "[PAD]"]))
s = "Айфонизация кинематографа: переобученные сценаристы снимают неинтересные ремейки"
print("наш BPE-8k:", tok.encode(s).tokens)
try:
    from transformers import AutoTokenizer
    for name in ["gpt2", "sberbank-ai/rugpt3small_based_on_gpt2"]:
        t = AutoTokenizer.from_pretrained(name)
        for x in [s, "Overfitted screenwriters keep shooting boring remakes"]:
            ids = t.encode(x)
            print(f"{name.split('/')[-1][:12]:12s} {len(x.split())} слов → {len(ids):2d} токенов: {[t.decode([i]) for i in ids][:8]}")
except Exception as e:
    print("не удалось загрузить с Hugging Face Hub (нужен интернет):", type(e).__name__, str(e)[:120])
# GPT-2 — byte-level BPE на английском корпусе: кириллическая буква = 2 байта, токен-«огрызок» печатается как �
```

Результат запуска:

```text
(36591, 3) {'Good': 0.745, 'Bad': 0.13, 'Neutral': 0.125}
наш BPE-8k: ['А', 'й', 'ф', 'они', 'за', 'ция', 'кинематографа', ':', 'пере', 'обу', 'ченные', 'сценаристы', 'снимают', 'не', 'интересные', 'ре', 'мей', 'ки']
gpt2         7 слов → 90 токенов: ['�', '�', '�', '�', '�', '�', 'о', 'н']
gpt2         6 слов →  9 токенов: ['Over', 'fitted', ' screen', 'writers', ' keep', ' shooting', ' boring', ' rem']
rugpt3small_ 7 слов → 19 токенов: ['Ай', 'фони', 'зация', ' кинемат', 'ографа', ':', ' пере', 'об']
rugpt3small_ 6 слов → 14 токенов: ['O', 'ver', 'fit', 'ted', ' screen', 'writ', 'ers', ' keep']
```

```python
# TF-IDF + логистическая регрессия: сильный бейзлайн. Оценки «N из 10» в тексте удаляем — иначе это подсказка ответа
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

GRADE = r"\d+(?:[.,]\d+)?\s*(?:из|/)\s*10\b"
low = df.text.str.lower().str.replace("ё", "е")
print(f"рецензий с оценкой в тексте: {low.str.contains(GRADE).mean():.0%}")
df["clean"] = low.str.replace(GRADE, " ", regex=True)
tr, te = train_test_split(df, test_size=0.25, random_state=0, stratify=df.grade3)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True)
Xtr, Xte = vec.fit_transform(tr.clean), vec.transform(te.clean)
clf = LogisticRegression(C=5, max_iter=2000, class_weight="balanced").fit(Xtr, tr.grade3)
print(classification_report(te.grade3, clf.predict(Xte), digits=3))
fn = vec.get_feature_names_out()
for k, c in enumerate(clf.classes_): print(f"{c:8s}", list(fn[np.argsort(-clf.coef_[k])[:9]]))
```

Результат запуска:

```text
рецензий с оценкой в тексте: 68%
              precision    recall  f1-score   support

         Bad      0.705     0.772     0.737      1188
        Good      0.907     0.918     0.913      6816
     Neutral      0.377     0.311     0.341      1144

    accuracy                          0.823      9148
   macro avg      0.663     0.667     0.663      9148
weighted avg      0.814     0.823     0.818      9148

Bad      ['бред', 'увы', 'зачем', 'сценарий', 'вообще', 'ничего', 'не понравился', 'абсолютно', 'нет']
Good     ['каждый', 'браво', 'не только', 'конечно же', 'отлично', 'великолепный', 'он', 'очень', 'идеально']
Neutral  ['нейтральная', 'большего', 'одной стороны', 'можно посмотреть', 'могу сказать', 'неплохой', 'те же', 'остальное', 'местами']
```

```python
# Дистрибутивная семантика своими руками: совстречаемость (окно ±4) → PPMI → SVD = плотные эмбеддинги слов
import scipy.sparse as sp
from collections import Counter
from sklearn.utils.extmath import randomized_svd

docs = [re.findall(r"[а-яa-z]+", t) for t in low]
V = [w for w, _ in Counter(w for d in docs for w in d).most_common(20000)]; ix = {w: i for i, w in enumerate(V)}
rows, cols = [], []
for d in docs:
    ids = np.array([ix.get(w, -1) for w in d])
    for off in range(1, 5):
        a, b = ids[:-off], ids[off:]; m = (a >= 0) & (b >= 0)
        rows += [a[m], b[m]]; cols += [b[m], a[m]]
C = sp.coo_matrix((np.ones(sum(map(len, rows)), np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(len(V),) * 2).tocsr().tocoo()
total = C.data.sum(); pw = np.bincount(C.row, C.data, len(V)) / total
pc = np.bincount(C.col, C.data, len(V)) ** 0.75; pc /= pc.sum()                 # сглаживание контекстов ^0.75, как в word2vec
pmi = np.log(C.data / total / (pw[C.row] * pc[C.col]))
M = sp.csr_matrix((np.maximum(pmi, 0), (C.row, C.col)), shape=C.shape)
U, S, _ = randomized_svd(M, 200, random_state=0)
E = U * np.sqrt(S); E /= np.linalg.norm(E, axis=1, keepdims=True)

nearest = lambda w, k=7: [V[j] for j in np.argsort(-(E @ E[ix[w]]))[1:k + 1]]
for w in ["отличный", "скучный", "комедия", "режиссер", "музыка", "смешно"]:
    print(f"{w:9s} → {nearest(w)}")
def analogy(a, b, c):                                                          # b - a + c
    s = E @ (E[ix[b]] - E[ix[a]] + E[ix[c]]); s[[ix[a], ix[b], ix[c]]] = -9
    return V[int(np.argmax(s))]
print("мужчина:женщина = король:?", analogy("мужчина", "женщина", "король"), "| он:она = актер:?", analogy("он", "она", "актер"))
```

Результат запуска:

```text
отличный  → ['великолепный', 'потрясающий', 'замечательный', 'хороший', 'прекрасный', 'качественный', 'шикарный']
скучный   → ['нудный', 'глупый', 'затянутый', 'неинтересный', 'пошлый', 'предсказуемый', 'примитивный']
комедия   → ['мелодрама', 'драма', 'смешная', 'романтическая', 'пародия', 'криминальная', 'фантастика']
режиссер  → ['сценарист', 'нолан', 'снял', 'продюсер', 'автор', 'спилберг', 'кристофер']
музыка    → ['саундтрек', 'сопровождение', 'музыкальное', 'мелодия', 'музыкальная', 'заглавная', 'подобрана']
смешно    → ['забавно', 'глупо', 'весело', 'нелепо', 'скучно', 'грустно', 'противно']
мужчина:женщина = король:? королева | он:она = актер:? играет
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Курс Лены Войты: [Word Embeddings](https://lena-voita.github.io/nlp_course/word_embeddings.html) · [NLP Course For You](https://lena-voita.github.io/nlp_course.html) · репозиторий [yandexdataschool/nlp_course](https://github.com/yandexdataschool/nlp_course) (week01 — эмбеддинги)
- Учебник ML Яндекса: [Нейросети для работы с последовательностями → Word Embeddings](https://education.yandex.ru/handbook/ml/article/nejroseti-dlya-raboty-s-posledovatelnostyami)
- Запись лекции «Word Embedding» ([Яндекс Диск](https://disk.yandex.ru/i/PbqlnI7XyNPwrg))
- Шпаргалка CS230: [раздел Learning word representation](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/#word-representation) · Stanford [CS224n](https://web.stanford.edu/class/cs224n/) (лекции 1–2)
- Туториалы: [McCormick — word2vec skip-gram](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) · [McCormick — negative sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) · [Lilian Weng — Learning word embedding](https://lilianweng.github.io/posts/2017-10-15-word-embedding/) · [awesome-sentence-embedding](https://github.com/Separius/awesome-sentence-embedding) · [Universal Sentence Encoder (amitness)](https://amitness.com/2020/06/universal-sentence-encoder/) · [OpenAI tokenizer](https://platform.openai.com/tokenizer) · [fasttext.cc](https://fasttext.cc) · [GloVe](https://nlp.stanford.edu/projects/glove/) · [sbert.net](https://www.sbert.net/) · [RusVectōrēs](https://rusvectores.org/)
- Статьи: [Mikolov 2013a — word2vec](https://arxiv.org/abs/1301.3781) · [Mikolov 2013b — negative sampling, фразы](https://arxiv.org/abs/1310.4546) · [Mikolov 2013 — языки](https://arxiv.org/abs/1309.4168) · [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) · [fastText](https://arxiv.org/abs/1607.04606) · [Levy & Goldberg 2014 — SGNS как разложение PMI](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) · [Hamilton 2016 — семантический сдвиг](https://www.aclweb.org/anthology/P16-1141.pdf) · [ELMo](https://arxiv.org/abs/1802.05365) · [Sentence-BERT](https://arxiv.org/abs/1908.10084) · [BERTScore](https://arxiv.org/abs/1904.09675) · [BPE (Sennrich)](https://arxiv.org/abs/1508.07909) · [WordPiece (Schuster & Nakajima)](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf) · [Unigram LM (Kudo)](https://arxiv.org/abs/1804.10959) · [SentencePiece](https://arxiv.org/abs/1808.06226) · [Kim 2014 — CNN for sentence classification](https://arxiv.org/abs/1408.5882) · [Морфология в эмбеддингах (Luong 2013)](https://nlp.stanford.edu/~lmthang/data/papers/conll13_morpho.pdf)
- Кобозева И. М. «Лингвистическая семантика»; Соссюр Ф. «Курс общей лингвистики» (из лекции Сириуса).

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Лекция_08.pptx` (DEEP LEARNING, день 8, 55 слайдов), слайды 1–38: OHE и counts, word2vec (CBOW, skip-gram, тонкости), fastText, GloVe, проверка представлений, семантический сдвиг, наложение языков, контекстные представления (ELMo), представления текстов (SentenceBERT, BERTScore), BPE, BPE в GPT-2, WordPiece, CNN для текстов. Слайды 39–53 — [тема 22](../22_seq2seq_attention/README.md).
- `NLP лекция_Сириус_Мамаев.html`: семиотика, one-hot, хэширование, дистрибутивная семантика, матрица совстречаемости, SVD и теорема Эккарта–Янга, LSA, word2vec с выводом градиента, negative sampling, примеры RusVectōrēs; языковые модели — [тема 24](../24_language_models_gpt/README.md).
- `simple_sentiment_analysis.ipynb`: тональность рецензий IMDB, CountVectorizer/TF-IDF + LR/SVM/SGD, `min_df`, стоп-слова, n-граммы.
- `0000.parquet` — рецензии Кинопоиска ([blinoff/kinopoisk](https://huggingface.co/datasets/blinoff/kinopoisk), 70 МБ, в репозиторий не копируется): данные L1-примеров.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Mikolov et al. [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781) · [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546), 2013.
- Goldberg, Levy. [word2vec Explained: deriving Mikolov et al.'s negative-sampling](https://arxiv.org/abs/1402.3722), 2014.
- Levy, Goldberg. [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization), 2014 · Levy, Goldberg, Dagan. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/), 2015.
- Levy, Goldberg. [Linguistic Regularities in Sparse and Explicit Word Representations](https://aclanthology.org/W14-1618/) (3CosMul), 2014.
- Pennington, Socher, Manning. [GloVe](https://nlp.stanford.edu/pubs/glove.pdf), 2014 · Bojanowski et al. [fastText](https://arxiv.org/abs/1607.04606), 2017.
- Smith et al. [Offline bilingual word vectors, orthogonal transformations](https://arxiv.org/abs/1702.03859), 2017 · Hamilton et al. [Diachronic Word Embeddings](https://www.aclweb.org/anthology/P16-1141.pdf), 2016.
- Halko, Martinsson, Tropp. [Finding structure with randomness](https://arxiv.org/abs/0909.4061), 2011.
- Sennrich et al. [BPE](https://arxiv.org/abs/1508.07909) · Kudo. [Subword Regularization](https://arxiv.org/abs/1804.10959) · Peters et al. [ELMo](https://arxiv.org/abs/1802.05365).

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL, лекция 8: представления текста, seq2seq, внимание · NLP-лекция (Сириус): дистрибутивная семантика, word2vec, LM
- **Семинары:** [Тональность: мешок слов и TF-IDF](../../materials/seminars/21_simple_sentiment_analysis.ipynb) · [Семинар 7: RNN, LSTM, предобработка текста](../../materials/seminars/20_Семинар_7_RNN_предобработка_текста.ipynb) ([Colab](https://colab.research.google.com/drive/1D_XIvuqGltfgfkJ2sDcwh5ivuPFKW6ue))
- **Видео и внешние курсы:** [ Видео: «Word Embedding»](https://disk.yandex.ru/i/PbqlnI7XyNPwrg) · [Курс NLP ШАД (yandexdataschool/nlp_course)](https://github.com/yandexdataschool/nlp_course) · [Шпаргалка CS230 по рекуррентным сетям](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/) · [Stanford CS224n — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)

</details>

---

[← 20. Рекуррентные сети: RNN, LSTM, GRU](../20_rnn/README.md) · [Оглавление курса](../../README.md) · [22. Seq2seq и механизм внимания →](../22_seq2seq_attention/README.md)
