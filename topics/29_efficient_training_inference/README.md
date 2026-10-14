# 29. Эффективное обучение и инференс: точность вычислений, память, квантизация, дистилляция

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Обучение и применение моделей упираются в три ресурса: **вычисления** (FLOPs), **память** (веса, градиенты,
состояния оптимизатора, активации) и **пропускную способность** памяти и сети. Обучение ускоряют **смешанной точностью**
(bf16/fp16 вместо fp32), экономией памяти (**gradient checkpointing**, накопление градиента), быстрыми ядрами
(`torch.compile`, FlashAttention) и **распределённым** обучением (DDP, шардирование FSDP/ZeRO). Готовую модель
сжимают для инференса: **квантизация** (int8/int4 вместо float), **дистилляция** (маленький ученик учится на ответах
большого учителя), **прунинг** (удаление весов). Правило: сначала измерить, где узкое место, потом оптимизировать.

| Термин | Определение |
|---|---|
| FLOPs / FLOP/s | число операций с плавающей точкой / скорость устройства |
| Arithmetic intensity | FLOPs на байт прочитанной памяти; мало — упор в память, много — в вычисления |
| fp16 / bf16 | 16-битные форматы: у fp16 узкий диапазон (нужен loss scaling), у bf16 диапазон как у fp32, точность ниже |
| Mixed precision | вычисления в 16 битах, мастер-веса и накопление в fp32 |
| Gradient checkpointing | не хранить активации, пересчитывать на обратном проходе: память ↓, время ↑ ~30% |
| DDP / FSDP (ZeRO) | копия модели на каждой GPU + all-reduce градиентов / шардирование весов, градиентов, состояний оптимизатора |
| Квантизация | хранение и вычисления в int8/int4: $x\approx s\,(q-z)$ |
| Дистилляция | ученик минимизирует KL до «мягких» распределений учителя с температурой $T$ |
| Прунинг | обнуление малозначимых весов (неструктурный) или целых каналов и голов (структурный) |

**Ключевые формулы**

$$
\text{AI}=\frac{\text{FLOPs}}{\text{байты}},\qquad
q=\operatorname{round}\Big(\frac{x}{s}\Big)+z,\ \ s=\frac{x_{\max}-x_{\min}}{2^b-1},\qquad
\mathcal L_{\mathrm{KD}}=T^2\,\mathrm{KL}\Big(\operatorname{softmax}\tfrac{z_t}{T}\,\Big\|\,\operatorname{softmax}\tfrac{z_s}{T}\Big)
$$

## Конспект

Разделы раскрываются по клику.

**Часть A. Где тратятся время и память**

<details>
<summary><b>1. Как устроена GPU и что ограничивает скорость</b></summary>

GPU — тысячи простых ядер и специализированные **тензорные ядра** для матричных умножений в низкой точности (в разы быстрее
fp32). У A100: ~312 TFLOP/s в bf16 на тензорных ядрах и ~2 ТБ/с пропускной способности HBM-памяти. Отсюда **roofline**:
операция ограничена либо вычислениями, либо памятью — смотря сколько FLOPs приходится на байт данных (arithmetic intensity).

- Большие матричные умножения (линейные слои, свёртки) — compute-bound, хорошо загружают GPU.
- Поэлементные операции (активации, нормализации, сложения, dropout) — memory-bound: читают и пишут тензор ради пары
  операций. Их **слияние** в одно ядро (kernel fusion, `torch.compile`) экономит проходы по памяти.
- Внимание наивно материализует матрицу $T\times T$; **FlashAttention** считает его блоками в быстрой SRAM, не записывая матрицу
  в HBM: та же математика, в разы меньше обращений к памяти.
- Генерация LLM по одному токену — memory-bound: на каждый токен читаются все веса (тема 24) — поэтому важны батчинг и квантизация.

Частые узкие места вне GPU: загрузка и аугментация данных на CPU (`DataLoader(num_workers=..., pin_memory=True)`, предварительная
обработка), маленький батч (GPU простаивает), синхронизации `.item()` и печать в цикле, перенос данных CPU↔GPU.

**Как мерить:** прогрев (первые итерации медленные — компиляция, кэширование), `torch.cuda.synchronize()` перед засечкой
времени (вызовы CUDA асинхронны), несколько повторов, `torch.profiler` для разбивки по операциям.

</details>

<details>
<summary><b>2. Память при обучении</b></summary>

| Что хранится | Объём |
|---|---|
| веса | $N\cdot b$ байт ($b=2$ в bf16, 4 в fp32) |
| градиенты | столько же, сколько весов |
| состояния Adam | 2 тензора размера весов, обычно fp32: $8N$ |
| мастер-копия весов fp32 (при смешанной точности) | $4N$ |
| активации | $\propto$ батч × длина × ширина × число слоёв — часто больше всего остального |

Итого для Adam со смешанной точностью ~16 байт на параметр без активаций (тема 25). Способы уместиться:
- **Gradient accumulation:** считать градиенты на нескольких маленьких батчах и делать шаг оптимизатора раз в $k$ батчей —
  эффективный батч больше без роста памяти.
- **Gradient checkpointing:** хранить активации только на границах блоков, остальные пересчитывать на обратном проходе.
- **Шардирование** (ZeRO/FSDP) — разложить веса, градиенты и состояния по нескольким GPU (часть C).
- Оптимизаторы поменьше: 8-битный Adam, Adafactor, SGD с моментумом (1 состояние вместо 2).
- **PEFT** — обучать маленькую добавку вместо всех весов (LoRA, тема 25).

</details>

<details>
<summary><b>3. Форматы чисел и смешанная точность</b></summary>

| Формат | Биты (знак/экспонента/мантисса) | Диапазон | Шаг около 1 |
|---|---|---|---|
| fp32 | 1 / 8 / 23 | $\sim10^{\pm38}$ | $1.2\cdot10^{-7}$ |
| tf32 (внутри тензорных ядер) | 1 / 8 / 10 | как fp32 | $\sim10^{-3}$ |
| fp16 | 1 / 5 / 10 | $6\cdot10^{-5}$ … $6.5\cdot10^{4}$ | $\sim10^{-3}$ |
| bf16 | 1 / 8 / 7 | как fp32 | $7.8\cdot10^{-3}$ |
| fp8 (E4M3 / E5M2) | 1 / 4 / 3 или 1 / 5 / 2 | узкий | грубый |

Пример кода этой темы: в fp16 число $10^{-8}$ становится нулём, а 70 000 — бесконечностью; в bf16 оба представимы, но
$1+10^{-3}$ округляется до 1.

**Смешанная точность (AMP):** матричные умножения и свёртки — в bf16/fp16 (быстро, тензорные ядра), чувствительные операции
(softmax, нормализации, суммирование лосса) и мастер-веса — в fp32.

```python
scaler = torch.amp.GradScaler("cuda")                      # нужен только для fp16
for x, y in loader:
    with torch.autocast("cuda", dtype=torch.float16):        # для bf16: dtype=torch.bfloat16 и без scaler
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()                            # домножить лосс, чтобы градиенты не обнулились в fp16
    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
```

С **bf16** (A100, H100, современные TPU) масштабирование лосса не нужно — диапазон как у fp32; поэтому сейчас по умолчанию
bf16. fp8 используют для самых больших моделей со специальной поддержкой (Transformer Engine).

</details>

<details>
<summary><b>4. Ускорение кода</b></summary>

- `torch.compile(model)` — трассирует модель, сливает операции, генерирует ядра; ускорение 1.3–2× без изменения кода.
- `F.scaled_dot_product_attention` — автоматически выбирает FlashAttention / memory-efficient ядро.
- `torch.backends.cuda.matmul.allow_tf32 = True` — fp32-матрицы на тензорных ядрах.
- Для свёрточных сетей — `channels_last`, размеры каналов кратные 8.
- Данные: `num_workers`, `pin_memory`, `persistent_workers`, предвычисленные признаки, быстрые форматы (WebDataset, memmap).
- Батч максимально большой, который влезает (с подбором lr: линейное правило или прогрев, тема 15).

</details>

**Часть B. Распределённое обучение**

<details>
<summary><b>5. Параллелизм по данным и по модели</b></summary>

| Подход | Что делится между устройствами | Когда |
|---|---|---|
| **DDP** (data parallel) | данные; на каждой GPU полная копия модели, градиенты усредняются all-reduce | модель влезает в одну GPU |
| **ZeRO-1/2/3, FSDP** | состояния оптимизатора / + градиенты / + веса; собираются по слою при необходимости | модель не влезает, но слой влезает |
| **Tensor parallel** | матрицы внутри слоя (по строкам или столбцам), коммуникация внутри каждого слоя | очень широкие слои, быстрая связь (NVLink) |
| **Pipeline parallel** | слои по устройствам, микробатчи «конвейером» | очень глубокие модели, много узлов |
| **Sequence / context parallel** | длинная последовательность режется по устройствам | контексты в сотни тысяч токенов |

Большие LLM обучают «3D-параллелизмом»: tensor внутри узла, pipeline между узлами, data parallel сверху. Для одного узла с
несколькими GPU обычно достаточно DDP или FSDP. Запуск: `torchrun --nproc_per_node=8 train.py`, обёртка
`DistributedDataParallel(model)` или `FullyShardedDataParallel(model)`, `DistributedSampler` для данных; удобные обёртки —
Hugging Face Accelerate, DeepSpeed, Lightning.

</details>

**Часть C. Сжатие и ускорение инференса**

<details>
<summary><b>6. Квантизация</b></summary>

Хранить и считать в целых числах: $x\approx s\,(q-z)$, $q\in\lbrace -128,\dots,127\rbrace$ для int8.

- **Гранулярность:** одна шкала на тензор (per-tensor) грубее, чем на канал (per-channel) или на группу из 64–128 весов (group-wise,
  стандарт для int4).
- **Динамическая** квантизация: веса заранее в int8, активации квантуются «на лету» — проще всего, хорошо для линейных слоёв и LSTM на CPU.
- **Статическая (PTQ):** шкалы активаций калибруются на небольшой выборке.
- **QAT** (quantization-aware training): обучение с имитацией квантования (fake quant, straight-through градиент) — лучшее качество
  при низкой битности.
- **LLM:** weight-only квантизация в int8/int4 — GPTQ (послойная компенсация ошибки по Гессиану), AWQ (защита «важных» каналов по
  активациям), bitsandbytes NF4 (тема 25), форматы GGUF (тема 26). Проблема — выбросы в активациях (несколько каналов с огромными
  значениями); LLM.int8() держит их в fp16, SmoothQuant переносит масштаб из активаций в веса.

Пример кода этой темы: динамическая int8-квантизация MLP уменьшает размер в 4 раза (1.15 → 0.29 МБ) без потери точности.

</details>

<details>
<summary><b>7. Дистилляция знаний</b></summary>

Hinton et al. (2015): маленький **ученик** учится повторять **мягкие** распределения большого **учителя**. Распределение учителя
с температурой $T>1$ ($\operatorname{softmax}(z/T)$) несёт «тёмное знание»: какие неправильные классы похожи на правильный
(«семёрка похожа на единицу больше, чем на восьмёрку»). Лосс — смесь KL до учителя и обычной кросс-энтропии с метками.

Плюсы: ученик получает гораздо больше информации на пример, чем из одной метки; можно учить на **неразмеченных** данных
(разметку даёт учитель). Пример кода этой темы: ученик с 16 нейронами на 150 размеченных цифрах — 0.88, тот же ученик, обученный на
мягких метках учителя по всему (неразмеченному) train, — 0.96 (учитель — 0.976).

Варианты: дистилляция признаков промежуточных слоёв (FitNets), DistilBERT (на 40% меньше BERT, 97% качества на GLUE, на 60% быстрее), для LLM —
обучение на генерациях большой модели (sequence-level KD) и на её логитах; самодистилляция (DINO, тема 28).

</details>

<details>
<summary><b>8. Прунинг и разреженность</b></summary>

- **Неструктурный** (по модулю веса): обнулить долю самых маленьких весов. Модель часто выдерживает 50–90% нулей, особенно
  с дообучением после прунинга, но ускорение получается только на специальных библиотеках или аппаратуре (полу-структурная
  разреженность 2:4 на GPU Ampere — 2 нуля из каждых 4).
- **Структурный**: удалить целые каналы свёрток, головы внимания, слои — модель реально становится меньше и быстрее на любом
  железе, но теряет больше качества.
- **Гипотеза лотерейного билета** (Frankle & Carbin, 2019): в большой сети есть разреженная подсеть, которая, обученная с тех же
  начальных весов, достигает того же качества.
- Итеративный прунинг: обрезать понемногу и дообучать.

Пример кода этой темы: без дообучения точность держится до 50% обрезанных весов, падает при 80% (0.86) и рушится при 90% (0.33);
с коротким дообучением (с той же маской) даже 90% нулей почти не стоят качества — 0.974 против 0.976 у исходной модели, 95% — 0.912.

</details>

<details>
<summary><b>9. Инференс больших моделей</b></summary>

- **KV-cache** и его память; MQA/GQA уменьшают кэш (тема 24).
- **Непрерывный батчинг** и **PagedAttention** (vLLM): кэш хранится страницами, запросы добавляются в батч по мере освобождения.
- **Спекулятивное декодирование:** маленькая модель предлагает несколько токенов, большая проверяет их за один проход.
- **Квантизация весов** (int8/int4) — модель влезает в меньшую память, генерация быстрее (memory-bound).
- Экспорт и рантаймы: ONNX Runtime, TensorRT, OpenVINO для CPU, llama.cpp для локальных LLM (тема 26).

</details>

<details>
<summary><b>10. Чек-лист оптимизации</b></summary>

1. Профилировать: где время — данные, вычисления, коммуникация, синхронизации?
2. bf16 autocast (или fp16 + GradScaler), `torch.compile`, SDPA/FlashAttention, TF32.
3. Не влезает: gradient checkpointing → накопление градиента → FSDP/ZeRO → PEFT.
4. Несколько GPU: DDP; не влезает — FSDP.
5. Инференс: батчинг, квантизация (int8 почти без потерь, int4 — проверять качество), дистилляция в меньшую модель,
   экспорт в оптимизированный рантайм.
6. Каждое изменение — с замером качества: «быстрее» не должно значить «хуже».

</details>

**Математика и выводы**

<details>
<summary><b>1. Roofline и arithmetic intensity</b></summary>

Пусть устройство даёт $P$ FLOP/s и $B$ байт/с. Время операции $\approx\max\big(\frac{F}{P},\frac{M}{B}\big)$, где $F$ — FLOPs, $M$ —
байты обмена с памятью. Операция compute-bound, если $\frac FM>\frac PB$ (для A100 в bf16: $312\cdot10^{12}/2\cdot10^{12}\approx156$ FLOP/байт).

- Умножение матриц $n\times n$ в bf16: $F=2n^3$, $M\approx3\cdot2n^2$ байт → $\mathrm{AI}\approx n/3$; при $n\gtrsim500$ упираемся в вычисления.
- Поэлементная операция над $n$ числами: $F\approx n$, $M=2\cdot2n$ → $\mathrm{AI}\approx0.25$ — глубоко memory-bound.
- Генерация LLM с батчем 1: на каждый параметр (2 байта) — 2 FLOPs, $\mathrm{AI}\approx1$ → скорость определяется чтением весов;
  батч $b$ поднимает AI примерно в $b$ раз — отсюда выигрыш батчинга и квантизации.

</details>

<details>
<summary><b>2. Почему fp16 требует масштабирования лосса</b></summary>

Наименьшее нормальное fp16 $\approx6.1\cdot10^{-5}$, субнормальное $\approx6\cdot10^{-8}$; градиенты глубоких сетей часто меньше и
обнуляются (underflow). Решение: умножить лосс на $S$ (например, $2^{16}$) — по цепному правилу все градиенты умножатся на $S$ и
окажутся в представимом диапазоне; перед шагом оптимизатора разделить на $S$ в fp32. **Динамический** $S$: если в градиентах
появились inf/NaN — шаг пропускается, $S$ уменьшается вдвое; если $N$ шагов без переполнения — увеличивается. У bf16 диапазон
как у fp32 — underflow не происходит, зато 7 бит мантиссы: суммирование многих малых чисел нужно вести в fp32.

</details>

<details>
<summary><b>3. Ошибка квантизации</b></summary>

Равномерная квантизация с шагом $s$: ошибка округления равномерна на $[-s/2,s/2]$, её дисперсия $s^2/12$. Для $b$ бит и диапазона
$[x_{\min},x_{\max}]$: $s=\frac{x_{\max}-x_{\min}}{2^b-1}$ → каждый убранный бит удваивает шаг и учетверяет MSE. Отношение
сигнал/шум $\approx6.02b$ дБ. Отсюда:

- Выброс увеличивает диапазон и шаг для **всех** значений тензора → per-channel и group-wise шкалы, обрезка (clipping) по перцентилю.
- Для линейного слоя $y=Wx$ ошибка в $y$ складывается из $\sum_j\delta W_{ij}x_j$: важны веса, умножаемые на большие активации —
  идея AWQ (масштабировать такие каналы перед квантизацией).
- GPTQ квантует веса по одному столбцу и корректирует ещё не квантованные столбцы, минимизируя $\lVert WX-\hat WX\rVert^2$ с
  использованием обратного Гессиана $H=XX^\top$ (как в Optimal Brain Surgeon).

</details>

<details>
<summary><b>4. Дистилляция: зачем $T^2$</b></summary>

$p_i=\operatorname{softmax}(z_i/T)$. Градиент кросс-энтропии до мягких целей $q_i$ по логиту ученика:
$\frac{\partial\mathcal L}{\partial z_i}=\frac1T(p_i-q_i)$. При больших $T$ и логитах с нулевым средним
$p_i-q_i\approx\frac{z_i-v_i}{KT}$ ($v$ — логиты учителя, $K$ — число классов), и градиент $\approx\frac{1}{KT^2}(z_i-v_i)$: дистилляция при
высокой температуре — это подгонка **логитов** по MSE, а масштаб градиента убывает как $1/T^2$. Множитель $T^2$ возвращает его к
масштабу обычного лосса, чтобы вес $\alpha$ между KD и кросс-энтропией не зависел от $T$.

</details>

<details>
<summary><b>5. Стоимость all-reduce</b></summary>

DDP после каждого шага усредняет градиенты размера $M$ байт между $p$ устройствами. Кольцевой all-reduce: reduce-scatter + all-gather,
каждое устройство передаёт $2\frac{p-1}{p}M$ байт — почти не зависит от $p$. Время $\approx\frac{2M}{\text{bandwidth}}$. Для модели на 1 млрд
параметров в bf16 ($M=2$ ГБ) по NVLink 300 ГБ/с — ~13 мс, по сети 25 ГБ/с — ~160 мс на шаг. Поэтому DDP перекрывает коммуникацию с
обратным проходом (градиенты отправляются «корзинами» по мере готовности) и выигрывает от накопления градиента (синхронизация раз в
$k$ батчей). ZeRO-3/FSDP добавляет all-gather весов на прямом и обратном проходе — ~1.5× трафика DDP ради деления памяти на $p$.

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему bf16 вытеснил fp16?</summary>

диапазон как у fp32: не нужен loss scaling, нет переполнений; меньшая точность мантиссы компенсируется fp32-накоплением.

</details>

<details>
<summary>Обучение медленное, GPU загружена на 30% — что проверить?</summary>

загрузку данных (num_workers, предобработка), маленький батч, синхронизации `.item()`, перенос CPU↔GPU.

</details>

<details>
<summary>Gradient accumulation и большой батч — одно и то же?</summary>

по градиенту да (с усреднением), но BatchNorm видит маленькие батчи, а время на шаг больше.

</details>

<details>
<summary>Чем FSDP отличается от DDP?</summary>

DDP хранит полную копию модели и состояний на каждой GPU, FSDP шардирует их и собирает слой при необходимости.

</details>

<details>
<summary>Почему неструктурный прунинг часто не ускоряет модель?</summary>

плотные матричные ядра не пропускают нули; нужна структурная разреженность или специальная поддержка (2:4).

</details>

<details>
<summary>Зачем в дистилляции температура?</summary>

при $T=1$ распределение учителя почти one-hot, информации о похожести классов мало; $T>1$ открывает «тёмное знание».

</details>

<details>
<summary>Квантизация int4 весов LLM — что теряем?</summary>

немного качества (обычно 1–3% на бенчмарках, больше на маленьких моделях и задачах на точность); веса в 4 раза меньше, генерация быстрее.

</details>

<details>
<summary>Почему генерация LLM упирается в память, а не в вычисления?</summary>

на каждый токен читаются все веса, а FLOPs на байт ~1; батчинг и квантизация это меняют.

</details>

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# Форматы чисел: диапазон и точность fp32 / fp16 / bf16
import io, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt)[6:]:9s} max={fi.max:.1e} min_normal={fi.tiny:.1e} eps={fi.eps:.1e} | "
          f"1e-8 → {torch.tensor(1e-8, dtype=dt).item():.1e}, 70000 → {torch.tensor(7e4, dtype=dt).item():.0f}, "
          f"1 + 1e-3 → {(torch.tensor(1.0, dtype=dt) + 1e-3).item():.4f}")
```

Результат запуска:

```text
float32   max=3.4e+38 min_normal=1.2e-38 eps=1.2e-07 | 1e-8 → 1.0e-08, 70000 → 70000, 1 + 1e-3 → 1.0010
float16   max=6.6e+04 min_normal=6.1e-05 eps=9.8e-04 | 1e-8 → 0.0e+00, 70000 → inf, 1 + 1e-3 → 1.0010
bfloat16  max=3.4e+38 min_normal=1.2e-38 eps=7.8e-03 | 1e-8 → 1.0e-08, 70000 → 70144, 1 + 1e-3 → 1.0000
```

```python
# Дистилляция: маленький ученик на 150 размеченных примерах против ученика, обученного на мягких метках учителя
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d = load_digits(); X = torch.tensor(d.data / 16, dtype=torch.float32); y = torch.tensor(d.target)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
mlp = lambda h: nn.Sequential(nn.Linear(64, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10))
def acc(m):
    with torch.no_grad(): return (m(Xte).argmax(1) == yte).float().mean().item()
def fit(m, X_, y_=None, teacher=None, T=4.0, epochs=300):
    opt = torch.optim.Adam(m.parameters(), 3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        logits = m(X_)
        if teacher is None: loss = F.cross_entropy(logits, y_)
        else:
            with torch.no_grad(): soft = F.softmax(teacher(X_) / T, 1)
            loss = F.kl_div(F.log_softmax(logits / T, 1), soft, reduction="batchmean") * T * T
        opt.zero_grad(); loss.backward(); opt.step()
    return m
torch.manual_seed(0); teacher = fit(mlp(512), Xtr, ytr, epochs=200)
labeled = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(0))[:150]
torch.manual_seed(1); s1 = fit(mlp(16), Xtr[labeled], ytr[labeled])
torch.manual_seed(1); s2 = fit(mlp(16), Xtr, teacher=teacher)               # метки не нужны — их даёт учитель
n = lambda m: sum(p.numel() for p in m.parameters())
print(f"учитель ({n(teacher):,} параметров): {acc(teacher):.3f}")
print(f"ученик ({n(s1):,}), 150 размеченных:      {acc(s1):.3f}")
print(f"ученик ({n(s2):,}), мягкие метки учителя: {acc(s2):.3f}")
```

Результат запуска:

```text
учитель (301,066 параметров): 0.976
ученик (1,482), 150 размеченных:      0.878
ученик (1,482), мягкие метки учителя: 0.960
```

```python
# Прунинг по модулю весов (без дообучения и с дообучением) и динамическая int8-квантизация
import torch.nn.utils.prune as prune
def pruned(model, amount, finetune=False):
    m = copy.deepcopy(model)
    for layer in m:
        if isinstance(layer, nn.Linear): prune.l1_unstructured(layer, "weight", amount)   # маска остаётся при дообучении
    if finetune: fit(m, Xtr, ytr, epochs=50)
    for layer in m:
        if isinstance(layer, nn.Linear): prune.remove(layer, "weight")
    return m
for amount in (0.5, 0.8, 0.9, 0.95):
    torch.manual_seed(0)
    print(f"прунинг {amount:.0%}: без дообучения {acc(pruned(teacher, amount)):.3f}, с дообучением {acc(pruned(teacher, amount, True)):.3f}")

def size_mb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return buf.tell() / 2**20
def latency_ms(m, reps=20):
    xb = Xte.repeat(50, 1)
    with torch.no_grad():
        m(xb); t = time.perf_counter()
        for _ in range(reps): m(xb)
    return (time.perf_counter() - t) / reps * 1000
q = torch.ao.quantization.quantize_dynamic(teacher, {nn.Linear}, dtype=torch.qint8)
print(f"fp32: {size_mb(teacher):.2f} МБ, точность {acc(teacher):.3f}, {latency_ms(teacher):.0f} мс на {len(Xte) * 50} объектов")
print(f"int8: {size_mb(q):.2f} МБ, точность {acc(q):.3f}, {latency_ms(q):.0f} мс")
```

Результат запуска:

```text
прунинг 50%: без дообучения 0.969, с дообучением 0.975
прунинг 80%: без дообучения 0.857, с дообучением 0.978
прунинг 90%: без дообучения 0.332, с дообучением 0.974
прунинг 95%: без дообучения 0.102, с дообучением 0.912
fp32: 1.15 МБ, точность 0.976, 161 мс на 35950 объектов
int8: 0.29 МБ, точность 0.978, 135 мс
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Курсы: [DL2 HSE, неделя 2: Essentials of GPU, Deep Learning Bottlenecks, and Benchmarking Basics](https://github.com/thecrazymage/DL2_HSE/tree/main/week_02) · [курс NLP ШАД, неделя 8: Efficiency](https://github.com/yandexdataschool/nlp_course) · [Stanford CS336](https://cs336.stanford.edu/) (GPU, ядра, параллелизм) · Учебник ML Яндекса: [Дистилляция знаний](https://education.yandex.ru/handbook/ml/article/distillyaciya-znanij)
- Руководства: [Hugging Face — Efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one) · [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) · [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) · [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) · [Lilian Weng — Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) · [The Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- Статьи: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [ZeRO](https://arxiv.org/abs/1910.02054) · [Megatron-LM](https://arxiv.org/abs/1909.08053) · [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) · [DistilBERT](https://arxiv.org/abs/1910.01108) · [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) · [Quantization survey (Gholami et al.)](https://arxiv.org/abs/2103.13630) · [LLM.int8()](https://arxiv.org/abs/2208.07339) · [SmoothQuant](https://arxiv.org/abs/2211.10438) · [GPTQ](https://arxiv.org/abs/2210.17323) · [AWQ](https://arxiv.org/abs/2306.00978) · [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)

</details>

<details>
<summary><b>Источники курса</b></summary>

- Тема дополнительная: в лекциях курса её нет. Материал по курсам DL2 HSE (неделя 2), NLP ШАД (неделя 8), CS336 и оригинальным статьям; основа — темы 15, 24, 25.

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Внешние курсы:** [DL2 HSE, неделя 2](https://github.com/thecrazymage/DL2_HSE/tree/main/week_02) · [Курс NLP ШАД (yandexdataschool/nlp_course)](https://github.com/yandexdataschool/nlp_course) · [Stanford CS336](https://cs336.stanford.edu/)
- **Связанные темы:** [15. Обучение нейросетей](../15_nn_training/README.md) · [24. Языковые модели](../24_language_models_gpt/README.md) · [25. Дообучение LLM](../25_llm_finetuning_peft/README.md) · [26. RAG и агенты](../26_llm_rag_agents/README.md)

</details>

---

[← 28. Vision Transformer, self-supervised и мультимодальные модели](../28_vit_ssl_multimodal/README.md) · [Оглавление курса](../../README.md) · [30. Интерпретируемость моделей: важность признаков, PDP, SHAP, LIME →](../30_interpretability/README.md)
