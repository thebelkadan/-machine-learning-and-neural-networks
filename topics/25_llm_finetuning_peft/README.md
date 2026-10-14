# 25. Дообучение LLM: SFT, PEFT (LoRA, QLoRA, prompt tuning), выравнивание (RLHF, DPO)

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 24. Языковые модели и GPT](../24_language_models_gpt/README.md) · [26. Промптинг, RAG, агенты →](../26_llm_rag_agents/README.md)

---

## Часть A. Из языковой модели — ассистент

### 1. Три стадии

| Стадия | Данные | Цель | Результат |
|---|---|---|---|
| Предобучение | триллионы токенов текста и кода | следующий токен | **base**-модель: знания, язык, продолжение текста |
| SFT / instruction tuning | $10^4$–$10^6$ пар «инструкция → ответ», диалоги | следующий токен **ответа** | **instruct**-модель: следует просьбам, формат диалога |
| Выравнивание по предпочтениям | пары ответов «лучше / хуже» от людей или модели | RLHF (RM + PPO), DPO, … | полезность, безопасность, честность, стиль |

**Base vs Instruct (семинар 10, SmolLM2-135M):** на вопрос «What is the purpose of the list C in the code below?» base-модель продолжает
текст как статью («The code runs in less than 1 second…» или повторяет код), instruct-модель с chat template отвечает по существу (хотя
маленькая модель и ошибается в деталях). Знания у них одни и те же — разное **поведение**.

### 2. Chat template

Диалог — список сообщений с ролями `system` (инструкции, персона), `user`, `assistant` (а также `tool`). Каждая модель обучалась на своём
формате со спецтокенами, и использовать надо **именно его**:

```python
messages = [{"role": "system", "content": "Ты кратко отвечаешь на вопросы."},
            {"role": "user", "content": "Сколько будет 2 + 2?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# SmolLM2 / Qwen (ChatML):
# <|im_start|>system\nТы кратко отвечаешь на вопросы.<|im_end|>\n<|im_start|>user\nСколько будет 2 + 2?<|im_end|>\n<|im_start|>assistant\n
```

`add_generation_prompt=True` дописывает начало реплики ассистента — модель продолжит именно ответ. Чужой или самодельный формат
ощутимо ухудшает качество. `pipeline("text-generation")` принимает список сообщений и применяет шаблон сам.

### 3. SFT: что именно оптимизируем

Тот же лосс LM, но **только на токенах ответа**: метки промпта (system + user) = `-100`.

```python
def build(example):                                   # completion-only loss вручную
    p_ids = tok.apply_chat_template(example["prompt"], add_generation_prompt=True)
    c_ids = tok(example["completion"] + tok.eos_token, add_special_tokens=False).input_ids
    return {"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids}
```

В TRL: `SFTTrainer` с conversational-датасетом `{"prompt": [...], "completion": [...]}` и `SFTConfig(completion_only_loss=True)`
(раньше — `DataCollatorForCompletionOnlyLM(response_template=...)`, который ищет в метках начало ответа и зануляет всё до него; в семинаре он
приведён как справочный). Не забыть `eos` в конце ответа — иначе модель не научится останавливаться.

**Семинар 10: SFT на анекдотах.** Датасет `igorktech/anekdots_dialogs` (100 834 анекдота) → 2000 самых залайканных (`upvote`) → формат
«system: ты генератор анекдотов; user: сгенерируй смешной анекдот; assistant: <анекдот>»; SmolLM2-135M-Instruct + LoRA (своя
реализация), `SFTConfig`: батч 1 × накопление 32, lr 1e-4, linear + warm-up 10%, `max_length=1024`, bf16/fp16. До обучения модель
на русском выдаёт бессвязный текст — 135M-модель почти не видела русского; SFT на 2000 примеров учит формату, но не языку.

**Главный практический урок** (L1-пример): LoRA-SFT ruGPT-3 small на 120 шагов учит отвечать JSON-ом `{"ответ": …}` — валидный формат
0–4% → **100%**, а правильных сумм двузначных чисел так и остаётся **≈0%**. SFT меняет **формат и поведение** быстро и дёшево; новые
**навыки и знания** так не добавить (для знаний — RAG, [тема 26](../26_llm_rag_agents/README.md); для навыков — гораздо больше данных
или RL).

### 4. Данные для SFT

- Качество важнее количества: LIMA (2023) — 1000 тщательно отобранных примеров дают сильного ассистента («superficial alignment
  hypothesis»: знания из предобучения, SFT учит формату взаимодействия).
- Разнообразие задач и форматов; без утечек тестовых бенчмарков; дедупликация.
- Синтетические данные от сильной модели (Self-Instruct, Alpaca, Evol-Instruct) — дёшево, но наследуют ошибки и лицензионные ограничения.
- Смешивать с общими данными, чтобы не забывать умения (**catastrophic forgetting**).

---

## Часть B. Parameter-Efficient Fine-Tuning

### 5. Зачем PEFT

Полное дообучение с Adam требует ~16 байт на параметр (bf16 веса + bf16 градиенты + fp32 мастер-копия + два момента Adam в fp32): для
7B-модели — ~104 ГБ без активаций (L1-калькулятор). PEFT замораживает модель и учит **малую** добавку:

- **экономия памяти** — градиенты и состояния оптимизатора только для 0.1–2% параметров;
- **быстрее** обучение и **маленькие чекпойнты** (мегабайты вместо гигабайт);
- **модульность** — одна база, много адаптеров под задачи/клиентов, переключение на лету;
- меньше забывания — база не меняется.

### 6. LoRA (Hu et al., 2021)

Гипотеза: обновление весов при дообучении имеет **низкий внутренний ранг**. Вместо $\Delta W\in\mathbb R^{d\times k}$ учим

$$
h = W_0x + \Delta Wx = W_0x + \frac{\alpha}{r}BAx,\qquad B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},\ r\ll\min(d, k).
$$

- **Инициализация:** $A$ — случайная (Kaiming/гаусс), $B = 0$ ⇒ на старте $\Delta W = 0$, модель не испорчена, а градиент по $B$ ненулевой.
- **Масштаб $\alpha/r$:** при смене $r$ не надо перенастраивать lr; обычно $\alpha = 2r$ (семинар: $r = 16$, $\alpha = 32$).
- **Параметры:** $r(d + k)$ вместо $dk$; для $4096\times4096$ и $r = 8$ — 65 536 вместо 16.8 млн (0.39%).
- **Слияние:** после обучения $W = W_0 + \frac{\alpha}{r}BA$ — **нулевая задержка** на инференсе; `merge`/`unmerge` позволяют переключаться.
- **Куда:** оригинал — $W_Q, W_V$; на практике лучше **все** линейные слои (q, k, v, o, gate, up, down) — это важнее, чем большой $r$.
- **Гиперпараметры:** $r$ 8–64, $\alpha = r$…$2r$, dropout 0.05–0.1, lr **выше**, чем при полном дообучении (1e-4…2e-4 против 1e-5…2e-5).

```python
class LoRALayer(nn.Module):                                  # семинар 10 (сокращённо)
    def __init__(self, base_layer: nn.Linear, r, lora_alpha, dropout_p=0.0):
        super().__init__()
        self.base_layer, self.scale = base_layer, lora_alpha / r
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout_p)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.base_layer(x) + self.scale * self.lora_B(self.lora_A(self.lora_dropout(x)))
    @torch.no_grad()
    def merge(self):
        self.base_layer.weight.data += self.scale * self.lora_B.weight @ self.lora_A.weight

# обход модели: заморозить всё, заменить nn.Linear с именами из target_modules на LoRALayer
for p in model.parameters(): p.requires_grad = False
for name, module in list(model.named_modules()):
    if isinstance(module, nn.Linear) and name.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], LoRALayer(module, r=16, lora_alpha=32, dropout_p=0.05))
# SmolLM2-135M: 1 843 200 обучаемых из 136 358 208 (1.35%)
```

То же через библиотеку 🤗 PEFT:

```python
from peft import LoraConfig, get_peft_model, TaskType
cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1,
                 target_modules=["c_attn"], bias="none")           # имена модулей зависят от архитектуры (GPT-2: c_attn, c_proj, c_fc)
model = get_peft_model(model, cfg); model.print_trainable_parameters()
# GPT-2: trainable 294 912 из 124 734 720 (0.24%); с c_attn + c_proj + c_fc и r=16 — 1.86%
model.save_pretrained("adapter/")                                   # сохраняется только адаптер (МБ)
merged = model.merge_and_unload()                                   # влить в веса
```

Варианты: **DoRA** (раздельно учит направление и норму весов), **rsLoRA** (масштаб $\alpha/\sqrt r$), **LoRA+** (разный lr для $A$ и $B$),
**PiSSA** (инициализация $A, B$ из главных сингулярных векторов $W_0$).

### 7. AdaLoRA (Zhang et al., 2023)

Разные матрицы нуждаются в разном ранге. AdaLoRA параметризует $\Delta W = P\Lambda Q$ в форме, похожей на SVD (с регуляризацией
ортогональности $P$, $Q$), оценивает важность каждой «сингулярной тройки» по чувствительности лосса и **обрезает** неважные, распределяя
бюджет ранга: от `init_r=12` к `target_r=8` по расписанию `tinit` → `tfinal` с шагом `deltaT`. Дольше обучение, чуть лучше качество при том же
бюджете. GPT-2 в семинаре: 0.97% обучаемых.

### 8. QLoRA (Dettmers et al., 2023)

Базовая модель хранится в **4 битах**, LoRA-адаптеры учатся в bf16 поверх неё. Компоненты:
- **NF4** (4-bit NormalFloat): 16 уровней = квантили нормального распределения (веса предобученных сетей ≈ нормальны) — информационно
  оптимально для таких весов; квантование **блоками** по 64 со своей шкалой (absmax).
- **Double quantization:** сами шкалы квантуются в 8 бит — ещё ~0.37 бита на параметр экономии.
- **Paged optimizers:** состояния оптимизатора выгружаются в CPU-память при пиках (unified memory).
- Вычисления: вес деквантуется в bf16 «на лету» для умножения; градиент течёт сквозь замороженную 4-битную базу к LoRA.

Результат: 65B-модель дообучается на одной 48 ГБ GPU, 7B — на ~6–10 ГБ (L1-калькулятор: ~4 ГБ без активаций), качество близко к 16-битному
дообучению.

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)              # LayerNorm в fp32, gradient checkpointing, входные градиенты
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
```

### 9. Prompt tuning, prefix tuning, адаптеры

| Метод | Что учим | Доля параметров (GPT-2, семинар) |
|---|---|---|
| **Prompt tuning** (Lester et al., 2021) | $m$ «виртуальных токенов» — эмбеддинги, приписываемые ко входу; можно инициализировать текстом («Classify the following text:») | 20 токенов: 15 360 (0.012%) |
| **Prefix tuning** (Li & Liang, 2021) | обучаемые векторы-префиксы $K$, $V$ **в каждом слое** (через MLP-репараметризацию при обучении) | 10 токенов: 184 320 (0.15%) |
| **P-tuning v2** | глубокие промпты для NLU-задач | — |
| **Адаптеры** (Houlsby et al., 2019) | маленький bottleneck-MLP $d\to m\to d$ с residual после подблоков | 0.5–5%; добавляет задержку |
| **(IA)³** | поэлементные масштабы для $K$, $V$ и FFN | ~0.01% |
| **BitFit** | только bias | ~0.1% |

Prompt tuning хорошо работает лишь на больших моделях (≥ 10B — догоняет полное дообучение); на малых проигрывает LoRA.

---

## Часть C. Выравнивание по предпочтениям

### 10. RLHF (InstructGPT, Ouyang et al., 2022)

1. **SFT** на демонстрациях людей (~13k промптов).
2. **Reward-модель:** для промпта генерируются несколько ответов, люди ранжируют; RM (та же LM с головой-скаляром) учится по модели
   Брэдли–Терри: $\mathcal L = -\log\sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$ (~33k промптов сравнений).
3. **PPO:** политика $\pi_\theta$ (инициализирована SFT) максимизирует $\mathbb E[r_\phi(x, y)] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\text{SFT}})$ (~31k промптов).
   KL-штраф не даёт уйти от разумного языка и «взломать» RM (reward hacking).

Результат: ответы 1.3B InstructGPT люди предпочитали ответам 175B GPT-3; меньше выдумок и токсичности; небольшая плата на бенчмарках
(«alignment tax»), компенсируемая подмешиванием LM-лосса. Минусы: четыре модели в памяти (политика, референс, RM, critic), нестабильный PPO.

### 11. DPO и другие методы без RL

**DPO** (Rafailov et al., 2023): оптимальная политика задачи «reward − β·KL» выражается через reward в замкнутой форме, и обратно —
reward через политику. Подстановка в Брэдли–Терри даёт **классификационный** лосс на парах (вывод — [deep §4](deep.md#4-dpo-из-rlhf)):

$$
\mathcal L_{\mathrm{DPO}} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big).
$$

Нет RM, нет сэмплирования, две модели (политика и замороженный референс); обучается как SFT (`DPOTrainer` в TRL). Ограничения: офлайн
(данные не от текущей политики), может переоптимизироваться. Родственники: **IPO**, **KTO** (не пары, а отдельные «хорошо/плохо»),
**ORPO** (SFT + предпочтения одним лоссом без референса), **SimPO**. **RLAIF / Constitutional AI** — предпочтения размечает модель по
набору принципов. **GRPO** (DeepSeek) — PPO без critic: преимущество считается относительно группы ответов на тот же промпт; основа
обучения «рассуждающих» моделей с проверяемыми наградами (математика, код).

### 12. Как выбрать подход

| Ситуация | Что делать |
|---|---|
| Нужен формат / стиль / тон | промпт и few-shot ([тема 26](../26_llm_rag_agents/README.md)) → если мало, LoRA-SFT на сотнях–тысячах примеров |
| Нужны знания из своих документов | **RAG**, не дообучение |
| Узкая задача классификации/извлечения | маленькая encoder-модель (BERT) + полное дообучение или LoRA |
| Мало памяти GPU | QLoRA |
| Много клиентов/задач на одной базе | отдельные LoRA-адаптеры, переключение без перезагрузки базы |
| Нужно «как больше нравится людям» | DPO на парах предпочтений |

### Чек-лист

- [ ] Правильный chat template модели; `eos` в конце ответа; лосс только на ответе.
- [ ] LoRA: все линейные слои, $r$ 8–64, $\alpha\approx2r$, lr ~1e-4…2e-4, warm-up, малое число эпох (1–3).
- [ ] Валидация на отложенных промптах + ручной просмотр; следить за забыванием общих навыков.
- [ ] QLoRA: `nf4`, double quant, compute dtype bf16, `prepare_model_for_kbit_training`, gradient checkpointing.
- [ ] Сохранять адаптер отдельно; вливать в веса для инференса.
- [ ] Токены (HF, wandb) — через секреты, не в коде.

---

## Что почитать / посмотреть

- 📘 [yandexdataschool/nlp_course](https://github.com/yandexdataschool/nlp_course): week05 — LLM, week06 — Prompting & In-Context Learning, week07 — Fine-tuning (PEFT & RLHF), week08 — Efficiency
- 📘 [DL2 HSE, неделя 4: RLHF and LLM Agents](https://github.com/thecrazymage/DL2_HSE/tree/main/week_04) · неделя 5: агентное RL · [CS336](https://cs336.stanford.edu/) (alignment, SFT, RL)
- 📄 Документация: [🤗 PEFT](https://huggingface.co/docs/peft) · [TRL (SFTTrainer, DPOTrainer, GRPOTrainer)](https://huggingface.co/docs/trl) · [Chat templates](https://huggingface.co/docs/transformers/chat_templating) · [bitsandbytes](https://huggingface.co/docs/bitsandbytes) · [исходники LoRA в PEFT](https://github.com/huggingface/peft/tree/main/src/peft/tuners/lora)
- 📄 Блоги: [HF — Illustrating RLHF](https://huggingface.co/blog/rlhf) · [Lilian Weng — Reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) · [S. Raschka — Practical tips for LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) · [HF — 4-bit QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- 📄 Статьи: [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [AdaLoRA](https://arxiv.org/abs/2303.10512) · [DoRA](https://arxiv.org/abs/2402.09353) · [rsLoRA](https://arxiv.org/abs/2312.03732) · [Prompt Tuning](https://arxiv.org/abs/2104.08691) · [Prefix-Tuning](https://arxiv.org/abs/2101.00190) · [Adapters (Houlsby)](https://arxiv.org/abs/1902.00751) · [(IA)³](https://arxiv.org/abs/2205.05638) · [BitFit](https://arxiv.org/abs/2106.10199) · [Intrinsic Dimensionality](https://arxiv.org/abs/2012.13255) · [FLAN](https://arxiv.org/abs/2109.01652) · [LIMA](https://arxiv.org/abs/2305.11206) · [Self-Instruct](https://arxiv.org/abs/2212.10560) · [InstructGPT](https://arxiv.org/abs/2203.02155) · [PPO](https://arxiv.org/abs/1707.06347) · [DPO](https://arxiv.org/abs/2305.18290) · [KTO](https://arxiv.org/abs/2402.01306) · [ORPO](https://arxiv.org/abs/2403.07691) · [Constitutional AI](https://arxiv.org/abs/2212.08073) · [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) · [SmolLM2](https://arxiv.org/abs/2502.02737)

## Источники курса

- Семинар 10 «Transformer → PEFT» ([Colab](https://colab.research.google.com/drive/1yQ4b8tCat38RpLbeQIIIMqYpz4QIpoRE)), вторая часть: SmolLM2-135M base vs instruct, chat template, LoRA с нуля (`LoRALayer`, `LoRAModel`, merge/unmerge), completion-only collator, SFT на анекдотах (TRL), библиотека PEFT: LoRA, LoRA на нескольких модулях, AdaLoRA, QLoRA, обучение LoRA на WikiText-2, Prompt Tuning, Prefix Tuning, задания.
- `Лекция 10.pptx`, слайды 63–65: GPT-3.5 и RLHF, эмерджентные способности.


<!-- materials:start -->
**Файлы курса в репозитории:**

- DL, лекция 10: языковые модели, GPT (лекция курса, файл в репозиторий не включён)
- [Семинар 10: от трансформера к PEFT и SFT](../../materials/seminars/23-25_Семинар_10_Transformer_PEFT_SFT.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1yQ4b8tCat38RpLbeQIIIMqYpz4QIpoRE)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [26. Промптинг, RAG, агенты →](../26_llm_rag_agents/README.md)
