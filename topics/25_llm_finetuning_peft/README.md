# 25. Дообучение LLM: SFT, PEFT (LoRA, QLoRA), выравнивание (RLHF, DPO)

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Предобученная LM умеет продолжать текст, но не «слушаться». Путь к ассистенту: **предобучение** (следующий токен на
триллионах токенов) → **SFT / instruction tuning** (пары «инструкция → ответ» в формате **chat template**, лосс только на ответе) →
**выравнивание по предпочтениям** (RLHF: reward-модель + PPO; или DPO без RL). Полное дообучение дорого по памяти, поэтому
**PEFT**: замораживаем модель и учим малую добавку — **LoRA** ($W+\frac{\alpha}{r}BA$, ранг $r\ll d$, $B=0$ на старте),
**QLoRA** (базовая модель в 4 бит NF4 + LoRA), prompt/prefix tuning, адаптеры. Главное понимать: **SFT быстро учит формату и
стилю, но почти не добавляет знаний и навыков**.

| Термин | Определение |
|---|---|
| Base / Instruct | предобученная LM (продолжает текст) / дообученная на инструкциях (отвечает) |
| Chat template | как роли `system` / `user` / `assistant` превращаются в токены (свои спецтокены у каждой модели) |
| Completion-only loss | метки `-100` на промпте: учим только ответ |
| LoRA | $h=W_0x+\frac{\alpha}{r}BAx$, $A\in\mathbb R^{r\times k}$, $B\in\mathbb R^{d\times r}$; после обучения можно влить: $W=W_0+\frac{\alpha}{r}BA$ |
| QLoRA | веса базы в 4 бит (NF4, double quantization), вычисления в bf16, градиенты только в LoRA; paged optimizers |
| Prompt / prefix tuning | обучаемые «виртуальные токены» на входе / векторы префикса K, V в каждом слое |
| AdaLoRA | распределяет ранг между матрицами по важности (SVD-параметризация, обрезка сингулярных чисел) |
| RLHF | SFT → reward-модель на парах «лучше/хуже» (Брэдли–Терри) → PPO с штрафом KL к SFT-модели |
| DPO | оптимизирует предпочтения напрямую лоссом-логрегрессией на разности лог-отношений, без reward-модели и RL |

$$
h=W_0x+\frac{\alpha}{r}BAx,\qquad
\mathcal L_{\text{RM}}=-\log\sigma\big(r(x,y_w)-r(x,y_l)\big),\qquad
\mathcal L_{\text{DPO}}=-\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)
$$

## Конспект

Разделы раскрываются по клику.

**Часть A. Из языковой модели — ассистент**

<details>
<summary><b>1. Три стадии</b></summary>

| Стадия | Данные | Цель | Результат |
|---|---|---|---|
| Предобучение | триллионы токенов текста и кода | следующий токен | **base**-модель: знания, язык, продолжение текста |
| SFT / instruction tuning | $10^4$–$10^6$ пар «инструкция → ответ», диалоги | следующий токен **ответа** | **instruct**-модель: следует просьбам, формат диалога |
| Выравнивание по предпочтениям | пары ответов «лучше / хуже» от людей или модели | RLHF (RM + PPO), DPO, … | полезность, безопасность, честность, стиль |

**Base vs Instruct (семинар 10, SmolLM2-135M):** на вопрос «What is the purpose of the list C in the code below?» base-модель продолжает
текст как статью («The code runs in less than 1 second…» или повторяет код), instruct-модель с chat template отвечает по существу (хотя
маленькая модель и ошибается в деталях). Знания у них одни и те же — разное **поведение**.

</details>

<details>
<summary><b>2. Chat template</b></summary>

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

</details>

<details>
<summary><b>3. SFT: что именно оптимизируем</b></summary>

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

</details>

<details>
<summary><b>4. Данные для SFT</b></summary>

- Качество важнее количества: LIMA (2023) — 1000 тщательно отобранных примеров дают сильного ассистента («superficial alignment
  hypothesis»: знания из предобучения, SFT учит формату взаимодействия).
- Разнообразие задач и форматов; без утечек тестовых бенчмарков; дедупликация.
- Синтетические данные от сильной модели (Self-Instruct, Alpaca, Evol-Instruct) — дёшево, но наследуют ошибки и лицензионные ограничения.
- Смешивать с общими данными, чтобы не забывать умения (**catastrophic forgetting**).

</details>

**Часть B. Parameter-Efficient Fine-Tuning**

<details>
<summary><b>5. Зачем PEFT</b></summary>

Полное дообучение с Adam требует ~16 байт на параметр (bf16 веса + bf16 градиенты + fp32 мастер-копия + два момента Adam в fp32): для
7B-модели — ~104 ГБ без активаций (L1-калькулятор). PEFT замораживает модель и учит **малую** добавку:

- **экономия памяти** — градиенты и состояния оптимизатора только для 0.1–2% параметров;
- **быстрее** обучение и **маленькие чекпойнты** (мегабайты вместо гигабайт);
- **модульность** — одна база, много адаптеров под задачи/клиентов, переключение на лету;
- меньше забывания — база не меняется.

</details>

<details>
<summary><b>6. LoRA (Hu et al., 2021)</b></summary>

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

То же через библиотеку Hugging Face PEFT:

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

</details>

<details>
<summary><b>7. AdaLoRA (Zhang et al., 2023)</b></summary>

Разные матрицы нуждаются в разном ранге. AdaLoRA параметризует $\Delta W = P\Lambda Q$ в форме, похожей на SVD (с регуляризацией
ортогональности $P$, $Q$), оценивает важность каждой «сингулярной тройки» по чувствительности лосса и **обрезает** неважные, распределяя
бюджет ранга: от `init_r=12` к `target_r=8` по расписанию `tinit` → `tfinal` с шагом `deltaT`. Дольше обучение, чуть лучше качество при том же
бюджете. GPT-2 в семинаре: 0.97% обучаемых.

</details>

<details>
<summary><b>8. QLoRA (Dettmers et al., 2023)</b></summary>

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

</details>

<details>
<summary><b>9. Prompt tuning, prefix tuning, адаптеры</b></summary>

| Метод | Что учим | Доля параметров (GPT-2, семинар) |
|---|---|---|
| **Prompt tuning** (Lester et al., 2021) | $m$ «виртуальных токенов» — эмбеддинги, приписываемые ко входу; можно инициализировать текстом («Classify the following text:») | 20 токенов: 15 360 (0.012%) |
| **Prefix tuning** (Li & Liang, 2021) | обучаемые векторы-префиксы $K$, $V$ **в каждом слое** (через MLP-репараметризацию при обучении) | 10 токенов: 184 320 (0.15%) |
| **P-tuning v2** | глубокие промпты для NLU-задач | — |
| **Адаптеры** (Houlsby et al., 2019) | маленький bottleneck-MLP $d\to m\to d$ с residual после подблоков | 0.5–5%; добавляет задержку |
| **(IA)³** | поэлементные масштабы для $K$, $V$ и FFN | ~0.01% |
| **BitFit** | только bias | ~0.1% |

Prompt tuning хорошо работает лишь на больших моделях (≥ 10B — догоняет полное дообучение); на малых проигрывает LoRA.

</details>

**Часть C. Выравнивание по предпочтениям**

<details>
<summary><b>10. RLHF (InstructGPT, Ouyang et al., 2022)</b></summary>

1. **SFT** на демонстрациях людей (~13k промптов).
2. **Reward-модель:** для промпта генерируются несколько ответов, люди ранжируют; RM (та же LM с головой-скаляром) учится по модели
   Брэдли–Терри: $\mathcal L = -\log\sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$ (~33k промптов сравнений).
3. **PPO:** политика $\pi_\theta$ (инициализирована SFT) максимизирует $\mathbb E[r_\phi(x, y)] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\text{SFT}})$ (~31k промптов).
   KL-штраф не даёт уйти от разумного языка и «взломать» RM (reward hacking).

Результат: ответы 1.3B InstructGPT люди предпочитали ответам 175B GPT-3; меньше выдумок и токсичности; небольшая плата на бенчмарках
(«alignment tax»), компенсируемая подмешиванием LM-лосса. Минусы: четыре модели в памяти (политика, референс, RM, critic), нестабильный PPO.

</details>

<details>
<summary><b>11. DPO и другие методы без RL</b></summary>

**DPO** (Rafailov et al., 2023): оптимальная политика задачи «reward − β·KL» выражается через reward в замкнутой форме, и обратно —
reward через политику. Подстановка в Брэдли–Терри даёт **классификационный** лосс на парах (вывод — см. раздел «4. DPO из RLHF»):

$$
\mathcal L_{\mathrm{DPO}} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big).
$$

Нет RM, нет сэмплирования, две модели (политика и замороженный референс); обучается как SFT (`DPOTrainer` в TRL). Ограничения: офлайн
(данные не от текущей политики), может переоптимизироваться. Родственники: **IPO**, **KTO** (не пары, а отдельные «хорошо/плохо»),
**ORPO** (SFT + предпочтения одним лоссом без референса), **SimPO**. **RLAIF / Constitutional AI** — предпочтения размечает модель по
набору принципов. **GRPO** (DeepSeek) — PPO без critic: преимущество считается относительно группы ответов на тот же промпт; основа
обучения «рассуждающих» моделей с проверяемыми наградами (математика, код).

</details>

<details>
<summary><b>12. Как выбрать подход</b></summary>

| Ситуация | Что делать |
|---|---|
| Нужен формат / стиль / тон | промпт и few-shot ([тема 26](../26_llm_rag_agents/README.md)) → если мало, LoRA-SFT на сотнях–тысячах примеров |
| Нужны знания из своих документов | **RAG**, не дообучение |
| Узкая задача классификации/извлечения | маленькая encoder-модель (BERT) + полное дообучение или LoRA |
| Мало памяти GPU | QLoRA |
| Много клиентов/задач на одной базе | отдельные LoRA-адаптеры, переключение без перезагрузки базы |
| Нужно «как больше нравится людям» | DPO на парах предпочтений |

</details>

<details>
<summary><b>Чек-лист</b></summary>

- [ ] Правильный chat template модели; `eos` в конце ответа; лосс только на ответе.
- [ ] LoRA: все линейные слои, $r$ 8–64, $\alpha\approx2r$, lr ~1e-4…2e-4, warm-up, малое число эпох (1–3).
- [ ] Валидация на отложенных промптах + ручной просмотр; следить за забыванием общих навыков.
- [ ] QLoRA: `nf4`, double quant, compute dtype bf16, `prepare_model_for_kbit_training`, gradient checkpointing.
- [ ] Сохранять адаптер отдельно; вливать в веса для инференса.
- [ ] Токены (HF, wandb) — через секреты, не в коде.

</details>

**Математика и выводы**

<details>
<summary><b>1. Почему низкий ранг работает</b></summary>

**Внутренняя размерность** (Aghajanyan et al., 2020): будем дообучать не все $D$ параметров, а только $z\in\mathbb R^{d}$ в случайном
подпространстве: $\theta = \theta_0 + Pz$, $P\in\mathbb R^{D\times d}$ — случайная проекция. Наименьшее $d$, при котором достигается 90%
качества полного дообучения ($d_{90}$), для предобученных моделей оказывается порядка сотен–тысяч при $D\sim10^8$–$10^9$ и **уменьшается** с ростом
размера и длительности предобучения. То есть дообучение «живёт» в очень маломерном подпространстве — LoRA предлагает
структурированный вариант: малый ранг каждой матрицы.

Эксперименты LoRA: даже $r = 1$–$4$ на $W_Q, W_V$ GPT-3 почти не уступает $r = 64$; подпространства, выученные при разных $r$, сильно
пересекаются в верхних направлениях; $\Delta W$ усиливает направления, которые в $W_0$ были **не главными** (не повторяет его топ-сингулярные векторы).

</details>

<details>
<summary><b>2. Градиенты LoRA и инициализация</b></summary>

$h = W_0x + s\,BAx$, $s = \alpha/r$. Пусть $G = \partial\mathcal L/\partial h\,x^{\top}$ — градиент по «полному» $\Delta W$ (для батча — сумма). Тогда

$$
\frac{\partial\mathcal L}{\partial B} = s\,G\,A^{\top},\qquad \frac{\partial\mathcal L}{\partial A} = s\,B^{\top}G .
$$

- При $B = 0$: $\partial\mathcal L/\partial A = 0$, но $\partial\mathcal L/\partial B = sGA^{\top}\ne0$ — на первом шаге учится $B$, затем оживает и $A$.
- Если бы $A = B = 0$, оба градиента нулевые навсегда (седловая точка). Если обе случайные — $\Delta W\ne0$ на старте, модель испорчена.
- Обновление эффективного $\Delta W$ за шаг SGD с lr $\eta$ (с точностью до второго порядка):
  $\Delta(BA) \approx -\eta s\,(GA^{\top}A + BB^{\top}G)$ — это проекция полного градиента $G$ на подпространства строк $A$ и столбцов $B$.
  Отсюда интуиция: LoRA — градиентный спуск в изменяющемся подпространстве ранга $r$.
- **Масштаб:** при инициализации $A$ с дисперсией $\sim1/k$ величина $\lVert\Delta W\rVert$ после обучения растёт с $r$; множитель $s = \alpha/r$
  компенсирует это для первых шагов. rsLoRA (Kalajdzievski, 2023) показывает, что для стабильности при больших $r$ правильно $s = \alpha/\sqrt r$.

**Счёт параметров** ($r(d_{\text{in}} + d_{\text{out}})$ на матрицу):
- SmolLM2-135M (семинар), $r = 16$, слой: $q$ 576→576: $16\cdot1152 = 18\,432$; $k$, $v$ 576→192: по $16\cdot768 = 12\,288$; $o$: 18 432 — итого
  61 440 на слой × 30 слоёв = **1 843 200** (1.35%).
- GPT-2, `c_attn` 768→2304, $r = 8$: $8\cdot3072 = 24\,576$ × 12 слоёв = **294 912** (0.24%).
- Prompt tuning: $m\cdot d = 20\cdot768 = 15\,360$. Prefix tuning: $m\cdot L\cdot2\cdot d = 10\cdot12\cdot2\cdot768 = 184\,320$ (K и V в каждом слое).

</details>

<details>
<summary><b>3. Память: откуда 16 байт на параметр</b></summary>

Смешанная точность с Adam на параметр:

| Что | Байт |
|---|---|
| веса bf16 (для прямого/обратного прохода) | 2 |
| градиенты bf16 | 2 |
| мастер-копия весов fp32 | 4 |
| первый момент Adam $m$ fp32 | 4 |
| второй момент Adam $v$ fp32 | 4 |
| **итого** | **16** |

Плюс **активации**: $\approx$ $L\cdot T\cdot B\cdot d\cdot c$ байт ($c$ — десятки, зависит от архитектуры); **gradient checkpointing** хранит только входы
блоков и пересчитывает остальное на обратном проходе — память $O(\sqrt L)$ или $O(1)$ на слой вместо $O(L)$ ценой ~+33% вычислений.

LoRA: база — 2 байта на параметр (bf16) без градиентов и состояний; адаптер — 16 байт на свой параметр. QLoRA: база — 4 бита + шкалы.

</details>

<details>
<summary><b>4. DPO из RLHF</b></summary>

Задача выравнивания с KL-регуляризацией:

$$
\max_\pi\ \mathbb E_{x,\,y\sim\pi}\big[r(x, y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$

Для фиксированного $x$ это $\sum_y\pi(y)\big[r(y) - \beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\big]$ при $\sum_y\pi(y) = 1$. Перепишем как
$-\beta\,\mathrm{KL}\big(\pi\,\|\,\tfrac1Z\pi_{\text{ref}}e^{r/\beta}\big) + \beta\log Z$, откуда оптимум (распределение Гиббса):

$$
\pi^{*}(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\Big(\frac{r(x, y)}{\beta}\Big)
\quad\Longleftrightarrow\quad
r(x, y) = \beta\log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x).
$$

Модель предпочтений **Брэдли–Терри**: $p(y_w\succ y_l\mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$. В разности наград **$\log Z(x)$ сокращается** —
не нужно вычислять сумму по всем текстам. Подставляя параметризацию $\pi_\theta$ вместо $\pi^{*}$ и максимизируя правдоподобие пар:

$$
\mathcal L_{\mathrm{DPO}}(\theta) = -\mathbb E_{(x, y_w, y_l)}\log\sigma\Big(\underbrace{\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\hat r_\theta(x, y_w)} - \underbrace{\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\hat r_\theta(x, y_l)}\Big).
$$

«Языковая модель — тайно reward-модель». **Градиент:**

$$
\nabla_\theta\mathcal L_{\mathrm{DPO}} = -\beta\,\mathbb E\Big[\sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big]:
$$

повышаем вероятность предпочтённого, понижаем отвергнутого, с весом тем большим, чем сильнее неявная награда **ошибается** в порядке пары.
$\log\pi(y\mid x) = \sum_t\log\pi(y_t\mid x, y_{<t})$ — сумма по токенам ответа (как в SFT, с маской промпта).

</details>

<details>
<summary><b>5. Reward-модель и PPO</b></summary>

**RM:** $\mathcal L = -\mathbb E\log\sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$ — логистическая регрессия на разностях; если на промпт $K$ ответов,
используют все $\binom K2$ пар (InstructGPT: $K = 4$…$9$, пары из одного промпта — в одном батче, чтобы не переобучаться). Награда определена
с точностью до константы — её нормируют.

**PPO для LM:** эпизод — генерация ответа; награда на последнем токене $r_\phi(x, y)$, а на каждом токене — KL-штраф:

$$
R_t = -\beta\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)} + [t = T]\,r_\phi(x, y).
$$

Политика обновляется обрезанной целью $\mathbb E\big[\min(\rho_tA_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t)\big]$, $\rho_t = \pi_\theta/\pi_{\text{old}}$, преимущества
$A_t$ — по GAE с value-моделью (critic). **GRPO** выкидывает critic: для $G$ ответов на промпт $A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r)}$ —
групповой бейзлайн, плюс KL к референсу в лоссе.

</details>

<details>
<summary><b>6. Квантование NF4</b></summary>

**Absmax-квантование блока** $w\in\mathbb R^{64}$: $c = \max_i\lvert w_i\rvert$, $q_i = \operatorname{round}(w_i/c\cdot\text{уровень})$. Для 4 бит есть 16 значений; вопрос —
где их расставить. Если веса $\sim\mathcal N(0, \sigma^2)$, то после деления на absmax они лежат в $[-1, 1]$ с известным распределением, и
**равновероятные** ячейки (каждая получает $1/16$ весов) — квантили нормального распределения:

$$
q_i = \frac12\Big(Q_{\mathcal N}\Big(\frac{i}{2^4 + 1}\Big) + Q_{\mathcal N}\Big(\frac{i + 1}{2^4 + 1}\Big)\Big),
$$

нормированные в $[-1, 1]$, с точным нулём (асимметрично: 8 положительных, 7 отрицательных уровней + 0). Это минимизирует ожидаемую
ошибку при равномерном использовании кодов — «информационно оптимальный» тип для нормальных данных.

**Биты на параметр:** 4 бита + шкала fp32 на блок 64 = $4 + 32/64 = 4.5$. **Double quantization:** шкалы квантуются в 8 бит блоками по 256 с
одной fp32-шкалой второго уровня: $8/64 + 32/(64\cdot256)\approx0.127$ бита вместо 0.5 — экономия ≈ 0.37 бита на параметр (≈ 3 ГБ на 65B модели).

Матричное умножение: $Y = X\cdot\mathrm{dequant}(W_{\mathrm{NF4}}, c) + s\,XA^{\top}B^{\top}$ в bf16; градиенты по $X$ проходят через
деквантованный вес, по $W$ не считаются.

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему $B$ инициализируют нулём, а $A$ — случайно?</summary>

на старте $\Delta W=0$ (модель не испорчена), но градиент по $B$ ненулевой; если оба нуля — градиенты нулевые навсегда.

</details>

<details>
<summary>Зачем множитель $\alpha/r$?</summary>

чтобы при смене $r$ не перенастраивать lr: масштаб обновления меньше зависит от ранга (rsLoRA предлагает $\alpha/\sqrt r$).

</details>

<details>
<summary>Куда ставить LoRA?</summary>

минимум $W_Q, W_V$; лучше — все линейные слои (внимание и MLP), это часто важнее, чем большой $r$.

</details>

<details>
<summary>Увеличивает ли LoRA время инференса?</summary>

нет, если влить $BA$ в $W$; не вливают, когда нужно переключать много адаптеров на одной базе.

</details>

<details>
<summary>Почему лосс только на ответе?</summary>

иначе модель тратит ёмкость на моделирование промптов (системного, повторяющегося) и учится «писать вопросы».

</details>

<details>
<summary>SFT добавляет знания?</summary>

в основном формат, стиль, поведение (в примере: JSON 0–4% → 100%, правильная сумма ≈0%); новые факты учатся плохо и провоцируют галлюцинации — для знаний используют RAG (тема 26).

</details>

<details>
<summary>Зачем KL-штраф в RLHF?</summary>

без него политика «взламывает» reward-модель (reward hacking): уходит в странные тексты с высоким скором.

</details>

<details>
<summary>Чем DPO проще RLHF?</summary>

нет reward-модели, сэмплирования и PPO: это классификационный лосс на парах; но офлайн и чувствителен к качеству пар.

</details>

<details>
<summary>Catastrophic forgetting?</summary>

полное дообучение на узкой задаче портит общие умения; LoRA, малый lr, смешивание с общими данными уменьшают эффект.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- **Семинар 10 (вторая часть):** SmolLM2-135M base vs Instruct на вопросе о коде — base продолжает текст, instruct с chat template отвечает (но может ошибаться); LoRA с нуля (`LoRALayer` с `merge`/`unmerge`, `LoRAModel` подменяет `q_proj, k_proj, v_proj, o_proj`, $r=16$, $\alpha=32$: 1.84M обучаемых из 136M, 1.35%); SFT на анекдотах (`igorktech/anekdots_dialogs`, 2000 самых залайканных, `SFTTrainer`, `completion_only_loss=True`, lr 1e-4, накопление градиента 32); библиотека `peft` на GPT-2: LoRA на `c_attn` (0.24%), на `c_attn, c_proj, c_fc` с $r=16$ (1.86%), AdaLoRA (0.97%), QLoRA (4 бит NF4), Prompt Tuning на 20 виртуальных токенов (0.012%), Prefix Tuning на 10 (0.15%).
- **Таксономия PEFT:** аддитивные (адаптеры Houlsby, prefix/prompt tuning, (IA)³), селективные (BitFit — только bias), репараметризация (LoRA, DoRA, AdaLoRA).
- **QLoRA:** NF4 — квантили нормального распределения (веса ≈ нормальны), блоки по 64 со своей шкалой, double quantization шкал, paged AdamW; 65B модель дообучается на одной 48 ГБ GPU.
- **Цепочка InstructGPT:** SFT (13k демонстраций) → RM (33k сравнений) → PPO (31k промптов); 1.3B InstructGPT предпочитали 175B GPT-3. Далее: RLAIF / Constitutional AI, ORPO, KTO, GRPO (RL для рассуждений без critic).

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# LoRA с нуля: на старте модель не меняется (B = 0), после обучения адаптер вливается в веса без потерь
import math, torch, torch.nn as nn
torch.manual_seed(0)

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))               # ноль → ΔW = 0 при инициализации
        self.drop = nn.Dropout(dropout)
        for p in self.base.parameters(): p.requires_grad = False                # база заморожена
    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.T @ self.B.T)
    @torch.no_grad()
    def merged(self) -> nn.Linear:                                              # W = W0 + (α/r)·B·A
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if self.base.bias is not None: out.bias.copy_(self.base.bias)
        return out

base = nn.Linear(512, 512); lora = LoRALinear(base, r=8); x = torch.randn(4, 512)
print("на старте выход не изменился:", torch.allclose(lora(x), base(x)))
with torch.no_grad(): lora.B.normal_()                                          # «обучили» адаптер
print("слияние точное:", torch.allclose(lora(x), lora.merged()(x), atol=1e-4))
for d, r in [(4096, 8), (4096, 64), (8192, 16)]:
    print(f"матрица {d}×{d}: полное дообучение {d * d / 1e6:5.1f}M параметров, LoRA r={r}: {2 * d * r / 1e6:5.3f}M ({2 * r / d:.2%})")
```

Результат запуска:

```text
на старте выход не изменился: True
слияние точное: True
матрица 4096×4096: полное дообучение  16.8M параметров, LoRA r=8: 0.066M (0.39%)
матрица 4096×4096: полное дообучение  16.8M параметров, LoRA r=64: 0.524M (3.12%)
матрица 8192×8192: полное дообучение  67.1M параметров, LoRA r=16: 0.262M (0.39%)
```

```python
# Сколько нужно памяти GPU для дообучения модели на 7B параметров (без активаций)
N = 7e9
GB = lambda bytes_: bytes_ / 2**30
full = N * (2 + 2 + 4 + 4 + 4)            # bf16 веса + bf16 градиенты + fp32 мастер-копия + Adam m и v (fp32)
lora_params = 0.005 * N                   # ~0.5% обучаемых
lora = N * 2 + lora_params * (2 + 2 + 4 + 4 + 4)
qlora = N * 0.5 * 1.03 + lora_params * (2 + 2 + 4 + 4 + 4)                    # 4 бит + константы квантования
print(f"инференс bf16:          {GB(N * 2):6.1f} ГБ")
print(f"полное дообучение Adam: {GB(full):6.1f} ГБ   (≈16 байт на параметр)")
print(f"LoRA (база bf16):       {GB(lora):6.1f} ГБ")
print(f"QLoRA (база 4 бит):     {GB(qlora):6.1f} ГБ   → помещается на одну потребительскую GPU")
```

Результат запуска:

```text
инференс bf16:            13.0 ГБ
полное дообучение Adam:  104.3 ГБ   (≈16 байт на параметр)
LoRA (база bf16):         13.6 ГБ
QLoRA (база 4 бит):        3.9 ГБ   → помещается на одну потребительскую GPU
```

```python
# Chat template: как диалог превращается в строку для модели (нужен только токенизатор, несколько МБ)
import os, warnings; warnings.filterwarnings("ignore"); os.environ.setdefault("USE_TF", "0")
try:
    from transformers import AutoTokenizer
    chat_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    messages = [{"role": "system", "content": "Ты кратко отвечаешь на вопросы."},
                {"role": "user", "content": "Сколько будет 2 + 2?"}]
    print(chat_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    # add_generation_prompt=True дописывает начало реплики ассистента — модель продолжит именно ответ
except Exception as e:
    print("не удалось загрузить с Hugging Face Hub (нужен интернет):", type(e).__name__, str(e)[:120])
```

Результат запуска:

```text
<|im_start|>system
Ты кратко отвечаешь на вопросы.<|im_end|>
<|im_start|>user
Сколько будет 2 + 2?<|im_end|>
<|im_start|>assistant
```

```python
# Мини-SFT с LoRA (ruGPT-3 small, CPU ~2 мин): учим отвечать в формате JSON. Формат выучится, арифметика — нет
import json, random, time
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "sberbank-ai/rugpt3small_based_on_gpt2"
    tok, lm = AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name)
    class LoRAConv1D(nn.Module):                                                 # в GPT-2 линейные слои — Conv1D, вес [in, out]
        def __init__(self, base, r=8, alpha=16):
            super().__init__(); d_in, d_out = base.weight.shape; self.base, self.scale = base, alpha / r
            self.A = nn.Parameter(torch.randn(r, d_in) / math.sqrt(d_in)); self.B = nn.Parameter(torch.zeros(d_out, r))
        def forward(self, x): return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
    for p in lm.parameters(): p.requires_grad = False
    for blk in lm.transformer.h:                                                 # LoRA на проекции внимания
        blk.attn.c_attn, blk.attn.c_proj = LoRAConv1D(blk.attn.c_attn), LoRAConv1D(blk.attn.c_proj)
    n_tr = sum(p.numel() for p in lm.parameters() if p.requires_grad); n_all = sum(p.numel() for p in lm.parameters())
    print(f"обучаемых параметров: {n_tr:,} из {n_all:,} ({n_tr / n_all:.2%})")

    def example(rng):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"Вопрос: сколько будет {a} + {b}?\nОтвет:", " " + json.dumps({"ответ": a + b}, ensure_ascii=False) + tok.eos_token, a + b
    def evaluate(n=30):
        lm.eval(); rng = random.Random(123); fmt = correct = 0
        for _ in range(n):
            prompt, _, s = example(rng); enc = tok(prompt, return_tensors="pt")
            with torch.no_grad(): out = lm.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
            ans = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
            try: d = json.loads(ans); fmt += 1; correct += d.get("ответ") == s
            except Exception: pass
        lm.train(); return f"валидный JSON {fmt / n:.0%}, верная сумма {correct / n:.0%}, пример: {ans!r}"
    print("до SFT:   ", evaluate())

    opt, rng, t0 = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-3), random.Random(0), time.time()
    for step in range(120):
        ids, labels = [], []
        for prompt, completion, _ in (example(rng) for _ in range(8)):
            p_ids, c_ids = tok(prompt).input_ids, tok(completion).input_ids
            ids.append(p_ids + c_ids); labels.append([-100] * len(p_ids) + c_ids)        # completion-only loss
        L = max(map(len, ids))
        X = torch.tensor([s + [tok.eos_token_id] * (L - len(s)) for s in ids])
        Y = torch.tensor([s + [-100] * (L - len(s)) for s in labels])
        M = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in ids])
        loss = lm(input_ids=X, attention_mask=M, labels=Y).loss; opt.zero_grad(); loss.backward(); opt.step()
    print(f"после SFT ({time.time() - t0:.0f} с, loss {loss.item():.2f}):", evaluate())
except Exception as e:
    print("не удалось загрузить с Hugging Face Hub (нужен интернет):", type(e).__name__, str(e)[:120])
```

Результат запуска:

```text
обучаемых параметров: 442,368 из 125,673,984 (0.35%)
до SFT:    валидный JSON 3%, верная сумма 0%, пример: '65 + 97'
после SFT (78 с, loss 0.70): валидный JSON 100%, верная сумма 0%, пример: '{"ответ": 141}'
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- [yandexdataschool/nlp_course](https://github.com/yandexdataschool/nlp_course): week05 — LLM, week06 — Prompting & In-Context Learning, week07 — Fine-tuning (PEFT & RLHF), week08 — Efficiency
- [DL2 HSE, неделя 4: RLHF and LLM Agents](https://github.com/thecrazymage/DL2_HSE/tree/main/week_04) · неделя 5: агентное RL · [CS336](https://cs336.stanford.edu/) (alignment, SFT, RL)
- Документация: [Hugging Face PEFT](https://huggingface.co/docs/peft) · [TRL (SFTTrainer, DPOTrainer, GRPOTrainer)](https://huggingface.co/docs/trl) · [Chat templates](https://huggingface.co/docs/transformers/chat_templating) · [bitsandbytes](https://huggingface.co/docs/bitsandbytes) · [исходники LoRA в PEFT](https://github.com/huggingface/peft/tree/main/src/peft/tuners/lora)
- Блоги: [HF — Illustrating RLHF](https://huggingface.co/blog/rlhf) · [Lilian Weng — Reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) · [S. Raschka — Practical tips for LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) · [HF — 4-bit QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- Статьи: [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [AdaLoRA](https://arxiv.org/abs/2303.10512) · [DoRA](https://arxiv.org/abs/2402.09353) · [rsLoRA](https://arxiv.org/abs/2312.03732) · [Prompt Tuning](https://arxiv.org/abs/2104.08691) · [Prefix-Tuning](https://arxiv.org/abs/2101.00190) · [Adapters (Houlsby)](https://arxiv.org/abs/1902.00751) · [(IA)³](https://arxiv.org/abs/2205.05638) · [BitFit](https://arxiv.org/abs/2106.10199) · [Intrinsic Dimensionality](https://arxiv.org/abs/2012.13255) · [FLAN](https://arxiv.org/abs/2109.01652) · [LIMA](https://arxiv.org/abs/2305.11206) · [Self-Instruct](https://arxiv.org/abs/2212.10560) · [InstructGPT](https://arxiv.org/abs/2203.02155) · [PPO](https://arxiv.org/abs/1707.06347) · [DPO](https://arxiv.org/abs/2305.18290) · [KTO](https://arxiv.org/abs/2402.01306) · [ORPO](https://arxiv.org/abs/2403.07691) · [Constitutional AI](https://arxiv.org/abs/2212.08073) · [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) · [SmolLM2](https://arxiv.org/abs/2502.02737)

</details>

<details>
<summary><b>Источники курса</b></summary>

- Семинар 10 «Transformer → PEFT» ([Colab](https://colab.research.google.com/drive/1yQ4b8tCat38RpLbeQIIIMqYpz4QIpoRE)), вторая часть: SmolLM2-135M base vs instruct, chat template, LoRA с нуля (`LoRALayer`, `LoRAModel`, merge/unmerge), completion-only collator, SFT на анекдотах (TRL), библиотека PEFT: LoRA, LoRA на нескольких модулях, AdaLoRA, QLoRA, обучение LoRA на WikiText-2, Prompt Tuning, Prefix Tuning, задания.
- `Лекция 10.pptx`, слайды 63–65: GPT-3.5 и RLHF, эмерджентные способности.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Aghajanyan et al. [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255), 2020.
- Hu et al. [LoRA](https://arxiv.org/abs/2106.09685), 2021 · Kalajdzievski. [rsLoRA](https://arxiv.org/abs/2312.03732), 2023 · Hayou et al. [LoRA+](https://arxiv.org/abs/2402.12354), 2024.
- Dettmers et al. [QLoRA](https://arxiv.org/abs/2305.14314), 2023 · Dettmers et al. [LLM.int8()](https://arxiv.org/abs/2208.07339), 2022.
- Rafailov et al. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023 · Ouyang et al. [InstructGPT](https://arxiv.org/abs/2203.02155), 2022 · Schulman et al. [PPO](https://arxiv.org/abs/1707.06347), 2017 · Shao et al. [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300), 2024.
- Chen et al. [Training Deep Nets with Sublinear Memory Cost (checkpointing)](https://arxiv.org/abs/1604.06174), 2016 · Rajbhandari et al. [ZeRO](https://arxiv.org/abs/1910.02054), 2019 (16 байт на параметр и их шардирование).

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL, лекция 10: языковые модели, GPT
- **Семинары:** [Семинар 10: от трансформера к PEFT и SFT](../../materials/seminars/23-25_Семинар_10_Transformer_PEFT_SFT.ipynb) ([Colab](https://colab.research.google.com/drive/1yQ4b8tCat38RpLbeQIIIMqYpz4QIpoRE))
- **Видео и внешние курсы:** [Курс NLP ШАД (yandexdataschool/nlp_course)](https://github.com/yandexdataschool/nlp_course) · [DL2 HSE (thecrazymage/DL2_HSE)](https://github.com/thecrazymage/DL2_HSE) · [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/)

</details>

---

[← 24. Языковые модели, декодирование, GPT](../24_language_models_gpt/README.md) · [Оглавление курса](../../README.md) · [26. Работа с LLM: промптинг, RAG, агенты, запуск и оценка →](../26_llm_rag_agents/README.md)
