# 31. Обучение с подкреплением: от уравнения Беллмана до PPO и RLHF

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** В обучении с подкреплением (RL) **агент** действует в **среде**, получает **награды** и учится **политике** —
правилу выбора действий, максимизирующему суммарную награду. Правильных ответов нет: обучающий сигнал отложен и зависит от
собственных действий агента, поэтому нужен баланс **исследования** и **использования**. Формальная модель — **марковский
процесс принятия решений** (MDP); ключевые объекты — функции ценности $V(s)$, $Q(s,a)$ и **уравнение Беллмана**. Подходы:
**динамическое программирование** (модель среды известна), **value-based** (Q-learning, DQN), **policy gradient**
(REINFORCE, actor-critic) и их сочетание — **PPO**, тот же алгоритм, что в RLHF для языковых моделей (тема 25).

| Термин | Определение |
|---|---|
| MDP | состояния $S$, действия $A$, переходы $P(s'\mid s,a)$, награды $r$, дисконт $\gamma$ |
| Политика $\pi(a\mid s)$ | правило выбора действия; детерминированная или стохастическая |
| Return $G_t$ | дисконтированная сумма будущих наград $\sum_k\gamma^kr_{t+k}$ |
| $V^\pi(s)$ / $Q^\pi(s,a)$ | ожидаемый return из состояния / после действия $a$ в состоянии $s$ |
| Advantage $A(s,a)$ | $Q(s,a)-V(s)$: насколько действие лучше среднего |
| On-policy / off-policy | учимся на данных текущей политики / на данных любой (например, из replay buffer) |
| Exploration / exploitation | пробовать новое / пользоваться лучшим известным ($\varepsilon$-greedy, энтропия) |
| Model-based / model-free | используем или учим модель переходов / учимся только по опыту |

**Ключевые формулы**

$$
V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)\big[r+\gamma V^*(s')\big],\qquad
Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big],\qquad
\nabla_\theta J=\mathbb E\big[\nabla_\theta\log\pi_\theta(a\mid s)\,A(s,a)\big]
$$

## Конспект

Разделы раскрываются по клику.

**Часть A. Постановка и планирование**

<details>
<summary><b>1. Чем RL отличается от обучения с учителем</b></summary>

| | С учителем | С подкреплением |
|---|---|---|
| Сигнал | правильный ответ на каждый объект | скалярная награда, часто редкая и отложенная |
| Данные | фиксированная выборка, i.i.d. | собираются самим агентом, зависят от его политики |
| Цель | минимизировать ошибку | максимизировать суммарную награду во времени |
| Главные трудности | переобучение | исследование, присвоение заслуг (какое действие привело к награде), нестабильность |

Применения: игры (Atari, го — AlphaGo, StarCraft), робототехника и управление (часто в симуляторе с переносом в реальность),
рекомендации и реклама (долгосрочная вовлечённость, тема 13), управление ресурсами (охлаждение дата-центров, чипы), и сейчас
главное — **дообучение LLM**: RLHF, обучение рассуждениям с проверяемой наградой, агенты (темы 25, 26).

</details>

<details>
<summary><b>2. MDP, return, функции ценности</b></summary>

**Марковское свойство:** будущее зависит только от текущего состояния и действия, а не от истории. Эпизод — траектория
$s_0,a_0,r_0,s_1,a_1,r_1,\dots$ до терминального состояния. **Return** $G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\dots$; дисконт
$\gamma\in[0,1)$ делает сумму конечной и задаёт «горизонт» $\approx\frac{1}{1-\gamma}$ шагов.

- $V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s]$ — насколько хорошо находиться в состоянии при политике $\pi$.
- $Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a]$ — насколько хорошо сделать $a$ и дальше следовать $\pi$.
- Оптимальные $V^*$, $Q^*$; оптимальная политика жадная по $Q^*$: $\pi^*(s)=\arg\max_aQ^*(s,a)$.

**Уравнение Беллмана**: ценность состояния = немедленная награда + дисконтированная ценность следующего (вывод — раздел
«1. Уравнения Беллмана и сжатие» ниже).

</details>

<details>
<summary><b>3. Динамическое программирование: модель среды известна</b></summary>

Если известны $P$ и $r$, оптимальную политику можно **вычислить** без взаимодействия со средой:
- **Policy evaluation:** итерировать $V\leftarrow\sum_a\pi(a\mid s)\sum_{s'}P\,[r+\gamma V(s')]$ до сходимости.
- **Policy iteration:** оценить политику → сделать её жадной по $V$ → повторить; сходится за конечное число шагов.
- **Value iteration:** сразу итерировать оптимальный оператор Беллмана $V\leftarrow\max_a\sum_{s'}P\,[r+\gamma V(s')]$.

Пример кода этой темы: «скользкий» лабиринт 4×4 (с вероятностью 0.2 агента сносит вбок), value iteration сходится за ~30 итераций;
оптимальная политика обходит ямы с запасом, потому что сносит.

Ограничение: число состояний. В шахматах, го, робототехнике таблицу не построить — нужны аппроксимации (нейросети) и обучение
по опыту.

</details>

**Часть B. Обучение по опыту**

<details>
<summary><b>4. Monte Carlo, TD-обучение, Q-learning</b></summary>

Модель среды неизвестна — оцениваем ценности по собственному опыту:
- **Monte Carlo:** дождаться конца эпизода и обновить $V(s)$ к фактическому return. Без смещения, но с большой дисперсией.
- **TD(0):** обновлять сразу после шага к цели $r+\gamma V(s')$ — использует собственную оценку (bootstrapping): смещение, но меньше
  дисперсия, учится в процессе эпизода.
- **SARSA** (on-policy): цель $r+\gamma Q(s',a')$ с действием $a'$, которое реально выбрано политикой.
- **Q-learning** (off-policy): цель $r+\gamma\max_{a'}Q(s',a')$ — учит оптимальную $Q^*$, даже если данные собраны другой (исследующей)
  политикой.

**Исследование:** $\varepsilon$-greedy (с вероятностью $\varepsilon$ случайное действие, $\varepsilon$ уменьшается), оптимизм в начальных значениях,
UCB-бонусы, энтропийная регуляризация для стохастических политик.

Пример кода этой темы: табличный Q-learning за 20 000 эпизодов совпадает с оптимальной политикой в 8 из 11 состояний; расхождения —
в состояниях, где ценности действий почти равны или которые редко посещаются.

</details>

<details>
<summary><b>5. Deep Q-Network (DQN)</b></summary>

$Q(s,a)$ приближает нейросеть $Q_\theta$ (для Atari — CNN по 4 последним кадрам). Наивный Q-learning с нейросетью расходится:
данные сильно скоррелированы, а цель зависит от тех же весов, которые учатся. Приёмы DQN (Mnih et al., 2015):
- **Replay buffer:** хранить переходы $(s,a,r,s')$ и учиться на случайных мини-батчах — разрывает корреляции, данные используются многократно.
- **Target network:** цель $r+\gamma\max_{a'}Q_{\bar\theta}(s',a')$ считается по копии весов, обновляемой раз в $N$ шагов (или EMA).
- Лосс Хьюбера, клиппинг наград.

Улучшения: **Double DQN** (действие выбирает онлайн-сеть, оценивает целевая — убирает переоценку из-за $\max$),
Dueling (раздельные головы $V$ и $A$), prioritized replay, distributional RL, Rainbow (всё вместе).

Ограничение value-based методов: действия дискретные и их немного ($\max_a$ по всем действиям).

</details>

<details>
<summary><b>6. Policy gradient: REINFORCE</b></summary>

Оптимизировать политику $\pi_\theta(a\mid s)$ напрямую градиентным подъёмом по ожидаемому return $J(\theta)$. **Теорема о градиенте
политики** (вывод — раздел «3. Теорема о градиенте политики»):

$$
\nabla_\theta J=\mathbb E\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\Big].
$$

Смысл: повышать вероятность действий, после которых return был высоким. **REINFORCE**: сыграть эпизод, посчитать $G_t$, сделать шаг.
Работает с непрерывными действиями и стохастическими политиками. Главная проблема — **огромная дисперсия** оценки градиента;
её снижают **бейзлайном** $b(s)$: $G_t-b(s_t)$ (не вносит смещения), простейший — среднее return, лучший — $V(s)$.

Пример кода этой темы: REINFORCE на CartPole (своя реализация физики) за 400 эпизодов поднимает среднюю длину эпизода с ~36 до ~480
шагов, но кривая скачет и может откатиться — типичная нестабильность метода.

</details>

<details>
<summary><b>7. Actor-critic, A2C, PPO</b></summary>

- **Actor-critic:** «актёр» — политика $\pi_\theta$, «критик» — оценка ценности $V_\phi$. Преимущество $A_t\approx r_t+\gamma V(s_{t+1})-V(s_t)$ (TD-ошибка)
  вместо полного return: меньше дисперсия, обновления на каждом шаге. **GAE** смешивает многошаговые оценки с параметром $\lambda$.
- **A2C/A3C:** много параллельных сред для декорреляции данных.
- **TRPO:** ограничить шаг политики по KL-дивергенции — монотонное улучшение, но сложная оптимизация второго порядка.
- **PPO** (Schulman et al., 2017): та же идея проще — **обрезка** отношения вероятностей новой и старой политики в $[1-\epsilon,1+\epsilon]$
  ($\epsilon=0.2$); несколько эпох мини-батчей на одних данных; стабилен и прост — алгоритм по умолчанию.
- **SAC** (off-policy, непрерывные действия): максимизирует награду + энтропию политики; популярен в робототехнике.

</details>

**Часть C. RL сегодня**

<details>
<summary><b>8. RL для языковых моделей</b></summary>

- **RLHF** (тема 25): политика — LLM, действие — следующий токен, эпизод — ответ, награда — reward-модель в конце + штраф KL к
  референсной модели на каждом токене; оптимизатор — PPO.
- **RL с проверяемой наградой** (RLVR): награда — правильность ответа в математике или прохождение тестов в коде; **GRPO**
  (DeepSeek) убирает критика: преимущество считается относительно группы ответов на тот же вопрос. Так обучают «рассуждающие»
  модели: длинные цепочки рассуждений возникают, потому что они повышают долю правильных ответов.
- **Агентное RL:** награда за выполнение многошаговой задачи с инструментами (тема 26) — длинные горизонты, редкие награды,
  дорогие среды.

</details>

<details>
<summary><b>9. Практика и типичные трудности</b></summary>

- Среды: **Gymnasium** (CartPole, MuJoCo, Atari), PettingZoo (мультиагентные), Isaac Gym/Lab (робототехника на GPU).
- Библиотеки: **Stable-Baselines3** (готовые PPO, DQN, SAC), CleanRL (однофайловые понятные реализации), RLlib; для LLM — TRL, verl.
- **Награда** — самая важная часть: агент оптимизирует ровно то, что задано (reward hacking — лодка в игре крутится на месте,
  собирая бонусы, вместо гонки). Reward shaping ускоряет обучение, но может изменить оптимальное поведение.
- **Редкие награды** → curriculum, демонстрации (imitation learning, behaviour cloning), intrinsic motivation (любопытство).
- **Нестабильность:** несколько seed'ов обязательно, нормализация наблюдений и преимуществ, клиппинг градиентов.
- **Offline RL:** учиться по готовому логу без взаимодействия (рекомендации, медицина) — проблема действий, которых нет в данных
  (CQL, IQL); связан с contextual bandits и off-policy оценкой.
- **Sim2real:** обучение в симуляции с рандомизацией физики и перенос на робота.

**Чек-лист**
- Сначала проверить, нужна ли RL: если есть правильные ответы — это обучение с учителем; одношаговые решения — бандиты.
- Простая среда и известный бейзлайн (случайная политика, эвристика) до сложного алгоритма.
- PPO для начала; DQN для дискретных действий с дешёвой средой; SAC для непрерывного управления.
- Логировать return, длину эпизодов, энтропию политики, KL между обновлениями.

</details>

**Математика и выводы**

<details>
<summary><b>1. Уравнения Беллмана и сжатие</b></summary>

Из $G_t=r_t+\gamma G_{t+1}$ и марковости:
$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[r(s,a,s')+\gamma V^\pi(s')\big]$, а для оптимальной ценности
$V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)\big[r+\gamma V^*(s')\big]$, $Q^*(s,a)=\sum_{s'}P\,\big[r+\gamma\max_{a'}Q^*(s',a')\big]$.

Оператор $T V(s)=\max_a\sum_{s'}P\,[r+\gamma V(s')]$ — **сжатие** в норме $\lVert\cdot\rVert_\infty$ с коэффициентом $\gamma$:
$\lvert TV(s)-TU(s)\rvert\le\max_a\sum_{s'}P\,\gamma\lvert V(s')-U(s')\rvert\le\gamma\lVert V-U\rVert_\infty$ (разность максимумов не больше
максимума разностей). По теореме Банаха у $T$ единственная неподвижная точка $V^*$, и value iteration сходится к ней
геометрически: $\lVert V_k-V^*\rVert_\infty\le\gamma^k\lVert V_0-V^*\rVert_\infty$. При $\gamma=0.95$ точность $10^{-8}$ требует до ~360 итераций,
на практике меньше (в примере 31 — ценности быстро «перестают меняться» на маленькой сетке с поглощающими состояниями).

</details>

<details>
<summary><b>2. Q-learning как стохастическая аппроксимация</b></summary>

Обновление $Q\leftarrow Q+\alpha\big(\hat T Q-Q\big)$, где $\hat TQ(s,a)=r+\gamma\max_{a'}Q(s',a')$ — **несмещённая выборочная** оценка оптимального
оператора Беллмана $TQ(s,a)$ по одному переходу. Это алгоритм Роббинса–Монро для поиска неподвижной точки $TQ=Q$. Сходимость к
$Q^*$ (Watkins & Dayan, 1992) — если каждая пара $(s,a)$ посещается бесконечно часто, а шаги удовлетворяют $\sum\alpha_t=\infty$,
$\sum\alpha_t^2<\infty$. Off-policy: в цели стоит $\max$, а не действие поведенческой политики, — поэтому данные могут собираться
любой исследующей политикой. С нейросетью и off-policy данными гарантии теряются («смертельная триада»: аппроксимация +
bootstrapping + off-policy) — отсюда target network и replay в DQN. Смещение $\max$: $\mathbb E\max_a\hat Q\ge\max_a\mathbb E\hat Q$ — шум
превращается в переоценку, её убирает Double Q-learning.

</details>

<details>
<summary><b>3. Теорема о градиенте политики</b></summary>

$J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)]$, $p_\theta(\tau)=p(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$. Трюк с логарифмом:

$$
\nabla_\theta J=\int R(\tau)\nabla_\theta p_\theta(\tau)\,d\tau=\int R(\tau)\,p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)\,d\tau=\mathbb E\big[R(\tau)\nabla_\theta\log p_\theta(\tau)\big].
$$

В $\log p_\theta(\tau)$ от $\theta$ зависят только слагаемые $\log\pi_\theta(a_t\mid s_t)$ — **динамика среды не нужна**. Причинность (действие не
влияет на прошлые награды) позволяет заменить $R(\tau)$ на return с момента $t$: $\nabla J=\mathbb E\big[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)\,G_t\big]$.

**Бейзлайн не вносит смещения:** $\mathbb E_{a\sim\pi}\big[\nabla\log\pi(a\mid s)\,b(s)\big]=b(s)\sum_a\nabla\pi(a\mid s)=b(s)\nabla\sum_a\pi(a\mid s)=b(s)\nabla1=0$.
Оптимальный (по дисперсии) бейзлайн близок к $V(s)$ — отсюда $A=Q-V$ в actor-critic.

</details>

<details>
<summary><b>4. GAE и PPO</b></summary>

TD-ошибка $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ — оценка преимущества с малой дисперсией и смещением от ошибки $V$. **GAE**:

$$
\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},
$$

$\lambda=0$ — одношаговая TD (смещение), $\lambda=1$ — Monte Carlo минус $V$ (дисперсия); обычно $\lambda=0.95$.

**PPO-clip:** $\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}$,

$$
L^{\mathrm{CLIP}}=\mathbb E_t\Big[\min\big(\rho_t\hat A_t,\ \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big].
$$

При $\hat A>0$ выгода от увеличения вероятности действия ограничена ростом $\rho$ до $1+\epsilon$; при $\hat A<0$ — уменьшением до $1-\epsilon$.
$\min$ делает цель пессимистичной нижней оценкой суррогата TRPO: политика не уходит далеко от той, что собрала данные, и
данные можно переиспользовать несколько эпох. Полный лосс: $-L^{\mathrm{CLIP}}+c_1(V_\phi-\hat G)^2-c_2\,H(\pi_\theta)$.

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Чем RL отличается от обучения с учителем?</summary>

нет правильных ответов, награда отложена, данные собирает сам агент и они зависят от его политики.

</details>

<details>
<summary>Зачем дисконт $\gamma$?</summary>

делает сумму конечной, отражает неопределённость будущего; эффективный горизонт $\approx1/(1-\gamma)$.

</details>

<details>
<summary>On-policy или off-policy: Q-learning, SARSA, PPO?</summary>

Q-learning — off-policy, SARSA и PPO — on-policy (PPO частично переиспользует данные в пределах клиппинга).

</details>

<details>
<summary>Зачем в DQN replay buffer и target network?</summary>

разорвать корреляции в данных и стабилизировать цель, которая иначе меняется вместе с весами.

</details>

<details>
<summary>Почему REINFORCE нестабилен?</summary>

оценка градиента по return всего эпизода имеет огромную дисперсию; бейзлайн и критик её уменьшают.

</details>

<details>
<summary>Почему бейзлайн не смещает градиент?</summary>

$\mathbb E[\nabla\log\pi\cdot b(s)]=b(s)\nabla\sum_a\pi=0$.

</details>

<details>
<summary>Что ограничивает клиппинг в PPO?</summary>

отношение вероятностей новой и старой политики: обновление не уходит далеко от политики, собравшей данные.

</details>

<details>
<summary>Что такое reward hacking?</summary>

агент максимизирует формальную награду непредусмотренным способом; в RLHF — «взлом» reward-модели, от которого защищает KL-штраф.

</details>

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# Value iteration и табличный Q-learning в «скользком» лабиринте 4×4
import math, numpy as np, torch, torch.nn as nn
MAP = ["S..#", ".#..", "...#", "#..G"]                     # S старт, # яма (конец, награда 0), G цель (+1)
N = 4; MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]; ARROW = "↑→↓←"; SLIP = 0.2; gamma = 0.95
cell = lambda s: MAP[s // N][s % N]
def outcomes(s, a):                                         # 0.8 — куда хотели, по 0.1 — вбок
    res = []
    for turn, p in ((0, 1 - SLIP), (1, SLIP / 2), (-1, SLIP / 2)):
        dr, dc = MOVES[(a + turn) % 4]; r, c = s // N + dr, s % N + dc
        s2 = r * N + c if 0 <= r < N and 0 <= c < N else s
        res.append((p, s2, 1.0 if cell(s2) == "G" else 0.0, cell(s2) in "#G"))
    return res
q_value = lambda V, s, a: sum(p * (r + gamma * (0 if done else V[s2])) for p, s2, r, done in outcomes(s, a))
V = np.zeros(N * N)
for it in range(1, 500):
    V_new = np.array([0.0 if cell(s) in "#G" else max(q_value(V, s, a) for a in range(4)) for s in range(N * N)])
    if np.abs(V_new - V).max() < 1e-8: break
    V = V_new
opt_policy = [int(np.argmax([q_value(V, s, a) for a in range(4)])) for s in range(N * N)]
show = lambda pol: "\n".join(" ".join(cell(r * N + c) if cell(r * N + c) in "#G" else ARROW[pol[r * N + c]] for c in range(N)) for r in range(N))
print(f"value iteration: {it} итераций, V(старт) = {V[0]:.3f}\n{show(opt_policy)}\n")

rng = np.random.default_rng(0); Q = np.zeros((N * N, 4)); eps, lr = 1.0, 0.1
def env_step(s, a):
    res = outcomes(s, a); _, s2, r, done = res[rng.choice(3, p=[o[0] for o in res])]; return s2, r, done
for episode in range(20000):
    s, done = 0, False
    while not done:
        a = int(rng.integers(4)) if rng.random() < eps else int(Q[s].argmax())   # ε-greedy
        s2, r, done = env_step(s, a)
        Q[s, a] += lr * (r + gamma * (0 if done else Q[s2].max()) - Q[s, a])     # обновление Q-learning
        s = s2
    eps = max(0.05, eps * 0.9997)
free = [s for s in range(N * N) if cell(s) not in "#G"]
print(f"Q-learning (20 000 эпизодов): совпало с оптимальной в {sum(int(Q[s].argmax()) == opt_policy[s] for s in free)} из {len(free)} состояний\n"
      f"{show([int(Q[s].argmax()) for s in range(N * N)])}")
```

Результат запуска:

```text
value iteration: 32 итераций, V(старт) = 0.530
↓ → ↓ #
↓ # ↓ ←
→ ↓ ↓ #
# → → G

Q-learning (20 000 эпизодов): совпало с оптимальной в 8 из 11 состояний
↓ ← ← #
↓ # ↓ ←
→ ↓ ← #
# → → G
```

```python
# REINFORCE на CartPole (физика из классической задачи, без сторонних библиотек)
class CartPole:
    def reset(self, rng): self.x = rng.uniform(-0.05, 0.05, 4); return self.x.copy()
    def step(self, action):
        x, x_dot, th, th_dot = self.x; force = 10.0 if action == 1 else -10.0
        ct, st = math.cos(th), math.sin(th); tmp = (force + 0.05 * th_dot ** 2 * st) / 1.1
        th_acc = (9.8 * st - ct * tmp) / (0.5 * (4 / 3 - 0.1 * ct ** 2 / 1.1)); x_acc = tmp - 0.05 * th_acc * ct / 1.1
        self.x = np.array([x + 0.02 * x_dot, x_dot + 0.02 * x_acc, th + 0.02 * th_dot, th_dot + 0.02 * th_acc])
        return self.x.copy(), 1.0, abs(self.x[0]) > 2.4 or abs(self.x[2]) > 0.21     # награда 1 за каждый шаг
torch.manual_seed(0); env, rng = CartPole(), np.random.default_rng(0)
policy = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2)); opt = torch.optim.Adam(policy.parameters(), 1e-2)
lengths = []
for episode in range(400):
    s = env.reset(rng); log_probs, rewards = [], []
    for t in range(500):
        dist = torch.distributions.Categorical(logits=policy(torch.tensor(s, dtype=torch.float32)))
        a = dist.sample(); log_probs.append(dist.log_prob(a))
        s, r, done = env.step(a.item()); rewards.append(r)
        if done: break
    G, running = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))): running = rewards[t] + 0.99 * running; G[t] = running
    G = torch.tensor((G - G.mean()) / (G.std() + 1e-8), dtype=torch.float32)   # нормировка = простой бейзлайн
    loss = -(torch.stack(log_probs) * G).sum()                                  # градиент политики
    opt.zero_grad(); loss.backward(); opt.step(); lengths.append(len(rewards))
print("средняя длина эпизода (из 500 возможных) по блокам из 50 эпизодов:", [int(np.mean(lengths[i:i + 50])) for i in range(0, 400, 50)])
```

Результат запуска:

```text
средняя длина эпизода (из 500 возможных) по блокам из 50 эпизодов: [36, 124, 290, 138, 411, 482, 474, 155]
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Обучение с подкреплением](https://education.yandex.ru/handbook/ml/article/obuchenie-s-podkrepleniem)
- Книга: [Sutton & Barto — Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (свободная)
- Курсы: [OpenAI Spinning Up](https://spinningup.openai.com/) · [Practical RL (ШАД)](https://github.com/yandexdataschool/Practical_RL) · [DL2 HSE, недели 4–5: RLHF и агентное RL](https://github.com/thecrazymage/DL2_HSE) · [David Silver — UCL RL Course](https://www.davidsilver.uk/teaching/) · [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course)
- Инструменты: [Gymnasium](https://gymnasium.farama.org/) · [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) · [CleanRL](https://github.com/vwxyzjn/cleanrl)
- Статьи: [DQN (Atari)](https://arxiv.org/abs/1312.5602) · [Double DQN](https://arxiv.org/abs/1509.06461) · [Rainbow](https://arxiv.org/abs/1710.02298) · [A3C](https://arxiv.org/abs/1602.01783) · [TRPO](https://arxiv.org/abs/1502.05477) · [GAE](https://arxiv.org/abs/1506.02438) · [PPO](https://arxiv.org/abs/1707.06347) · [SAC](https://arxiv.org/abs/1801.01290) · [CQL (offline RL)](https://arxiv.org/abs/2006.04779) · [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) · [Lilian Weng — Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

</details>

<details>
<summary><b>Источники курса</b></summary>

- Тема дополнительная: в лекциях курса RL затрагивается только как часть RLHF (тема 25). Материал по учебнику ML Яндекса, курсу DL2 HSE (недели 4–5), книге Саттона и Барто и оригинальным статьям.

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Внешние курсы:** [Учебник ML Яндекса: обучение с подкреплением](https://education.yandex.ru/handbook/ml/article/obuchenie-s-podkrepleniem) · [DL2 HSE](https://github.com/thecrazymage/DL2_HSE) · [Practical RL (ШАД)](https://github.com/yandexdataschool/Practical_RL)
- **Связанные темы:** [25. Дообучение LLM, RLHF](../25_llm_finetuning_peft/README.md) · [26. Агенты](../26_llm_rag_agents/README.md) · [13. Рекомендательные системы](../13_recommender_systems/README.md)

</details>

---

[← 30. Интерпретируемость моделей: важность признаков, PDP, SHAP, LIME](../30_interpretability/README.md) · [Оглавление курса](../../README.md) · [32. Графовые нейросети →](../32_graph_neural_networks/README.md)
