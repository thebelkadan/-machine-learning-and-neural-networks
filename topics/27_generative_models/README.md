# 27. Генеративные модели: автокодировщики, VAE, VQ-VAE, GAN, диффузия

[← Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md) · [🔬 Выводы и математика →](deep.md) · [← 26. Промптинг, RAG, агенты](../26_llm_rag_agents/README.md)

---

## Часть A. Постановка

### 1. Дискриминативные и генеративные модели

Дискриминативная модель различает категории: $p(y\mid x)$. Генеративная — **порождает** объекты: учит $p(x)$ (или условное $p(x\mid y)$,
где условие $y$ — класс, текст-промпт, другая картинка). «Генерация объекта — более сложная задача, чем детектирование свойств»:
надо построить отображение «шум → что-то разумное».

| Семейство | Плотность | Сэмплирование | Качество / разнообразие |
|---|---|---|---|
| Авторегрессионные (PixelCNN, GPT) | явная, точная $\prod_ip(x_i\mid x_{<i})$ | последовательное, медленное | хорошие оценки плотности |
| VAE | явная нижняя оценка (ELBO) | один проход декодера | размыто, разнообразно |
| Нормализующие потоки | точная (замена переменных) | один проход | ограничения архитектуры |
| GAN | неявная (только сэмплер) | один проход | резко, но mode collapse |
| Диффузия / flow matching | оценка через ELBO / ODE | много шагов | лучшее качество и покрытие |

### 2. Как измерить расстояние между распределениями (семинар по GAN)

Обозначим $\pi(x)$ — данные, $p(x)$ — модель.
- **Прямой KL** $\mathrm{KL}(\pi\,\|\,p) = \mathbb E_\pi\log\frac{\pi}{p}$ — это максимум правдоподобия. Штрафует $p\approx0$ там, где есть данные, → модель **накрывает все моды**
  (mean-seeking / mode-covering), ценой массы между ними — размытость.
- **Обратный KL** $\mathrm{KL}(p\,\|\,\pi)$ штрафует $p > 0$ там, где данных нет, → модель **садится на одну моду** (mode-seeking) — чёткие, но не все варианты.
- **JSD** (то, что оптимизирует классический GAN) ведёт себя ближе к обратному KL.

L1-пример: одна гауссиана для смеси $0.8\,\mathcal N(-2, 0.5^2) + 0.2\,\mathcal N(2, 0.5^2)$: прямой KL → $\mu = -1.2$, $\sigma = 1.68$ (между модами);
обратный KL → $\mu = -2$ или $+2$, $\sigma = 0.5$ (в зависимости от старта). Пример из жизни: модель, обученная на портретах, по обратному
KL рисует только «самый типичный» тип лица.

---

## Часть B. Автокодировщики и VAE

### 3. Автокодировщик

«Кодировщик — узкое горло — декодировщик»: сжать вход в латентное пространство и восстановить; лосс — ошибка реконструкции (+ регуляризатор).
Полносвязный или свёрточный (`Conv2d` + `MaxPool` вниз, `ConvTranspose2d` вверх).

- Без нелинейностей и с MSE — эквивалент **PCA** (подпространство главных компонент, [тема 11](../11_clustering_dim_reduction/README.md)).
- Применения: снижение размерности (и любая задача в новом пространстве), кластеризация, **детектирование аномалий** (большая ошибка
  реконструкции, [тема 10](../10_anomaly_detection/README.md)), поиск, выделение признаков, предобучение, удаление шума, inpainting.
- Виды по регуляризации: **undercomplete** (узкое горло), **denoising** (Vincent et al., 2010: по зашумлённому входу восстановить чистый —
  заодно аугментация, признаки устойчивее), **contractive** (штраф на якобиан кодировщика $\lVert\partial h/\partial x\rVert_F^2$ — нечувствительность к малым
  изменениям), **sparse** (штраф за активации).

**Почему это не генеративная модель:** (1) латентное пространство не «компактно» — непонятно, откуда брать прообразы новых объектов;
(2) нет плотности и семантической устойчивости — точка между кодами двух объектов декодируется в мусор, коды «кластеризуются»
(запоминание данных).

### 4. VAE (Kingma & Welling, 2013)

Две идеи: (1) латентное пространство — **распределение** $p(z) = \mathcal N(0, I)$, откуда потом сэмплировать; (2) кодировщик отображает объект
не в точку, а в **распределение** $q_\phi(z\mid x) = \mathcal N(\mu_\phi(x), \operatorname{diag}\sigma^2_\phi(x))$, и декодер получает любую точку из
его окрестности — латентное пространство становится плотным и гладким.

- **Кодировщик** (recognition / inference network) $q_\phi(z\mid x)$, **декодировщик** (generation network) $p_\theta(x\mid z)$.
- Хотим максимизировать $\log p_\theta(x) = \log\int p_\theta(x\mid z)p(z)\,dz$ — интеграл не считается. Оптимизируем нижнюю оценку **ELBO**
  (вывод — [deep §1](deep.md#1-elbo-два-вывода)):

$$
\log p_\theta(x)\ \ge\ \underbrace{\mathbb E_{q_\phi(z\mid x)}\log p_\theta(x\mid z)}_{\text{реконструкция}} - \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{регуляризация латента}} .
$$

Разрыв — $\mathrm{KL}(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x))\ge0$ — его нельзя вычислить, поэтому его «отбрасывают», максимизируя оценку.
**VAE = автокодировщик со специальной регуляризацией латентного пространства.** Для гауссиан KL считается в замкнутой форме:
$\frac12\sum_j(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1)$.

- **Трюк репараметризации:** сэмплирование $z\sim\mathcal N(\mu, \sigma^2)$ недифференцируемо по $\mu, \sigma$. Пишем $z = \mu + \sigma\odot\varepsilon$, $\varepsilon\sim\mathcal N(0, I)$:
  случайность вынесена в отдельный вход, градиент свободно течёт через детерминированные $\mu$ и $\sigma$. Кодировщик выдаёт $\mu$ и $\log\sigma^2$
  (стабильнее, чем $\sigma$).
- **Правдоподобие декодера:** гауссово → MSE; бернуллиево (бинарные пиксели) → BCE.
- **Генерация:** $z\sim\mathcal N(0, I)$ → декодер. **Векторная арифметика:** «улыбающаяся женщина − женщина + мужчина ≈ улыбающийся мужчина»;
  координаты $z$ могут быть интерпретируемы (поворот, толщина линии).

```python
class VAE(nn.Module):
    def __init__(self, d=784, h=400, z=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 2 * z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))
    def forward(self, x):
        mu, log_var = self.enc(x).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)          # репараметризация
        return self.dec(z), mu, log_var

def vae_loss(logits, x, mu, log_var):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    kl = 0.5 * torch.sum(mu ** 2 + log_var.exp() - log_var - 1)
    return (rec + kl) / x.size(0)
```

L1-пример (цифры 8×8, $z\in\mathbb R^2$): классы образуют области в латентном пространстве, декодер на сетке $[-2, 2]^2$ плавно переходит
0 → 6 → 4 и 9 → 3 → 2 → 1; −ELBO ≈ 22.8 нат на объект (реконструкция 20.2 + KL 2.6).

**Недостатки VAE:** результаты генерации размыты (гауссово правдоподобие = усреднение правдоподобных вариантов; прямой KL накрывает моды);
трудности обучения; **posterior collapse** — у сильного (авторегрессионного) декодера KL уходит в ноль, и $z$ игнорируется (в VAE для
последовательностей; решение — KL-annealing, вес KL растёт от 0 к 1, Bowman et al., 2015). **β-VAE:** $\beta > 1$ у KL — более «распутанные»
факторы ценой реконструкции.

**CVAE** (условный): метка $y$ подаётся в кодировщик и декодировщик — можно генерировать объект заданного класса; латенту не нужно
хранить информацию, которую несёт условие (Attribute2Image, Yan et al., 2015).

### 5. VQ-VAE (van den Oord et al., 2017)

Отличия от VAE: выход кодировщика **дискретный** — каждый вектор $z_e(x)$ карты признаков ($w\times h$ векторов) заменяется **ближайшим**
(1-NN) из обучаемой кодовой книги $\lbrace e_k\rbrace_{k=1}^{K}$; **prior обучается** — авторегрессионная модель (PixelCNN, позже трансформер) над
сеткой кодов. Лосс:

$$
\mathcal L = -\log p(x\mid z_q(x)) + \big\lVert\mathrm{sg}[z_e(x)] - e\big\rVert_2^2 + \beta\,\big\lVert z_e(x) - \mathrm{sg}[e]\big\rVert_2^2,
$$

$\mathrm{sg}$ — stop-gradient. Второе слагаемое (**VQ loss**) двигает коды к выходам кодировщика (на практике заменяется **EMA**-обновлением
кодовой книги), третье (**commitment loss**, $\beta\approx0.25$) не даёт кодировщику «скакать» между кодами. Через недифференцируемый
argmin градиент **копируется** от декодера к кодировщику (straight-through: $z_q = z_e + \mathrm{sg}[z_q - z_e]$).

**VQ-VAE-2:** иерархия кодов (верхний уровень — глобальная структура, нижний — детали); кодировщик и декодер — простые сети, а
авторегрессия идёт в латентном пространстве, а не в пикселях (в разы быстрее); большое разнообразие без mode collapse. Потомки —
**VQ-GAN** (+ adversarial и перцептуальный лоссы) — токенизаторы картинок для авторегрессионной генерации (DALL·E, Parti), латентные
пространства для диффузии.

---

## Часть C. GAN

### 6. Идея (Goodfellow et al., 2014)

«Одна из лучших идей в ML», первый подход, давший реалистичные изображения. Фундаментальная проблема генерации — **как измерить качество**
сгенерированного «котика»? Ответ GAN: пусть это делает обучаемая сеть.

- **Генератор** $G(z)$ — порождает объект из шума $z\sim p(z)$.
- **Дискриминатор** $D(x)\in[0, 1]$ — классификатор «настоящий / сгенерированный».
- **Главное:** дискриминатор — это **дифференцируемая функция ошибки**; идею можно применять везде, где нет подходящего лосса (super-resolution,
  pix2pix). Плотности нет — **implicit density model**.

Применения: генерация изображений, inpainting, super-resolution, раскраска, улучшение речи, искусство, синтетические данные и симуляция.

### 7. Обучение: min-max игра

$$
\min_G\max_D\ V(D, G) = \mathbb E_{x\sim p_{\text{data}}}\log D(x) + \mathbb E_{z\sim p(z)}\log\big(1 - D(G(z))\big).
$$

Дискриминатор — обычный классификатор с кросс-энтропией: хочет $D(x)\to1$ на реальных, $D(G(z))\to0$ на фейках; генератор хочет $D(G(z))\to1$.
Алгоритм: шаг(и) по $D$ на батче реальных и фейковых, затем шаг по $G$.

- При фиксированном $G$ **оптимальный дискриминатор** $D^{*}(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$; подстановка даёт
  $V = 2\,\mathrm{JSD}(p_{\text{data}}\,\|\,p_g) - \log4$ ([deep §2](deep.md#5-gan-оптимальный-дискриминатор-и-jsd)): глобальный минимум при $p_g = p_{\text{data}}$, $D^{*}\equiv\frac12$.
- **Проблемы:** (1) если носители распределений не пересекаются (типично для многообразий малой размерности), JSD $= \log2$ — константа,
  градиенты **затухают**, когда $D$ хорошо обучен; (2) равновесие игры градиентным спуском может не находиться (колебания).
- **Non-saturating game:** вместо минимизации $\log(1 - D(G(z)))$ (плоско там, где $D$ уверенно отвергает фейк — в начале обучения) генератор
  **максимизирует** $\log D(G(z))$ — «правдоподобие того, что дискриминатор ошибся»; практически — поменять метки местами для генератора.

### 8. Проблемы на практике и советы

Первые модели путали глобальную структуру и число элементов (животные с лишними головами). Обучение нестабильно (нормы и дисперсия
градиентов), равновесие может не достигаться; добавление шума стабилизирует, но размывает. **Mode collapse:** генератор выдаёт лишь
часть мод (нет разнообразия) — L1: обычный GAN на 8 гауссианах покрыл 5–6 мод из 8 в трёх запусках. Средства: Unrolled GAN,
minibatch discrimination (D видит однообразие батча), WGAN, LSGAN.

Советы (Soumith Chintala, ganhacks; Salimans et al., 2016):
1. Изображения в $[-1, 1]$, на выходе генератора **tanh**.
2. Non-saturating loss (поменять метки real/fake для G).
3. $z$ из гауссианы, а не равномерного распределения.
4. Чистые минибатчи (только real или только fake); или virtual batch norm.
5. Без разреженных градиентов: **LeakyReLU** вместо ReLU, вниз — strided conv или average pooling, вверх — ConvTranspose2d / PixelShuffle.
6. **Label smoothing** (real → случайно из [0.7, 1.2]) — обычно только для реальных; иногда «переворачивать» часть меток.
7. DCGAN-архитектура или гибриды (VAE + GAN); трюки из RL (experience replay).
8. Adam для G (часто и для D), $\beta_1 = 0.5$; учить D несколько шагов на шаг G (особенно в WGAN).
9. Мониторить лоссы: loss(D) → 0 — провал; проверять нормы градиентов.
10. Шум ко входам D, dropout в G; условия-метки через Embedding.
11. **Feature matching:** генератор совпадает с реальными данными по средним признакам промежуточного слоя D; D может предсказывать и класс.

| Лосс дискриминатора | Модель |
|---|---|
| Binary cross-entropy | Vanilla GAN |
| Least squares (L2) | LSGAN (Mao et al.) |
| Earth-mover / Wasserstein-1 | WGAN |
| Wasserstein + gradient penalty | WGAN-GP |
| Hinge | SA-GAN, BigGAN |

### 9. Wasserstein GAN (Arjovsky, Chintala, Bottou, 2017)

**Расстояние Earth-Mover (Вассерштейна-1):** минимальная стоимость «переноса кучи песка» $p_g$ в $p_{\text{data}}$ (масса × расстояние) по всем
совместным распределениям $\gamma$ с заданными маргиналами:

$$
W(p_{\text{data}}, p_g) = \inf_{\gamma\in\Pi(p_{\text{data}}, p_g)}\mathbb E_{(x, y)\sim\gamma}\lVert x - y\rVert .
$$

В отличие от JSD, оно непрерывно и информативно даже при непересекающихся носителях. Вычисляется через **двойственность Канторовича–Рубинштейна**:

$$
W = \sup_{\lVert f\rVert_L\le1}\ \mathbb E_{p_{\text{data}}}f(x) - \mathbb E_{p_g}f(x),
$$

где супремум — по 1-липшицевым функциям. $f$ — нейросеть-**критик** (без сигмоиды, выдаёт число), которую максимизируют, а генератор
минимизирует $-\mathbb E f(G(z))$. Липшицевость в WGAN обеспечивали **обрезкой весов** ($[-0.01, 0.01]$) — грубо, но работает. Плюс: **лосс
критика отражает качество** генерации (в обычном GAN лоссы почти не информативны).

**WGAN-GP** (Gulrajani et al., 2017): вместо обрезки — мягкий штраф на норму градиента в точках на **отрезках между реальными и фейковыми**:

$$
\mathcal L_D = \mathbb E f(\tilde x) - \mathbb E f(x) + \lambda\,\mathbb E_{\hat x}\big(\lVert\nabla_{\hat x}f(\hat x)\rVert_2 - 1\big)^2,\qquad \hat x = \epsilon x + (1 - \epsilon)\tilde x,\ \lambda = 10 .
$$

Без BatchNorm в критике (штраф считается по объектам — нужен LayerNorm или ничего), $n_{\text{critic}} = 5$, Adam (1e-4, β = (0.5, 0.9)). Устойчив к
выбору архитектуры; использовался в ProGAN и др. L1: покрыл все 8 мод, но только ~40% точек у мод — «размазан» между ними.

```python
eps = torch.rand(B, 1, 1, 1)
x_hat = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
grad = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)[0]
gp = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
loss_D = critic(fake.detach()).mean() - critic(real).mean() + 10 * gp
loss_G = -critic(G(z)).mean()
```

### 10. Архитектуры GAN

- **DCGAN** (Radford et al., 2015): полносвёрточный, без пулинга — в D strided conv, в G fractionally-strided (transposed) conv; BatchNorm после
  каждого слоя кроме последнего у G и первого у D; без FC-слоёв (reshape); G — ReLU + Tanh на выходе, D — LeakyReLU + sigmoid. Впервые CNN
  для реалистичных картинок; арифметика в латентном пространстве. Пример PyTorch: $z\in\mathbb R^{100}$ → `ConvTranspose2d` 4×4 → 8×8 → … → 64×64×3.
- **cGAN:** условие подаётся **и** в генератор, **и** в дискриминатор; D оценивает реальность **пары** (условие, объект), иначе G сгенерирует
  «реального котика», игнорируя условие.
- **pix2pix** (Isola et al., 2017): перевод изображение → изображение по **парам** (силуэт → фото, сегментация → фото, день → ночь, карта ↔
  спутник); генератор **U-Net**, дискриминатор **PatchGAN** (реальность каждого патча 70×70 — быстрее, применимо к большим картинкам) + **L1**
  к целевой картинке на всём изображении.
- **CycleGAN** (Zhu et al., 2017): перенос стиля **без пар** (лошади ↔ зебры, фото ↔ Моне): два генератора $G: X\to Y$, $F: Y\to X$, два
  дискриминатора и **cycle consistency** $\lVert F(G(x)) - x\rVert_1 + \lVert G(F(y)) - y\rVert_1$. Неудачи — когда нужна смена геометрии (собака ↔ кошка).
- **BiGAN** (Donahue et al., 2016): учит и обратное отображение $E: x\to z$; D различает пары $(x, E(x))$ и $(G(z), z)$ — хорошие признаки и арифметика.
- **ProGAN** (NVIDIA, 2017): обучение с постепенным ростом разрешения 4×4 → 1024×1024.
- **BigGAN** (Brock et al., 2018): условный по классам ImageNet, 512×512, ~350M параметров; self-attention, hinge loss, class-conditional BN,
  общие эмбеддинги классов, skip-z, ортогональная регуляризация, D обновляется чаще G, EMA весов G, огромный батч; **truncation trick** —
  сэмплировать $z$ из усечённого нормального (компромисс качество ↔ разнообразие); артефакт «class leakage».
- **StyleGAN** (2019–2021): mapping network $z\to w$, стили через AdaIN, шум по слоям — управляемая генерация лиц.

---

## Часть D. Диффузионные модели

### 11. Идея

Высокое качество и разнообразие, обучение стабильнее GAN; поначалу медленная генерация. Два процесса:
- **Прямой (forward)** — фиксированная марковская цепь, постепенно добавляющая гауссов шум: $q(x_t\mid x_{t-1}) = \mathcal N(\sqrt{1 - \beta_t}\,x_{t-1}, \beta_tI)$,
  $t = 1..T$; при $T\to\infty$ данные превращаются в $\mathcal N(0, I)$. Замкнутая форма: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$,
  $\bar\alpha_t = \prod_{s\le t}(1 - \beta_s)$ — можно сразу «прыгнуть» на любой шаг.
- **Обратный (reverse)** — обучаемый: $p_\theta(x_{t-1}\mid x_t) = \mathcal N(\mu_\theta(x_t, t), \sigma_t^2I)$ — шаг денойзинга.

**DDPM** (Ho et al., 2020): параметризация через предсказание шума $\varepsilon_\theta(x_t, t)$ и упрощённый лосс

$$
\mathcal L_{\text{simple}} = \mathbb E_{t, x_0, \varepsilon}\big\lVert\varepsilon - \varepsilon_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\varepsilon,\ t\big)\big\rVert^2 .
$$

Генерация: $x_T\sim\mathcal N(0, I)$, для $t = T..1$: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\varepsilon_\theta(x_t, t)\big) + \sigma_tz$. Оригинал: $T = 1000$,
линейное расписание $\beta$ от $10^{-4}$ до 0.02, **U-Net** с residual-блоками, self-attention и синусоидальным эмбеддингом времени;
сеть фактически выдаёт **score-функцию** $\nabla_x\log p_t(x)\approx-\varepsilon_\theta/\sqrt{1 - \bar\alpha_t}$. L1: DDPM на 8 гауссианах — все 8 мод, 96% точек у мод.

```python
# обучение
t = torch.randint(0, T, (B,)); eps = torch.randn_like(x0)
xt = abar[t].sqrt().view(-1, 1, 1, 1) * x0 + (1 - abar[t]).sqrt().view(-1, 1, 1, 1) * eps
loss = F.mse_loss(eps_model(xt, t), eps)
# генерация
x = torch.randn(shape)
for t in reversed(range(T)):
    x = (x - betas[t] / (1 - abar[t]).sqrt() * eps_model(x, t)) / alphas[t].sqrt()
    if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
```

### 12. Непрерывная формулировка и ускорение

- **SDE:** прямой процесс — стохастическое дифференциальное уравнение $dx = f(x, t)\,dt + g(t)\,dw$ (с дифференциалом винеровского процесса);
  обратный — тоже SDE, зависящее от score $\nabla_x\log p_t(x)$; существует детерминированное **probability flow ODE** с теми же маргиналами
  (Song et al., 2021). Обзор — Yang et al., «Diffusion Models: A Comprehensive Survey».
- **Быстрое сэмплирование:** DDIM (детерминированные «большие» шаги, 20–50 вместо 1000), DPM-Solver, дистилляция (progressive distillation,
  consistency models, adversarial diffusion distillation — 1–4 шага).
- **Flow matching / rectified flow:** учим векторное поле прямой траектории от шума к данным — проще и быстрее (SD3, FLUX).
- **Условная генерация:** classifier guidance; **classifier-free guidance** $\hat\varepsilon = \varepsilon_\theta(x_t, \varnothing) + w\,(\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \varnothing))$ —
  ручка «следование промпту ↔ разнообразие»; ControlNet, IP-Adapter.
- **Латентная диффузия** (Stable Diffusion): диффузия в латентном пространстве автокодировщика (VAE/VQ-GAN, в 8 раз меньше по стороне),
  текст — через cross-attention к эмбеддингам текстового энкодера (CLIP/T5); U-Net → трансформеры (DiT). Видео, 3D, аудио — те же идеи.

### 13. Оценка генеративных моделей

- **FID** — расстояние Фреше между гауссианами, подогнанными к признакам Inception реальных и сгенерированных картинок (ниже — лучше).
- **Inception Score** — уверенность классификатора × разнообразие классов (устарел).
- **Precision / recall** для генеративных моделей — отдельно качество (сэмплы лежат на многообразии данных) и покрытие (все моды).
  Аналог в L1: «доля точек у мод» и «покрыто мод».
- **CLIP-score** — соответствие картинки тексту; оценка людьми (side-by-side).

### Итоги лекции

Автокодировщик — кодировщик-декодировщик для восстановления входа через узкое горло; виды задаются регуляризацией (шум — denoising,
якобиан — contractive); VAE — естественная вероятностная реализация автокодировщика с векторной арифметикой в латентном пространстве;
плотность хорошо оценивают авторегрессионные модели; GAN — состязательное обучение с массой трюков; диффузия — современный стандарт.

### Чек-лист

- [ ] VAE: $\log\sigma^2$ на выходе кодировщика, KL в замкнутой форме, KL-annealing при posterior collapse.
- [ ] GAN: tanh + вход в $[-1, 1]$, non-saturating loss, LeakyReLU в D, Adam $\beta_1 = 0.5$; мониторить разнообразие (mode collapse).
- [ ] WGAN-GP: критик без BN и сигмоиды, $\lambda = 10$, $n_{\text{critic}} = 5$.
- [ ] Диффузия: эмбеддинг времени, расписание шума, $\varepsilon$-предсказание + MSE; для скорости — DDIM/дистилляция.
- [ ] Оценка: FID + precision/recall + просмотр сэмплов; фиксировать seed — обучение GAN крайне стохастично.

---

## Что почитать / посмотреть

- 📘 Учебник ML Яндекса: [VAE](https://education.yandex.ru/handbook/ml/article/variational-autoencoder-(vae)) · [GAN](https://education.yandex.ru/handbook/ml/article/generativno-sostyazatelnye-seti-(gan)) · [Диффузионные модели](https://education.yandex.ru/handbook/ml/article/diffuzionnye-modeli)
- 📘 [Visual GenAI Course 2026 (Д. Баранчук и др.)](https://github.com/dbaranchuk/diffusion-course-and-seminars/tree/main/visual_genai_course_2026): DDPM и DSM, непрерывные SDE/ODE, flow matching, солверы (DDIM, DPM-Solver), практика, consistency/MeanFlow, distribution matching, авторегрессионная генерация (VQ-VAE/VQ-GAN, VAR, MAR), видео, мультимодальность (ControlNet, IP-Adapter), 3D; 7 домашних заданий
- 📘 [Berkeley CS294-158 «Deep Unsupervised Learning»](https://sites.google.com/view/berkeley-cs294-158-sp20/home) · [DL2 HSE](https://github.com/thecrazymage/DL2_HSE) · Stanford [CS231n](https://cs231n.stanford.edu/) (лекция Generative Models)
- 📄 Блоги: [Lilian Weng — From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) · [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) · [E. Jang — Variational Bayes](https://blog.evjang.com/2016/08/variational-bayes.html) · [Intuitively Understanding VAE](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf) · [CVAE (ijdykeman)](https://ijdykeman.github.io/ml/2016/12/21/cvae.html) · [J. Hui — WGAN, WGAN-GP](https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490) · [ganhacks](https://github.com/soumith/ganhacks) · [GAN Lab](https://poloclub.github.io/ganlab/) · [the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo) · [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) · [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) · [pix2pix](https://phillipi.github.io/pix2pix/) · [CycleGAN](https://junyanz.github.io/CycleGAN/)
- 📄 Статьи: [VAE (Kingma & Welling)](https://arxiv.org/abs/1312.6114) · [Tutorial on VAE (Doersch)](https://arxiv.org/abs/1606.05908) · [Denoising AE (Vincent 2010)](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) · [Generating Sentences from a Continuous Space (posterior collapse)](https://arxiv.org/abs/1511.06349) · [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl) · [Attribute2Image (CVAE)](https://arxiv.org/abs/1512.00570) · [VQ-VAE](https://arxiv.org/abs/1711.00937) · [VQ-VAE-2](https://arxiv.org/abs/1906.00446) · [VQ-GAN](https://arxiv.org/abs/2012.09841) · [GAN](https://arxiv.org/abs/1406.2661) · [NIPS 2016 GAN Tutorial](https://arxiv.org/abs/1701.00160) · [Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862) · [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) · [LSGAN](https://arxiv.org/abs/1611.04076) · [WGAN](https://arxiv.org/abs/1701.07875) · [WGAN-GP](https://arxiv.org/abs/1704.00028) · [Unrolled GAN](https://arxiv.org/abs/1611.02163) · [DCGAN](https://arxiv.org/abs/1511.06434) · [pix2pix](https://arxiv.org/abs/1611.07004) · [CycleGAN](https://arxiv.org/abs/1703.10593) · [BiGAN](https://arxiv.org/abs/1605.09782) · [ProGAN](https://arxiv.org/abs/1710.10196) · [BigGAN](https://arxiv.org/abs/1809.11096) · [StyleGAN](https://arxiv.org/abs/1812.04948) · [A Review on GANs](https://arxiv.org/abs/2001.06937) · [DDPM](https://arxiv.org/abs/2006.11239) · [Score SDE](https://arxiv.org/abs/2011.13456) · [DDIM](https://arxiv.org/abs/2010.02502) · [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) · [Latent Diffusion](https://arxiv.org/abs/2112.10752) · [Flow Matching](https://arxiv.org/abs/2210.02747) · [Diffusion Models: A Comprehensive Survey](https://arxiv.org/abs/2209.00796) · [FID](https://arxiv.org/abs/1706.08500)

## Источники курса

- `Лекция 12.pptx` (DEEP LEARNING, день 12, 112 слайдов): автокодировщики (свёрточный, denoising), генеративные и условные модели, VAE (вариационный вывод, ELBO, KL, репараметризация, недостатки, CVAE), VQ-VAE и VQ-VAE-2, GAN (min-max, оптимальный D, JSD, non-saturating, проблемы, советы), LSGAN, WGAN, WGAN-GP, mode collapse, DCGAN, cGAN, pix2pix, CycleGAN, BiGAN, BigGAN, ProGAN, диффузионные модели (идея, SDE, DDPM), итоги.
- Семинар по GAN ([Colab](https://colab.research.google.com/drive/1VV4cMQayOQDfYOR_uSH32Zo1OFTzB1Pz)): прямой и обратный KL, JSD, mode-seeking на смеси гауссиан, GAN на 1D-примере (коллапс в моду, зависимость от инициализации и числа шагов D), трюки, WGAN и WGAN-GP на MNIST.


<!-- materials:start -->
**Файлы курса в репозитории:**

- DL, лекция 12: генеративные модели (лекция курса, файл в репозиторий не включён)
- [Семинар: GAN, WGAN-GP](../../materials/seminars/27_Семинар_GAN_WGAN.ipynb) · [открыть в Colab](https://colab.research.google.com/drive/1VV4cMQayOQDfYOR_uSH32Zo1OFTzB1Pz)
<!-- materials:end -->

---

[← Шпаргалка](../../cheatsheet.ipynb) · [🔬 Выводы и математика →](deep.md) · [Карта курса](../../README.md)
