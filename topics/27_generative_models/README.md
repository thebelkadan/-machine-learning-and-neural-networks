# 27. Генеративные модели: автокодировщики, VAE, VQ-VAE, GAN, диффузия

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md) · [28. Vision Transformer, self-supervised и мультимодальные модели →](../28_vit_ssl_multimodal/README.md)

**Суть.** Генеративная модель учится порождать новые объекты из распределения данных $p(x)$ (условная — из $p(x\mid y)$, где $y$ —
класс, текст-промпт, картинка). **Автокодировщик** сжимает вход через узкое горло и восстанавливает, но генерировать не умеет:
латентное пространство «дырявое». **VAE** делает латентное пространство распределением: учит ELBO =
реконструкция − KL до априорного $\mathcal N(0,I)$, градиент — через **трюк репараметризации**. **VQ-VAE** — дискретные коды +
авторегрессионная модель над ними. **GAN** — игра генератора и дискриминатора, без явной плотности; чёткие картинки, но
нестабильное обучение и **mode collapse**; **WGAN-GP** стабилизирует через расстояние Вассерштейна. **Диффузия** постепенно
зашумляет данные и учит сеть убирать шум ($\varepsilon$-предсказание, простой MSE); генерация — много шагов денойзинга.
Сейчас это основа генерации изображений и видео.

| Термин | Определение |
|---|---|
| Явная / неявная плотность | модель даёт $p_\theta(x)$ (VAE — оценку снизу, авторегрессия, потоки) / только сэмплер (GAN) |
| ELBO | $\mathbb E_{q(z\mid x)}\log p(x\mid z)-\mathrm{KL}(q(z\mid x)\,\|\,p(z))\le\log p(x)$ |
| Репараметризация | $z=\mu+\sigma\odot\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$ — случайность вынесена из пути градиента |
| Posterior collapse | KL→0, декодер игнорирует $z$ (сильный декодер, особенно авторегрессионный) |
| VQ-VAE | $z_q=$ ближайший вектор кодовой книги; straight-through градиент; commitment loss |
| Min-max игра | $\min_G\max_D\ \mathbb E\log D(x)+\mathbb E\log(1-D(G(z)))$; при оптимальном $D$ — JSD |
| Non-saturating loss | генератор максимизирует $\log D(G(z))$ вместо минимизации $\log(1-D(G(z)))$ |
| Mode collapse | генератор выдаёт лишь часть мод данных |
| WGAN / WGAN-GP | критик без сигмоиды оценивает $W_1$; липшицевость клиппингом весов / штрафом на норму градиента |
| cGAN, pix2pix, CycleGAN | условие в $G$ и $D$; пары картинок (U-Net + PatchGAN + L1); без пар (cycle consistency) |
| DDPM | $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$; лосс $\lVert\varepsilon-\varepsilon_\theta(x_t,t)\rVert^2$ |

$$
\mathcal L_{\mathrm{VAE}}=-\mathbb E_{q_\phi(z\mid x)}\log p_\theta(x\mid z)+\mathrm{KL}\big(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,I)\big),\quad
\mathrm{KL}=\tfrac12\sum_j\big(\mu_j^2+\sigma_j^2-\ln\sigma_j^2-1\big)
$$

$$
D^*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)},\qquad
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad
\mathcal L_{\text{simple}}=\mathbb E_{t,x_0,\varepsilon}\big\lVert\varepsilon-\varepsilon_\theta(x_t,t)\big\rVert^2
$$

## Конспект

Разделы раскрываются по клику.

**Часть A. Постановка**

<details>
<summary><b>1. Дискриминативные и генеративные модели</b></summary>

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

</details>

<details>
<summary><b>2. Как измерить расстояние между распределениями (семинар по GAN)</b></summary>

Обозначим $\pi(x)$ — данные, $p(x)$ — модель.
- **Прямой KL** $\mathrm{KL}(\pi\,\|\,p) = \mathbb E_\pi\log\frac{\pi}{p}$ — это максимум правдоподобия. Штрафует $p\approx0$ там, где есть данные, → модель **накрывает все моды**
  (mean-seeking / mode-covering), ценой массы между ними — размытость.
- **Обратный KL** $\mathrm{KL}(p\,\|\,\pi)$ штрафует $p > 0$ там, где данных нет, → модель **садится на одну моду** (mode-seeking) — чёткие, но не все варианты.
- **JSD** (то, что оптимизирует классический GAN) ведёт себя ближе к обратному KL.

Пример кода этой темы: одна гауссиана для смеси $0.8\,\mathcal N(-2, 0.5^2) + 0.2\,\mathcal N(2, 0.5^2)$: прямой KL → $\mu = -1.2$, $\sigma = 1.68$ (между модами);
обратный KL → $\mu = -2$ или $+2$, $\sigma = 0.5$ (в зависимости от старта). Пример из жизни: модель, обученная на портретах, по обратному
KL рисует только «самый типичный» тип лица.

</details>

**Часть B. Автокодировщики и VAE**

<details>
<summary><b>3. Автокодировщик</b></summary>

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

</details>

<details>
<summary><b>4. VAE (Kingma & Welling, 2013)</b></summary>

Две идеи: (1) латентное пространство — **распределение** $p(z) = \mathcal N(0, I)$, откуда потом сэмплировать; (2) кодировщик отображает объект
не в точку, а в **распределение** $q_\phi(z\mid x) = \mathcal N(\mu_\phi(x), \operatorname{diag}\sigma^2_\phi(x))$, и декодер получает любую точку из
его окрестности — латентное пространство становится плотным и гладким.

- **Кодировщик** (recognition / inference network) $q_\phi(z\mid x)$, **декодировщик** (generation network) $p_\theta(x\mid z)$.
- Хотим максимизировать $\log p_\theta(x) = \log\int p_\theta(x\mid z)p(z)\,dz$ — интеграл не считается. Оптимизируем нижнюю оценку **ELBO**
  (вывод — см. раздел «1. ELBO: два вывода»):

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

Пример кода этой темы (цифры 8×8, $z\in\mathbb R^2$): классы образуют области в латентном пространстве, декодер на сетке $[-2, 2]^2$ плавно переходит
0 → 6 → 4 и 9 → 3 → 2 → 1; −ELBO ≈ 22.8 нат на объект (реконструкция 20.2 + KL 2.6).

**Недостатки VAE:** результаты генерации размыты (гауссово правдоподобие = усреднение правдоподобных вариантов; прямой KL накрывает моды);
трудности обучения; **posterior collapse** — у сильного (авторегрессионного) декодера KL уходит в ноль, и $z$ игнорируется (в VAE для
последовательностей; решение — KL-annealing, вес KL растёт от 0 к 1, Bowman et al., 2015). **β-VAE:** $\beta > 1$ у KL — более «распутанные»
факторы ценой реконструкции.

**CVAE** (условный): метка $y$ подаётся в кодировщик и декодировщик — можно генерировать объект заданного класса; латенту не нужно
хранить информацию, которую несёт условие (Attribute2Image, Yan et al., 2015).

</details>

<details>
<summary><b>5. VQ-VAE (van den Oord et al., 2017)</b></summary>

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

</details>

**Часть C. GAN**

<details>
<summary><b>6. Идея (Goodfellow et al., 2014)</b></summary>

«Одна из лучших идей в ML», первый подход, давший реалистичные изображения. Фундаментальная проблема генерации — **как измерить качество**
сгенерированного «котика»? Ответ GAN: пусть это делает обучаемая сеть.

- **Генератор** $G(z)$ — порождает объект из шума $z\sim p(z)$.
- **Дискриминатор** $D(x)\in[0, 1]$ — классификатор «настоящий / сгенерированный».
- **Главное:** дискриминатор — это **дифференцируемая функция ошибки**; идею можно применять везде, где нет подходящего лосса (super-resolution,
  pix2pix). Плотности нет — **implicit density model**.

Применения: генерация изображений, inpainting, super-resolution, раскраска, улучшение речи, искусство, синтетические данные и симуляция.

</details>

<details>
<summary><b>7. Обучение: min-max игра</b></summary>

$$
\min_G\max_D\ V(D, G) = \mathbb E_{x\sim p_{\text{data}}}\log D(x) + \mathbb E_{z\sim p(z)}\log\big(1 - D(G(z))\big).
$$

Дискриминатор — обычный классификатор с кросс-энтропией: хочет $D(x)\to1$ на реальных, $D(G(z))\to0$ на фейках; генератор хочет $D(G(z))\to1$.
Алгоритм: шаг(и) по $D$ на батче реальных и фейковых, затем шаг по $G$.

- При фиксированном $G$ **оптимальный дискриминатор** $D^{*}(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$; подстановка даёт
  $V = 2\,\mathrm{JSD}(p_{\text{data}}\,\|\,p_g) - \log4$ (см. раздел «5. GAN: оптимальный дискриминатор и JSD»): глобальный минимум при $p_g = p_{\text{data}}$, $D^{*}\equiv\frac12$.
- **Проблемы:** (1) если носители распределений не пересекаются (типично для многообразий малой размерности), JSD $= \log2$ — константа,
  градиенты **затухают**, когда $D$ хорошо обучен; (2) равновесие игры градиентным спуском может не находиться (колебания).
- **Non-saturating game:** вместо минимизации $\log(1 - D(G(z)))$ (плоско там, где $D$ уверенно отвергает фейк — в начале обучения) генератор
  **максимизирует** $\log D(G(z))$ — «правдоподобие того, что дискриминатор ошибся»; практически — поменять метки местами для генератора.

</details>

<details>
<summary><b>8. Проблемы на практике и советы</b></summary>

Первые модели путали глобальную структуру и число элементов (животные с лишними головами). Обучение нестабильно (нормы и дисперсия
градиентов), равновесие может не достигаться; добавление шума стабилизирует, но размывает. **Mode collapse:** генератор выдаёт лишь
часть мод (нет разнообразия) — в примере кода обычный GAN на 8 гауссианах покрыл 5–6 мод из 8 в трёх запусках. Средства: Unrolled GAN,
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

</details>

<details>
<summary><b>9. Wasserstein GAN (Arjovsky, Chintala, Bottou, 2017)</b></summary>

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
выбору архитектуры; использовался в ProGAN и др. В примере кода WGAN-GP покрыл все 8 мод, но только ~40% точек у мод — «размазан» между ними.

```python
eps = torch.rand(B, 1, 1, 1)
x_hat = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
grad = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)[0]
gp = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
loss_D = critic(fake.detach()).mean() - critic(real).mean() + 10 * gp
loss_G = -critic(G(z)).mean()
```

</details>

<details>
<summary><b>10. Архитектуры GAN</b></summary>

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

</details>

**Часть D. Диффузионные модели**

<details>
<summary><b>11. Идея</b></summary>

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
сеть фактически выдаёт **score-функцию** $\nabla_x\log p_t(x)\approx-\varepsilon_\theta/\sqrt{1 - \bar\alpha_t}$. В примере кода DDPM на 8 гауссианах — все 8 мод, 96% точек у мод.

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

</details>

<details>
<summary><b>12. Непрерывная формулировка и ускорение</b></summary>

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

</details>

<details>
<summary><b>13. Оценка генеративных моделей</b></summary>

- **FID** — расстояние Фреше между гауссианами, подогнанными к признакам Inception реальных и сгенерированных картинок (ниже — лучше).
- **Inception Score** — уверенность классификатора × разнообразие классов (устарел).
- **Precision / recall** для генеративных моделей — отдельно качество (сэмплы лежат на многообразии данных) и покрытие (все моды).
  Аналог в примере кода: «доля точек у мод» и «покрыто мод».
- **CLIP-score** — соответствие картинки тексту; оценка людьми (side-by-side).

</details>

<details>
<summary><b>Итоги лекции</b></summary>

Автокодировщик — кодировщик-декодировщик для восстановления входа через узкое горло; виды задаются регуляризацией (шум — denoising,
якобиан — contractive); VAE — естественная вероятностная реализация автокодировщика с векторной арифметикой в латентном пространстве;
плотность хорошо оценивают авторегрессионные модели; GAN — состязательное обучение с массой трюков; диффузия — современный стандарт.

</details>

<details>
<summary><b>Чек-лист</b></summary>

- [] VAE: $\log\sigma^2$ на выходе кодировщика, KL в замкнутой форме, KL-annealing при posterior collapse.
- [] GAN: tanh + вход в $[-1, 1]$, non-saturating loss, LeakyReLU в D, Adam $\beta_1 = 0.5$; мониторить разнообразие (mode collapse).
- [] WGAN-GP: критик без BN и сигмоиды, $\lambda = 10$, $n_{\text{critic}} = 5$.
- [] Диффузия: эмбеддинг времени, расписание шума, $\varepsilon$-предсказание + MSE; для скорости — DDIM/дистилляция.
- [] Оценка: FID + precision/recall + просмотр сэмплов; фиксировать seed — обучение GAN крайне стохастично.

</details>

**Математика и выводы**

<details>
<summary><b>1. ELBO: два вывода</b></summary>

Модель $p_\theta(x, z) = p_\theta(x\mid z)\,p(z)$, вариационное приближение $q_\phi(z\mid x)$.

**Через неравенство Йенсена** ($\log$ вогнут):

$$
\log p_\theta(x) = \log\int q_\phi(z\mid x)\,\frac{p_\theta(x, z)}{q_\phi(z\mid x)}\,dz\ \ge\ \mathbb E_{q_\phi}\log\frac{p_\theta(x, z)}{q_\phi(z\mid x)} = \mathbb E_{q_\phi}\log p_\theta(x\mid z) - \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big).
$$

**Через точное разложение** (видно, чего мы лишаемся):

$$
\log p_\theta(x) = \underbrace{\mathbb E_{q_\phi}\log\frac{p_\theta(x, z)}{q_\phi(z\mid x)}}_{\mathrm{ELBO}} + \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)}_{\ge0}.
$$

(Проверка: сумма равна $\mathbb E_q\log\frac{p_\theta(x, z)}{p_\theta(z\mid x)} = \mathbb E_q\log p_\theta(x) = \log p_\theta(x)$.) Максимизируя ELBO по $\phi$, мы приближаем $q$ к истинному
апостериорному $p_\theta(z\mid x)$ (разрыв уменьшается); по $\theta$ — увеличиваем правдоподобие. Разрыв не вычисляется — апостериорное неизвестно, —
но и не нужен. Нет гарантии, что он станет близок к нулю: семейство $q$ (диагональная гауссиана) может быть слишком бедным.

</details>

<details>
<summary><b>2. KL двух гауссиан в замкнутой форме</b></summary>

Для одномерных $q = \mathcal N(\mu, \sigma^2)$, $p = \mathcal N(0, 1)$:

$$
\mathrm{KL}(q\,\|\,p) = \mathbb E_q\Big[\log\frac{q}{p}\Big] = \mathbb E_q\Big[-\ln\sigma - \frac{(z - \mu)^2}{2\sigma^2} + \frac{z^2}{2}\Big] = -\ln\sigma - \frac12 + \frac{\mu^2 + \sigma^2}{2},
$$

т.е. $\frac12(\mu^2 + \sigma^2 - \ln\sigma^2 - 1)$; для диагональной ковариации — сумма по координатам. Минимум 0 при $\mu = 0$, $\sigma = 1$: регуляризатор
тянет коды к центру и не даёт дисперсиям схлопнуться в 0 (иначе VAE выродился бы в обычный автокодировщик с «точечными» кодами).
Общий случай: $\mathrm{KL}(\mathcal N_0\,\|\,\mathcal N_1) = \frac12\big[\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^{\top}\Sigma_1^{-1}(\mu_1 - \mu_0) - k + \ln\frac{\det\Sigma_1}{\det\Sigma_0}\big]$.

</details>

<details>
<summary><b>3. Репараметризация против REINFORCE</b></summary>

Нужен $\nabla_\phi\,\mathbb E_{q_\phi(z)}f(z)$. **Score-function (REINFORCE)** оценка:
$\nabla_\phi\mathbb E_qf = \mathbb E_q\big[f(z)\nabla_\phi\log q_\phi(z)\big]$ — работает для любых (даже дискретных) $z$, но с огромной дисперсией: $f$ входит как скаляр-множитель.
**Репараметризация:** $z = g_\phi(\varepsilon) = \mu_\phi + \sigma_\phi\odot\varepsilon$, $\varepsilon\sim\mathcal N(0, I)$ — распределение $\varepsilon$ от $\phi$ не зависит, поэтому

$$
\nabla_\phi\,\mathbb E_{q_\phi}f(z) = \mathbb E_\varepsilon\big[\nabla_\phi f(\mu_\phi + \sigma_\phi\odot\varepsilon)\big] = \mathbb E_\varepsilon\Big[\nabla_zf\cdot\frac{\partial z}{\partial\phi}\Big],
$$

используется производная $f$ — оценка по одному сэмплу на объект уже пригодна для SGD. Для дискретных латентов — Gumbel-softmax или
straight-through (как в VQ-VAE: $z_q = z_e + \mathrm{sg}[z_q - z_e]$ — вперёд значение $z_q$, назад градиент тождественный).

</details>

<details>
<summary><b>4. Прямой и обратный KL: mean-seeking и mode-seeking</b></summary>

$\mathrm{KL}(\pi\,\|\,p) = \int\pi\log\frac{\pi}{p}$: где $\pi(x) > 0$, а $p(x)\to0$, подынтегральное выражение $\to+\infty$ — модель **обязана** покрыть все
области с данными, даже ценой массы между модами (**zero-avoiding**). Максимум правдоподобия ≡ минимум прямого KL — поэтому VAE и
авторегрессионные модели покрывают моды, но «размывают».

$\mathrm{KL}(p\,\|\,\pi) = \int p\log\frac{p}{\pi}$: где $\pi\approx0$, а $p > 0$, штраф огромен; где $p = 0$ — штрафа нет. Выгодно сосредоточиться на одной
моде (**zero-forcing**). GAN с JSD ведёт себя похоже — отсюда mode collapse. Пример кода: одна гауссиана для смеси 0.8/0.2 — прямой KL даёт
$\mu = -1.2$, $\sigma = 1.68$, обратный — $\mu = \pm2$, $\sigma = 0.5$.

$\mathrm{JSD}(\pi, p) = \frac12\mathrm{KL}(\pi\,\|\,m) + \frac12\mathrm{KL}(p\,\|\,m)$, $m = \frac{\pi + p}{2}$ — симметрична и ограничена: $0\le\mathrm{JSD}\le\log2$.

</details>

<details>
<summary><b>5. GAN: оптимальный дискриминатор и JSD</b></summary>

При фиксированном $G$ (распределение фейков $p_g$):

$$
V(D, G) = \int\big[p_{\text{data}}(x)\log D(x) + p_g(x)\log(1 - D(x))\big]dx .
$$

Максимизируем подынтегральное выражение поточечно: $a\log y + b\log(1 - y)$ на $(0, 1)$ имеет максимум при $\frac{a}{y} = \frac{b}{1 - y}$, т.е.
$y = \frac{a}{a + b}$:

$$
D^{*}(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} .
$$

Подстановка:

$$
V(D^{*}, G) = \mathbb E_{p_{\text{data}}}\log\frac{p_{\text{data}}}{p_{\text{data}} + p_g} + \mathbb E_{p_g}\log\frac{p_g}{p_{\text{data}} + p_g}
= -\log4 + \mathrm{KL}\Big(p_{\text{data}}\,\Big\|\,\tfrac{p_{\text{data}} + p_g}{2}\Big) + \mathrm{KL}\Big(p_g\,\Big\|\,\tfrac{p_{\text{data}} + p_g}{2}\Big) = 2\,\mathrm{JSD} - \log4 .
$$

Минимум по $G$ — при $p_g = p_{\text{data}}$, тогда $D^{*}\equiv\frac12$ и $V = -\log4$. **Оговорки:** вывод требует оптимального $D$ на каждом шаге
(«существенное условие») и оптимизации в пространстве распределений, а не параметров сети.

**Насыщение.** Пусть $D(G(z)) = \sigma(\ell)$. Минимаксный лосс генератора $\log(1 - \sigma(\ell))$ имеет градиент по $\ell$ равный $-\sigma(\ell)$ → 0, когда $D$
уверенно отвергает фейк ($\ell\to-\infty$, начало обучения). Non-saturating $-\log\sigma(\ell)$ имеет градиент $-(1 - \sigma(\ell))\to-1$ — сильный сигнал
именно там, где генератор плох. Та же неподвижная точка, другая динамика.

</details>

<details>
<summary><b>6. Почему Вассерштейн лучше JSD</b></summary>

Пример Arjovsky et al.: $P_0$ — равномерное распределение на отрезке $\lbrace0\rbrace\times[0, 1]$, $P_\theta$ — на $\lbrace\theta\rbrace\times[0, 1]$. При $\theta\ne0$
носители не пересекаются:

$$
\mathrm{JSD}(P_0, P_\theta) = \log2,\qquad \mathrm{KL}(P_0\,\|\,P_\theta) = +\infty,\qquad W(P_0, P_\theta) = \lvert\theta\rvert .
$$

JSD постоянна — градиента по $\theta$ нет; $W$ непрерывна и даёт осмысленный градиент к $\theta = 0$. Для распределений на многообразиях
малой размерности (картинки) непересекающиеся носители — типичная ситуация, особенно в начале обучения.

**Двойственность Канторовича–Рубинштейна:** $W(P, Q) = \sup_{\lVert f\rVert_L\le1}\mathbb E_Pf - \mathbb E_Qf$. Критик $f_w$ из параметрического семейства
$K$-липшицевых функций даёт $K\cdot W$ — для обучения генератора масштаб не важен. Клиппинг весов делает сеть липшицевой, но с плохо
контролируемой константой и вырождает критик в слишком простые функции.

**Gradient penalty:** у оптимального критика (Gulrajani et al., утверждение 1) почти всюду на отрезках между точками, сопоставленными
оптимальным транспортным планом, $\lVert\nabla f^{*}\rVert = 1$. Отсюда штраф $(\lVert\nabla_{\hat x}f(\hat x)\rVert - 1)^2$ в точках
$\hat x = \epsilon x + (1 - \epsilon)\tilde x$ — дешёвая аппроксимация этих отрезков (двусторонний штраф работал лучше одностороннего $\max(0, \lVert\nabla\rVert - 1)^2$).

</details>

<details>
<summary><b>7. Диффузия: формулы DDPM</b></summary>

**Замкнутая форма прямого процесса.** $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1 - \alpha_t}\,\varepsilon_{t-1}$, $\alpha_t = 1 - \beta_t$. Подставляя рекурсивно и складывая
независимые гауссовы шумы (дисперсии складываются): $\sqrt{\alpha_t}\sqrt{1 - \alpha_{t-1}}\,\varepsilon + \sqrt{1 - \alpha_t}\,\varepsilon'$ имеет дисперсию $\alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\alpha_{t-1}$.
По индукции

$$
q(x_t\mid x_0) = \mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1 - \bar\alpha_t)I\big),\qquad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s .
$$

**Апостериорное прямого процесса** (по формуле Байеса для гауссиан) тоже гауссово:

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal N(\tilde\mu_t, \tilde\beta_tI),\quad
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}x_t,\quad
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t .
$$

Выразив $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\big(x_t - \sqrt{1 - \bar\alpha_t}\,\varepsilon\big)$, получаем $\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\varepsilon\big)$ — отсюда
параметризация $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\varepsilon_\theta(x_t, t)\big)$ и формула шага генерации.

**Лосс.** ELBO для цепочки распадается на сумму $\mathrm{KL}\big(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)$; для гауссиан с одинаковой дисперсией это
$\frac{1}{2\sigma_t^2}\lVert\tilde\mu_t - \mu_\theta\rVert^2 = \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1 - \bar\alpha_t)}\lVert\varepsilon - \varepsilon_\theta\rVert^2$. DDPM отбрасывает веса — $\mathcal L_{\text{simple}}$; это
перевзвешенная ELBO, которая лучше по качеству картинок (больше внимания большим $t$).

**Связь со score:** $\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}x_0}{1 - \bar\alpha_t} = -\frac{\varepsilon}{\sqrt{1 - \bar\alpha_t}}$, и денойзинг по Твиди даёт
$s_\theta(x_t, t) = -\varepsilon_\theta(x_t, t)/\sqrt{1 - \bar\alpha_t}\approx\nabla\log p_t(x_t)$ — предсказание шума = denoising score matching. Генерация — по сути
ланжевеновская динамика по выученному score с убывающим уровнем шума.

**Расписание.** $\bar\alpha_T$ должна быть $\approx0$, чтобы $x_T\approx\mathcal N(0, I)$: при линейном $\beta\in[10^{-4}, 0.02]$ и $T = 1000$
$\ln\bar\alpha_T\approx-\sum\beta_t\approx-10$. В примере кода ($T = 100$, $\beta$ до 0.1) $\sum\beta\approx5$, $\bar\alpha_T\approx0.007$.

</details>

<details>
<summary><b>8. FID</b></summary>

Признаки Inception-v3 (пул-слой, 2048) для реальных и сгенерированных изображений аппроксимируются гауссианами $(\mu_r, \Sigma_r)$, $(\mu_g, \Sigma_g)$;
расстояние Фреше (Вассерштейна-2) между ними:

$$
\mathrm{FID} = \lVert\mu_r - \mu_g\rVert^2 + \operatorname{tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\big).
$$

Учитывает и качество, и разнообразие (mode collapse сужает $\Sigma_g$), но смещён при малых выборках (нужно ≥ 10–50 тыс. картинок) и зависит от
Inception (ImageNet-признаки). Поэтому дополняют precision/recall в пространстве признаков (Kynkäänniemi et al., 2019).

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему обычный автокодировщик — не генератор?</summary>

латентное пространство не регуляризовано: случайная точка $z$ попадает в «дыру», декодер выдаёт мусор.

</details>

<details>
<summary>Зачем репараметризация?</summary>

градиент нельзя провести через операцию сэмплирования; $z=\mu+\sigma\varepsilon$ делает $z$ дифференцируемой функцией $\mu,\sigma$.

</details>

<details>
<summary>Почему VAE даёт размытые картинки?</summary>

гауссово правдоподобие = MSE, оптимум — среднее по правдоподобным вариантам; плюс прямой KL «накрывает» все моды (mean-seeking).

</details>

<details>
<summary>Почему GAN теряет моды?</summary>

генератор оптимизирует «обмануть текущий $D$», а не покрыть $p_{\text{data}}$ (обратный KL/JSD — mode-seeking); в примере обычный GAN покрывает 5–6 из 8 мод (WGAN-GP и DDPM — все 8; у WGAN-GP при этом лишь ~40% точек попадают в моды — размытость).

</details>

<details>
<summary>Почему обучение GAN нестабильно?</summary>

ищем седловую точку игры градиентным спуском; при непересекающихся носителях JSD = $\log2$ — градиенты исчезают, когда $D$ уже идеален.

</details>

<details>
<summary>Что гарантирует WGAN-GP?</summary>

мягкую 1-липшицевость критика; лосс критика коррелирует с качеством; нет BatchNorm в критике.

</details>

<details>
<summary>Чем диффузия лучше GAN?</summary>

стабильный MSE-лосс, покрытие всех мод, легко добавить условие; минус — медленная генерация (десятки–сотни шагов, лечат DDIM, дистилляцией, consistency models).

</details>

<details>
<summary>Как оценивать генеративные модели?</summary>

FID (расстояние между гауссианами признаков Inception), precision/recall (качество vs покрытие), CLIP-score для текст-картинка, люди.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- **Автокодировщики:** undercomplete (узкое горло; без нелинейности ≈ PCA), denoising (восстанавливать чистый вход по зашумлённому), contractive (штраф $\lVert\partial h/\partial x\rVert_F^2$), sparse. Применения: снижение размерности, аномалии (ошибка реконструкции — тема 10), поиск, признаки, предобучение, удаление шума, inpainting.
- **CVAE:** условие $y$ подаётся в кодировщик и декодировщик — генерация объекта заданного класса; латент перестаёт хранить «класс». **β-VAE:** вес $\beta>1$ у KL — более «распутанные» латенты; KL-annealing против posterior collapse (Bowman et al.).
- **VQ-VAE / VQ-VAE-2:** $\mathcal L=\log p(x\mid z_q)+\lVert\mathrm{sg}[z_e]-e\rVert^2+\beta\lVert z_e-\mathrm{sg}[e]\rVert^2$, кодовая книга обновляется EMA; prior — PixelCNN над кодами; иерархия кодов, разнообразие без mode collapse. Те же токенизаторы (VQ-GAN) — основа авторегрессионной генерации картинок.
- **Зоопарк GAN:** DCGAN (полносвёрточный, strided conv, BN, ReLU/Tanh в G, LeakyReLU в D), LSGAN, cGAN, pix2pix (U-Net + PatchGAN + L1), CycleGAN ($G:X\to Y$, $F:Y\to X$, $\lVert F(G(x))-x\rVert_1$), BiGAN (учит и обратное отображение), ProGAN (постепенный рост разрешения), BigGAN (self-attention, hinge loss, условный BN, truncation trick, EMA весов, батч 2048), StyleGAN.
- **Советы по обучению GAN:** вход в $[-1,1]$ и tanh на выходе; non-saturating loss; $z$ из гауссианы; чистые батчи (только real / только fake); LeakyReLU, без max-pool; label smoothing (real → 0.7–1.2); Adam; мониторить, не ушёл ли loss(D) в 0; шум ко входам D; feature matching, minibatch discrimination.
- **Диффузия:** DDPM ($T=1000$, линейное расписание $\beta$, U-Net с вниманием и эмбеддингом времени); score-интерпретация $\varepsilon_\theta\approx-\sqrt{1-\bar\alpha_t}\,\nabla\log p_t$; SDE/ODE-формулировки; DDIM — детерминированные быстрые шаги; classifier-free guidance; латентная диффузия (Stable Diffusion: VAE-латенты + текст через cross-attention); flow matching.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
# Прямой и обратный KL: какую часть смеси «выберет» одна гауссиана (основа поведения VAE vs GAN)
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
xs = torch.linspace(-6, 6, 2001); dx = xs[1] - xs[0]
normal = lambda x, m, s: torch.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
pi = 0.8 * normal(xs, -2.0, 0.5) + 0.2 * normal(xs, 2.0, 0.5)                   # данные: смесь двух мод
def fit(direction, mu0):
    mu, log_s = torch.tensor(mu0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([mu, log_s], 0.05)
    for _ in range(1500):
        p = normal(xs, mu, log_s.exp()) + 1e-12
        kl = (pi * (pi + 1e-12).log() - pi * p.log()).sum() * dx if direction == "forward" else (p * (p.log() - (pi + 1e-12).log())).sum() * dx
        opt.zero_grad(); kl.backward(); opt.step()
    return mu.item(), log_s.exp().item()
for direction in ("forward", "reverse"):
    for mu0 in (-1.0, 1.0):
        m, s = fit(direction, mu0)
        print(f"KL {direction:7s} (старт μ={mu0:+.0f}): μ={m:+.2f}, σ={s:.2f}  ->",
              "накрывает обе моды (mean-seeking)" if s > 1 else "садится на одну моду (mode-seeking)")
```

Результат запуска:

```text
KL forward (старт μ=-1): μ=-1.20, σ=1.68  -> накрывает обе моды (mean-seeking)
KL forward (старт μ=+1): μ=-1.20, σ=1.68  -> накрывает обе моды (mean-seeking)
KL reverse (старт μ=-1): μ=-2.00, σ=0.50  -> садится на одну моду (mode-seeking)
KL reverse (старт μ=+1): μ=+2.00, σ=0.50  -> садится на одну моду (mode-seeking)
```

```python
# VAE на рукописных цифрах 8×8: ELBO, репараметризация, генерация из N(0, I)
from sklearn.datasets import load_digits
digits = load_digits(); X = torch.tensor(digits.data / 16, dtype=torch.float32); y = digits.target

class VAE(nn.Module):
    def __init__(self, d=64, h=256, z=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 2 * z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))
    def forward(self, x):
        mu, log_var = self.enc(x).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()                 # трюк репараметризации
        return self.dec(z), mu, log_var

torch.manual_seed(0); vae = VAE(); opt = torch.optim.Adam(vae.parameters(), 1e-3)
for epoch in range(300):
    for idx in torch.randperm(len(X)).split(128):
        logits, mu, log_var = vae(X[idx])
        rec = F.binary_cross_entropy_with_logits(logits, X[idx], reduction="sum") / len(idx)
        kl = 0.5 * (mu ** 2 + log_var.exp() - log_var - 1).sum() / len(idx)
        loss = rec + kl; opt.zero_grad(); loss.backward(); opt.step()
print(f"реконструкция {rec.item():.1f} нат, KL {kl.item():.2f} нат, -ELBO {loss.item():.1f} на объект")

with torch.no_grad():
    mu_all = vae.enc(X).chunk(2, dim=-1)[0].numpy()
    grid = torch.tensor([[a, b] for b in np.linspace(2, -2, 8) for a in np.linspace(-2, 2, 8)], dtype=torch.float32)
    tiles = torch.sigmoid(vae.dec(grid)).reshape(8, 8, 8, 8).numpy()               # [строка, столбец, 8, 8]
    tiles = np.pad(tiles, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=0.3)    # серые рамки между цифрами
    gen = tiles.transpose(0, 2, 1, 3).reshape(80, 80)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
sc = ax[0].scatter(mu_all[:, 0], mu_all[:, 1], c=y, cmap="tab10", s=6); fig.colorbar(sc, ax=ax[0]); ax[0].set_title("латентное пространство (μ)")
ax[1].imshow(gen, cmap="gray"); ax[1].set_title("декодер на сетке z ∈ [-2, 2]²"); ax[1].axis("off"); plt.tight_layout(); plt.show()
```

Результат запуска:

```text
реконструкция 20.2 нат, KL 2.58 нат, -ELBO 22.8 на объект
```

График — в ноутбуке `cheatsheet.ipynb`.

```python
# GAN vs WGAN-GP vs диффузия (DDPM) на 8 гауссианах по кругу: сколько мод покрыто и насколько точно
centers = torch.tensor([[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]) * 2
sample = lambda n: centers[torch.randint(0, 8, (n,))] + 0.02 * torch.randn(n, 2)
mlp = lambda i, o, h=128: nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
def quality(x):                                                                # (число покрытых мод, доля точек рядом с модами)
    d = torch.cdist(x, centers); near = d.min(1).values < 0.25
    return int(torch.bincount(d.argmin(1)[near], minlength=8).gt(len(x) * 0.02).sum()), near.float().mean().item()

def train_gan(seed, wgan=False, steps=2000):
    torch.manual_seed(seed); G, D = mlp(2, 2), mlp(2, 1)
    lr, betas, n_critic = (1e-4, (0.5, 0.9), 5) if wgan else (1e-3, (0.5, 0.999), 1)
    oG, oD = torch.optim.Adam(G.parameters(), lr, betas=betas), torch.optim.Adam(D.parameters(), lr, betas=betas)
    for step in range(steps):
        for _ in range(n_critic):                                             # критика учим чаще генератора
            x, fake = sample(256), G(torch.randn(256, 2)).detach()
            if wgan:
                e = torch.rand(256, 1); xh = (e * x + (1 - e) * fake).requires_grad_(True)
                g = torch.autograd.grad(D(xh).sum(), xh, create_graph=True)[0]
                lD = D(fake).mean() - D(x).mean() + 10 * ((g.norm(dim=1) - 1) ** 2).mean()     # gradient penalty
            else:
                lD = F.binary_cross_entropy_with_logits(D(x), torch.ones(256, 1)) + F.binary_cross_entropy_with_logits(D(fake), torch.zeros(256, 1))
            oD.zero_grad(); lD.backward(); oD.step()
        out = D(G(torch.randn(256, 2)))
        lG = -out.mean() if wgan else F.binary_cross_entropy_with_logits(out, torch.ones(256, 1))   # non-saturating
        oG.zero_grad(); lG.backward(); oG.step()
    with torch.no_grad(): return G(torch.randn(5000, 2))

T = 100; betas = torch.linspace(1e-4, 0.1, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)
class EpsNet(nn.Module):                                                        # ε_θ(x_t, t): MLP + синусоидальное время
    def __init__(self):
        super().__init__(); self.net = mlp(2 + 16, 2)
    def forward(self, x, t):
        a = t[:, None].float() * torch.exp(-math.log(1000) * torch.arange(8) / 8)
        return self.net(torch.cat([x, torch.sin(a), torch.cos(a)], 1))
def train_ddpm(seed, steps=4000):
    torch.manual_seed(seed); eps_net = EpsNet(); opt = torch.optim.Adam(eps_net.parameters(), 1e-3)
    for _ in range(steps):
        x0, t = sample(256), torch.randint(0, T, (256,)); e = torch.randn_like(x0)
        xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * e            # прыжок сразу на шаг t
        loss = F.mse_loss(eps_net(xt, t), e); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                                                       # генерация: T шагов денойзинга
        x = torch.randn(5000, 2)
        for t in reversed(range(T)):
            e = eps_net(x, torch.full((5000,), t))
            x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
            if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
    return x

results = {f"GAN, seed {s}": train_gan(s) for s in (1, 2, 3)}
results["WGAN-GP"] = train_gan(1, wgan=True, steps=1500)
results["DDPM"] = train_ddpm(1)
fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.2))
for ax, (name, x) in zip(axes, results.items()):
    modes, prec = quality(x); print(f"{name:12s}: мод покрыто {modes}/8, точек у мод {prec:.0%}")
    ax.scatter(*x[:2000].T, s=1, alpha=0.4); ax.scatter(*centers.T, c="red", s=12); ax.set_title(f"{name}\n{modes}/8 мод"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
```

Результат запуска:

```text
GAN, seed 1 : мод покрыто 5/8, точек у мод 95%
GAN, seed 2 : мод покрыто 6/8, точек у мод 95%
GAN, seed 3 : мод покрыто 6/8, точек у мод 94%
WGAN-GP     : мод покрыто 8/8, точек у мод 41%
DDPM        : мод покрыто 8/8, точек у мод 96%
```

График — в ноутбуке `cheatsheet.ipynb`.

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [VAE](https://education.yandex.ru/handbook/ml/article/variational-autoencoder-(vae)) · [GAN](https://education.yandex.ru/handbook/ml/article/generativno-sostyazatelnye-seti-(gan)) · [Диффузионные модели](https://education.yandex.ru/handbook/ml/article/diffuzionnye-modeli)
- [Visual GenAI Course 2026 (Д. Баранчук и др.)](https://github.com/dbaranchuk/diffusion-course-and-seminars/tree/main/visual_genai_course_2026): DDPM и DSM, непрерывные SDE/ODE, flow matching, солверы (DDIM, DPM-Solver), практика, consistency/MeanFlow, distribution matching, авторегрессионная генерация (VQ-VAE/VQ-GAN, VAR, MAR), видео, мультимодальность (ControlNet, IP-Adapter), 3D; 7 домашних заданий
- [Berkeley CS294-158 «Deep Unsupervised Learning»](https://sites.google.com/view/berkeley-cs294-158-sp20/home) · [DL2 HSE](https://github.com/thecrazymage/DL2_HSE) · Stanford [CS231n](https://cs231n.stanford.edu/) (лекция Generative Models)
- Блоги: [Lilian Weng — From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) · [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) · [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) · [E. Jang — Variational Bayes](https://blog.evjang.com/2016/08/variational-bayes.html) · [Intuitively Understanding VAE](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf) · [CVAE (ijdykeman)](https://ijdykeman.github.io/ml/2016/12/21/cvae.html) · [J. Hui — WGAN, WGAN-GP](https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490) · [ganhacks](https://github.com/soumith/ganhacks) · [GAN Lab](https://poloclub.github.io/ganlab/) · [the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo) · [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) · [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) · [pix2pix](https://phillipi.github.io/pix2pix/) · [CycleGAN](https://junyanz.github.io/CycleGAN/)
- Статьи: [VAE (Kingma & Welling)](https://arxiv.org/abs/1312.6114) · [Tutorial on VAE (Doersch)](https://arxiv.org/abs/1606.05908) · [Denoising AE (Vincent 2010)](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) · [Generating Sentences from a Continuous Space (posterior collapse)](https://arxiv.org/abs/1511.06349) · [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl) · [Attribute2Image (CVAE)](https://arxiv.org/abs/1512.00570) · [VQ-VAE](https://arxiv.org/abs/1711.00937) · [VQ-VAE-2](https://arxiv.org/abs/1906.00446) · [VQ-GAN](https://arxiv.org/abs/2012.09841) · [GAN](https://arxiv.org/abs/1406.2661) · [NIPS 2016 GAN Tutorial](https://arxiv.org/abs/1701.00160) · [Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862) · [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) · [LSGAN](https://arxiv.org/abs/1611.04076) · [WGAN](https://arxiv.org/abs/1701.07875) · [WGAN-GP](https://arxiv.org/abs/1704.00028) · [Unrolled GAN](https://arxiv.org/abs/1611.02163) · [DCGAN](https://arxiv.org/abs/1511.06434) · [pix2pix](https://arxiv.org/abs/1611.07004) · [CycleGAN](https://arxiv.org/abs/1703.10593) · [BiGAN](https://arxiv.org/abs/1605.09782) · [ProGAN](https://arxiv.org/abs/1710.10196) · [BigGAN](https://arxiv.org/abs/1809.11096) · [StyleGAN](https://arxiv.org/abs/1812.04948) · [A Review on GANs](https://arxiv.org/abs/2001.06937) · [DDPM](https://arxiv.org/abs/2006.11239) · [Score SDE](https://arxiv.org/abs/2011.13456) · [DDIM](https://arxiv.org/abs/2010.02502) · [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) · [Latent Diffusion](https://arxiv.org/abs/2112.10752) · [Flow Matching](https://arxiv.org/abs/2210.02747) · [Diffusion Models: A Comprehensive Survey](https://arxiv.org/abs/2209.00796) · [FID](https://arxiv.org/abs/1706.08500)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Лекция 12.pptx` (DEEP LEARNING, день 12, 112 слайдов): автокодировщики (свёрточный, denoising), генеративные и условные модели, VAE (вариационный вывод, ELBO, KL, репараметризация, недостатки, CVAE), VQ-VAE и VQ-VAE-2, GAN (min-max, оптимальный D, JSD, non-saturating, проблемы, советы), LSGAN, WGAN, WGAN-GP, mode collapse, DCGAN, cGAN, pix2pix, CycleGAN, BiGAN, BigGAN, ProGAN, диффузионные модели (идея, SDE, DDPM), итоги.
- Семинар по GAN ([Colab](https://colab.research.google.com/drive/1VV4cMQayOQDfYOR_uSH32Zo1OFTzB1Pz)): прямой и обратный KL, JSD, mode-seeking на смеси гауссиан, GAN на 1D-примере (коллапс в моду, зависимость от инициализации и числа шагов D), трюки, WGAN и WGAN-GP на MNIST.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Kingma, Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), 2013 · Doersch. [Tutorial on VAE](https://arxiv.org/abs/1606.05908), 2016 · Jang et al. [Gumbel-Softmax](https://arxiv.org/abs/1611.01144), 2016.
- Goodfellow et al. [GAN](https://arxiv.org/abs/1406.2661), 2014 · Arjovsky, Bottou. [Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862), 2017 · Arjovsky et al. [WGAN](https://arxiv.org/abs/1701.07875), 2017 · Gulrajani et al. [WGAN-GP](https://arxiv.org/abs/1704.00028), 2017.
- Ho, Jain, Abbeel. [DDPM](https://arxiv.org/abs/2006.11239), 2020 · Song et al. [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456), 2021 · Vincent. [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf), 2011.
- Heusel et al. [FID](https://arxiv.org/abs/1706.08500), 2017 · Kynkäänniemi et al. [Improved Precision and Recall Metric](https://arxiv.org/abs/1904.06991), 2019.
- Villani. Optimal Transport: Old and New (двойственность Канторовича–Рубинштейна).

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL, лекция 12: генеративные модели
- **Семинары:** [Семинар: GAN, WGAN-GP](../../materials/seminars/27_Семинар_GAN_WGAN.ipynb) ([Colab](https://colab.research.google.com/drive/1VV4cMQayOQDfYOR_uSH32Zo1OFTzB1Pz))
- **Видео и внешние курсы:** [Visual GenAI Course 2026 (диффузия)](https://github.com/dbaranchuk/diffusion-course-and-seminars/tree/main/visual_genai_course_2026) · [Stanford CS231n — CNN for Visual Recognition](https://cs231n.stanford.edu/)

</details>

---

[← 26. Работа с LLM: промптинг, RAG, агенты, запуск и оценка](../26_llm_rag_agents/README.md) · [Оглавление курса](../../README.md) · [28. Vision Transformer, self-supervised и мультимодальные модели →](../28_vit_ssl_multimodal/README.md)
