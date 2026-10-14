# 27. Генеративные модели — математика и выводы

[← Конспект темы](README.md) · [Шпаргалка](../../cheatsheet.ipynb) · [Карта курса](../../README.md)

---

## 1. ELBO: два вывода

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

---

## 2. KL двух гауссиан в замкнутой форме

Для одномерных $q = \mathcal N(\mu, \sigma^2)$, $p = \mathcal N(0, 1)$:

$$
\mathrm{KL}(q\,\|\,p) = \mathbb E_q\Big[\log\frac{q}{p}\Big] = \mathbb E_q\Big[-\ln\sigma - \frac{(z - \mu)^2}{2\sigma^2} + \frac{z^2}{2}\Big] = -\ln\sigma - \frac12 + \frac{\mu^2 + \sigma^2}{2},
$$

т.е. $\frac12(\mu^2 + \sigma^2 - \ln\sigma^2 - 1)$; для диагональной ковариации — сумма по координатам. Минимум 0 при $\mu = 0$, $\sigma = 1$: регуляризатор
тянет коды к центру и не даёт дисперсиям схлопнуться в 0 (иначе VAE выродился бы в обычный автокодировщик с «точечными» кодами).
Общий случай: $\mathrm{KL}(\mathcal N_0\,\|\,\mathcal N_1) = \frac12\big[\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^{\top}\Sigma_1^{-1}(\mu_1 - \mu_0) - k + \ln\frac{\det\Sigma_1}{\det\Sigma_0}\big]$.

---

## 3. Репараметризация против REINFORCE

Нужен $\nabla_\phi\,\mathbb E_{q_\phi(z)}f(z)$. **Score-function (REINFORCE)** оценка:
$\nabla_\phi\mathbb E_qf = \mathbb E_q\big[f(z)\nabla_\phi\log q_\phi(z)\big]$ — работает для любых (даже дискретных) $z$, но с огромной дисперсией: $f$ входит как скаляр-множитель.
**Репараметризация:** $z = g_\phi(\varepsilon) = \mu_\phi + \sigma_\phi\odot\varepsilon$, $\varepsilon\sim\mathcal N(0, I)$ — распределение $\varepsilon$ от $\phi$ не зависит, поэтому

$$
\nabla_\phi\,\mathbb E_{q_\phi}f(z) = \mathbb E_\varepsilon\big[\nabla_\phi f(\mu_\phi + \sigma_\phi\odot\varepsilon)\big] = \mathbb E_\varepsilon\Big[\nabla_zf\cdot\frac{\partial z}{\partial\phi}\Big],
$$

используется производная $f$ — оценка по одному сэмплу на объект уже пригодна для SGD. Для дискретных латентов — Gumbel-softmax или
straight-through (как в VQ-VAE: $z_q = z_e + \mathrm{sg}[z_q - z_e]$ — вперёд значение $z_q$, назад градиент тождественный).

---

## 4. Прямой и обратный KL: mean-seeking и mode-seeking

$\mathrm{KL}(\pi\,\|\,p) = \int\pi\log\frac{\pi}{p}$: где $\pi(x) > 0$, а $p(x)\to0$, подынтегральное выражение $\to+\infty$ — модель **обязана** покрыть все
области с данными, даже ценой массы между модами (**zero-avoiding**). Максимум правдоподобия ≡ минимум прямого KL — поэтому VAE и
авторегрессионные модели покрывают моды, но «размывают».

$\mathrm{KL}(p\,\|\,\pi) = \int p\log\frac{p}{\pi}$: где $\pi\approx0$, а $p > 0$, штраф огромен; где $p = 0$ — штрафа нет. Выгодно сосредоточиться на одной
моде (**zero-forcing**). GAN с JSD ведёт себя похоже — отсюда mode collapse. L1: одна гауссиана для смеси 0.8/0.2 — прямой KL даёт
$\mu = -1.2$, $\sigma = 1.68$, обратный — $\mu = \pm2$, $\sigma = 0.5$.

$\mathrm{JSD}(\pi, p) = \frac12\mathrm{KL}(\pi\,\|\,m) + \frac12\mathrm{KL}(p\,\|\,m)$, $m = \frac{\pi + p}{2}$ — симметрична и ограничена: $0\le\mathrm{JSD}\le\log2$.

---

## 5. GAN: оптимальный дискриминатор и JSD

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

---

## 6. Почему Вассерштейн лучше JSD

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

---

## 7. Диффузия: формулы DDPM

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
$\ln\bar\alpha_T\approx-\sum\beta_t\approx-10$. В L1 ($T = 100$, $\beta$ до 0.1) $\sum\beta\approx5$, $\bar\alpha_T\approx0.007$.

---

## 8. FID

Признаки Inception-v3 (пул-слой, 2048) для реальных и сгенерированных изображений аппроксимируются гауссианами $(\mu_r, \Sigma_r)$, $(\mu_g, \Sigma_g)$;
расстояние Фреше (Вассерштейна-2) между ними:

$$
\mathrm{FID} = \lVert\mu_r - \mu_g\rVert^2 + \operatorname{tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\big).
$$

Учитывает и качество, и разнообразие (mode collapse сужает $\Sigma_g$), но смещён при малых выборках (нужно ≥ 10–50 тыс. картинок) и зависит от
Inception (ImageNet-признаки). Поэтому дополняют precision/recall в пространстве признаков (Kynkäänniemi et al., 2019).

---

## Ссылки

- Kingma, Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), 2013 · Doersch. [Tutorial on VAE](https://arxiv.org/abs/1606.05908), 2016 · Jang et al. [Gumbel-Softmax](https://arxiv.org/abs/1611.01144), 2016.
- Goodfellow et al. [GAN](https://arxiv.org/abs/1406.2661), 2014 · Arjovsky, Bottou. [Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862), 2017 · Arjovsky et al. [WGAN](https://arxiv.org/abs/1701.07875), 2017 · Gulrajani et al. [WGAN-GP](https://arxiv.org/abs/1704.00028), 2017.
- Ho, Jain, Abbeel. [DDPM](https://arxiv.org/abs/2006.11239), 2020 · Song et al. [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456), 2021 · Vincent. [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf), 2011.
- Heusel et al. [FID](https://arxiv.org/abs/1706.08500), 2017 · Kynkäänniemi et al. [Improved Precision and Recall Metric](https://arxiv.org/abs/1904.06991), 2019.
- Villani. Optimal Transport: Old and New (двойственность Канторовича–Рубинштейна).
