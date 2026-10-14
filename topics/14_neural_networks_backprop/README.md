# 14. Нейронные сети и backpropagation

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Нейрон — линейная модель с нелинейной активацией: $\sigma(\langle w,x\rangle+b)$. Слои нейронов последовательно
преобразуют признаковое пространство; один нейрон не решает XOR, два слоя — решают, а двухслойная сеть приближает
любую непрерывную функцию (но может требовать очень много нейронов). Учим SGD по мини-батчам; градиент считает
**backpropagation** — цепное правило от лосса к входу на графе вычислений.

| Термин | Определение |
|---|---|
| Активация | ReLU (по умолчанию), GELU (трансформеры), LeakyReLU/ELU (мёртвые нейроны); sigmoid/tanh — не в скрытых слоях |
| Логиты | сырые выходы последнего слоя; softmax/sigmoid — внутри `CrossEntropyLoss` / `BCEWithLogitsLoss` |
| FFN / MLP | полносвязная сеть прямого распространения |
| Батч / эпоха / итерация | группа объектов на один шаг / проход по выборке / один шаг |
| Прямой / обратный ход | вычисление и запоминание значений / градиенты от $L$ к входам |
| Vanishing gradient | произведение множителей $<1$ по слоям → ранние слои не учатся |
| Мёртвый нейрон | ReLU, выдающий 0 на всех объектах |

$$
\delta^{(l)}=\big(W^{(l+1)\top}\delta^{(l+1)}\big)\odot f'(z^{(l)}),\qquad
\frac{\partial L}{\partial W^{(l)}}=\delta^{(l)}a^{(l-1)\top},\qquad
\frac{\partial}{\partial a_k}\Big(-\ln\operatorname{softmax}(a)_y\Big)=p_k-[k=y]
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>0. О курсе Deep Learning</b></summary>

12 учебных дней: каждый день лекция + семинар (что разобрали — то же пишем кодом). Маршрут: **основы → компьютерное
зрение → последовательности, текст и генеративные модели**. Нужны Python и numpy, производные и матричные операции,
базовое ML (регрессия, классификация, градиентный спуск); на семинарах — PyTorch (Colab или локально).

</details>

<details>
<summary><b>1. Нейрон: линейные модели — это нейросети</b></summary>

$$
a(x) = \sigma\Big(\sum_{j} w_j x_j + b\Big) = \sigma(\langle w, x\rangle + b),
$$

$w$ — веса связей, $b$ — смещение (bias), $\sigma$ — функция активации.
- тождественная активация → **линейная регрессия**;
- пороговая $\operatorname{th}(z) = [z > 0]$ → **линейный классификатор** (перцептрон);
- сигмоида → **логистическая регрессия**.

**Функции активации**

| Где | Что брать |
|---|---|
| Скрытые слои (по умолчанию) | **ReLU** $\max(0, z)$ — быстрая, работает в полносвязных и свёрточных сетях |
| Трансформеры | **GELU** $z\,\Phi(z)$ — гладкий ReLU (темы 23–24) |
| Нейроны «умирают» (почти все выходы 0) | **LeakyReLU** $\max(0.01z, z)$ или **ELU** |
| Sigmoid, tanh в скрытых слоях | **не ставить**: насыщаются, градиент затухает (исключение — гейты LSTM, там по построению) |

| Выходной слой (диктует задача) | Активация + функция потерь |
|---|---|
| Регрессия | ничего + `MSELoss` |
| Да / нет | логит + `BCEWithLogitsLoss` |
| Один класс из $K$ | логиты + `CrossEntropyLoss` |
| Несколько меток | логиты + `BCEWithLogitsLoss` |

**Логиты** — сырые выходы последнего слоя; сигмоида и softmax уже **внутри** этих функций потерь, в устойчивом виде.
Типовая ошибка — поставить softmax перед `CrossEntropyLoss`: сеть обучится, но хуже.
Разница между ReLU, GELU и Swish — меньше процента; данные, темп обучения и нормализация решают несравнимо больше.

Ещё активации: Softplus $\ln(1 + e^z)$, SELU, Swish/SiLU $z\sigma(z)$, Maxout. Таблица с производными — см. раздел «2. Функции активации и их производные».

**Softmax** — обобщение сигмоиды на $K$ классов: $p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$, выходы положительны и в сумме дают 1.

```python
import torch
from torch.nn.functional import softmax
softmax(torch.tensor([1., 2., 3.]), dim=0)    # указать размерность! -> [0.09, 0.24, 0.67]
```

</details>

<details>
<summary><b>2. Что может и чего не может один нейрон</b></summary>

- **Может:** логическое И, ИЛИ (с пороговой активацией) — это линейно разделимые функции.
- **Не может:** исключающее ИЛИ (XOR) и эквивалентность — точки не разделить одной прямой (доказательство — см. раздел «1. Нейрон и что он умеет»).
- **Три нейрона решают эквивалентность:** $h_1$ = И (обе единицы), $h_2$ = НЕ-ИЛИ (оба нуля), $a$ = ИЛИ над ними.
  Два нейрона скрытого слоя режут плоскость двумя прямыми, третий складывает куски. Скрытый слой **строит признаки**,
  в которых задача уже линейно разделима.

| $x_1$ | $x_2$ | $h_1$ | $h_2$ | $a$ |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 0 | 1 |

**История:** 1958 — перцептрон Розенблатта (пресса обещала машину, которая «научится ходить, говорить, видеть»);
1969 — Минский и Паперт, «Perceptrons»: один слой не решает XOR → первая «зима ИИ» (~1974–1980);
1986 — Rumelhart, Hinton, Williams популяризуют **backpropagation**. Мешала не теорема (двуслойная сеть XOR решает),
а **не было способа обучать скрытый слой**.

</details>

<details>
<summary><b>3. Многослойные сети</b></summary>

**Теорема об универсальной аппроксимации (Hornik, 1991)**

Любую непрерывную функцию на компакте можно с любой точностью приблизить сетью **глубины 2** с неполиномиальной
активацией на скрытом слое и линейным выходом. **Но:** нейронов может понадобиться очень много (неизвестно сколько),
веса — экспоненциально большие, обучение — сложное.

**Зачем глубже?** Теорема гарантирует существование, но не размер. Глубина позволяет обходиться **экспоненциально
меньшим** числом нейронов для некоторых функций (пример Telgarsky — см. раздел «3. Универсальная аппроксимация и зачем глубина»).

**Сеть прямого распространения (FFN / MLP)**

Все нейроны предыдущего слоя связаны с нейронами следующего: входной слой → один или несколько скрытых → выходной.
(Термин MLP исторически — с пороговыми активациями.) Сеть — **ориентированный граф вычислений**: вершины — переменные/
нейроны, рёбра — зависимости.

**Глубокая сеть — последовательное преобразование признакового пространства**: каждый слой строит новое представление.
«Сейчас наука DL — в основном о том, как правильно представлять (преобразовывать) признаковые пространства».

```python
import torch

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

net = Feedforward(3, 5)
net(torch.tensor([1., 2., 3.]))     # tensor([0.53...], grad_fn=<SigmoidBackward0>)
```

**Зачем большие сети**

- *The Loss Surfaces of Multilayer Networks* (Choromanska, LeCun et al., 2015): сеть с одним скрытым слоем обучили 1000 раз
  с разной инициализацией — **чем больше сеть, тем уже разброс** test loss: плохие локальные минимумы у больших сетей почти не встречаются.
- *Scaling Laws* (Kaplan et al., 2020): большие модели **эффективнее используют каждый пример** — до того же loss доходят за
  меньшее число токенов (при условии, что данных достаточно; на маленьком датасете с многими эпохами большая модель переобучится).
- Рецепт глубокого обучения: **много слоёв + много данных + достаточно вычислений**.

</details>

<details>
<summary><b>4. Обучение</b></summary>

Минимизация регуляризованного эмпирического риска — задача **невыпуклая**:

$$
\frac{1}{\ell}\sum_{i=1}^{\ell}L\big(y_i, a(x_i, w)\big) + \lambda R(w) \to \min_w .
$$

**SGD:** 1) случайная инициализация весов (почему именно так — тема [15](../15_nn_training/README.md)); 2) цикл до сходимости:
случайный объект (мини-батч) → градиент → шаг $w \leftarrow w - \eta\nabla L$.

| Термин | Смысл |
|---|---|
| Обучающая выборка | по ней считаем градиент и двигаем веса |
| Валидационная | только проверяем качество, весами по ней не двигаем |
| **Батч** | небольшая группа объектов (32–256), по которой считается один градиент |
| **Эпоха** | один полный проход по обучающей выборке |
| **Темп обучения** $\eta$ | длина шага (тема 15) |

```python
for epoch in range(n_epochs):
    model.train()
    for X, y in train_loader:            # батч
        y_pred = model(X)                # прямой ход
        loss = criterion(y_pred, y)      # ошибка
        optimizer.zero_grad()            # обнуляем накопленные градиенты
        loss.backward()                  # обратный ход
        optimizer.step()                 # w ← w − η·∇L
    model.eval()
    evaluate(model, val_loader)          # после каждой эпохи, под torch.no_grad()
```

SGD — частный случай с батчем размера 1; батч даёт менее шумный градиент, и GPU считает его почти за то же время.

**Функция потерь для классификации — log loss (CrossEntropyLoss):** $L = -\ln p_y = -a_y + \ln\sum_j e^{a_j}$, где $a$ — логиты.
Наивная реализация неустойчива (сумма больших экспонент, разность близких чисел) → вычитают максимальный логит
(log-sum-exp — см. раздел «5. Softmax + кросс-энтропия: градиент и устойчивость»). Работает, когда верный класс ровно один;
несколько меток — сигмоиды и BCE.

</details>

<details>
<summary><b>5. Обратное распространение градиента (backpropagation)</b></summary>

Сеть — **суперпозиция** функций. Градиент по весам считаем по **цепному правилу** от лосса к первому слою, используя
значения, запомненные на прямом ходе:
- **прямой ход** (forward): считаем и **запоминаем** значения во всех вершинах;
- **обратный ход** (backward): идём от $L$ к входам; **вдоль пути производные перемножаются, разные пути складываются**;
  каждая производная берётся в своей точке с прямого хода.

**Частная производная сложной функции:** если $z = f(u, v)$, а $u, v$ зависят от $y$, то
$\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial y} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial y}$.
Пример: $f = (x + y)(y - z)$, $x, y, z = 1, 2, 3$: через $g = x + y$ вклад $h\cdot1 = -1$, через $h = y - z$ вклад $g\cdot1 = 3$,
итого $\partial f/\partial y = 2$.

```python
import torch
x = torch.tensor([1.], requires_grad=True)
y = torch.tensor([2.], requires_grad=True)
z = torch.tensor([3.], requires_grad=True)
f = (x + y) * (y - z)        # прямой проход — строится граф (AddBackward0, SubBackward0, MulBackward0)
f.backward()                 # обратный проход
x.grad, y.grad, z.grad       # (tensor([-1.]), tensor([2.]), tensor([-3.]))
# from torchviz import make_dot; make_dot(f)  — визуализация графа
```

**Backprop руками (лекция):** $x = (1, 2)$, два скрытых слоя ReLU по 2 нейрона, линейный выход, $L = \frac12(\hat y - 3)^2$.
Прямой ход: $a = (3, 1)$, $b = (2, 1)$, $\hat y = 4$, $L = 0.5$. Обратный ход даёт
`W1.grad = [[3, 6], [-5, -10]]`, `W2.grad = [[3, 1], [6, 2]]`, `v.grad = [2, 1]` — ровно как в PyTorch
(все шаги — см. раздел «4. Backpropagation»). Все веса обновляются разом по градиентам этого прохода.

```python
x  = torch.tensor([1., 2.])
W1 = torch.tensor([[1., 1.], [-1., 1.]], requires_grad=True)
W2 = torch.tensor([[1., -1.], [1., -2.]], requires_grad=True)
v  = torch.tensor([1., 2.], requires_grad=True)
loss = 0.5 * (v @ torch.relu(W2 @ torch.relu(W1 @ x)) - 3) ** 2
loss.backward()
W1.grad, W2.grad, v.grad
```

`loss.backward()` делает те же шаги для любой сети: в каждой вершине умножает пришедший градиент на локальную производную.
Матричные формулы ($\delta^{(l)} = (W^{(l+1)\top}\delta^{(l+1)})\odot f'(z^{(l)})$) и сравнение с другими способами дифференцирования —
см. раздел «4. Backpropagation».

</details>

<details>
<summary><b>6. Проблема затухания градиента</b></summary>

- При прохождении через нейрон градиент умножается на производную активации. **Производная сигмоиды в насыщении ≈ 0**
  (и не больше 0.25 вообще) → в глубокой сети градиент до первых слоёв почти не доходит.
- **ReLU** решает это: быстро считается, производная 1 на активной части, разреженные активации. Минусы: в нуле нет
  производной (берут 0), выходы не центрированы, **«мёртвые» нейроны** — выдают 0 на любом объекте выборки (вероятность
  мала, при обучении их число обычно уменьшается, но в глубоких сетях с ReLU это бывает проблемой).
- Борьба с затуханием/взрывом в целом — тема [15](../15_nn_training/README.md).

</details>

<details>
<summary><b>7. Итоги и проблемы нейросетей</b></summary>

НС — нелинейное обобщение линейных алгоритмов; последовательное преобразование признакового пространства; «ансамбль»
(суперпозиция логистических регрессий); граф вычислений с высокой функциональной выразимостью; обучение градиентными методами.

Проблемы: большая сложность модели → **переобучение**; **нестабильность** обучения; большое время обучения, много данных.

Поиграйте: [TensorFlow Playground](http://playground.tensorflow.org/).

</details>

**Математика и выводы**

<details>
<summary><b>1. Нейрон и что он умеет</b></summary>

Нейрон: $a(x) = \sigma\big(\langle w, x\rangle + b\big)$. С пороговой активацией $\operatorname{th}(z) = [z > 0]$ это линейный классификатор.

**И:** $w = (1, 1)$, $b = -1.5$: $x_1 + x_2 - 1.5 > 0$ только при $x_1 = x_2 = 1$.
**ИЛИ:** $w = (1, 1)$, $b = -0.5$. **НЕ-ИЛИ:** $w = (-1, -1)$, $b = 0.5$.

**XOR одним нейроном невозможен.** Пусть $\operatorname{th}(w_1x_1 + w_2x_2 + b)$ реализует XOR. Тогда
$(0,0) \to 0$: $b \le 0$; $(1,1) \to 0$: $w_1 + w_2 + b \le 0$; $(1,0) \to 1$: $w_1 + b > 0$; $(0,1) \to 1$: $w_2 + b > 0$.
Сложим два последних: $w_1 + w_2 + 2b > 0$, т.е. $w_1 + w_2 + b > -b \ge 0$ — противоречие со вторым условием. ∎
(Геометрически: точки XOR не разделяются одной прямой.)

**Эквивалентность (¬XOR) тремя нейронами** (слайд лекции): $h_1 = \text{И}(x_1, x_2)$, $h_2 = \text{НЕ-ИЛИ}(x_1, x_2)$, $a = \text{ИЛИ}(h_1, h_2)$.
Скрытый слой строит признаки $h_1, h_2$, в которых задача **уже линейно разделима** — это и есть роль скрытых слоёв.

</details>

<details>
<summary><b>2. Функции активации и их производные</b></summary>

| Активация | $f(z)$ | $f'(z)$ | Замечания |
|---|---|---|---|
| Сигмоида | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma(1 - \sigma) \le \frac14$ | насыщается, выход не центрирован |
| tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$ | $1 - \tanh^2 z \le 1$ | центрирована, но насыщается |
| ReLU | $\max(0, z)$ | $[z > 0]$ (в 0 берут 0) | быстрая, разреженная, «мёртвые» нейроны |
| LeakyReLU | $\max(\alpha z, z)$, $\alpha \approx 0.01$ | $1$ или $\alpha$ | нет мёртвых нейронов |
| ELU | $z$ при $z > 0$, $\alpha(e^z - 1)$ иначе | $1$ или $\alpha e^z$ | гладкая слева, среднее ближе к 0 |
| SELU | $\lambda\,\text{ELU}_\alpha(z)$, $\lambda \approx 1.0507$, $\alpha \approx 1.6733$ | | самонормализующиеся сети |
| Softplus | $\ln(1 + e^z)$ | $\sigma(z)$ | гладкий ReLU |
| GELU | $z\,\Phi(z)$ | $\Phi(z) + z\varphi(z)$ | трансформеры (BERT, GPT) |
| Swish / SiLU | $z\,\sigma(\beta z)$ ($\beta = 1$ — SiLU) | $\sigma(\beta z) + \beta z\,\sigma(\beta z)(1 - \sigma(\beta z))$ | немного лучше ReLU в глубоких сетях |
| Maxout | $\max_k(\langle w_k, x\rangle + b_k)$ | | обобщает ReLU, но больше параметров |
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | $\frac{\partial p_k}{\partial z_j} = p_k([k = j] - p_j)$ | выход классификатора; обобщение сигмоиды |

Softmax от $(z, 0)$ — это $(\sigma(z), 1 - \sigma(z))$, поэтому сигмоида — частный случай softmax для двух классов.
Сигмоида при умножении аргумента на большое число стремится к пороговой функции ($\sigma(kz) \to [z > 0]$), но, в отличие от
неё, дифференцируема — её можно учить градиентом.

**Почему нужна нелинейность.** Композиция линейных отображений линейна: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$.
Сеть любой глубины без нелинейностей — просто линейная модель.

</details>

<details>
<summary><b>3. Универсальная аппроксимация и зачем глубина</b></summary>

**Теорема (Cybenko 1989, Hornik 1991, Leshno et al. 1993).** Пусть $f$ — непрерывна на компакте $K \subset \mathbb{R}^d$, $\sigma$ —
непрерывная **неполиномиальная** функция. Тогда для любого $\varepsilon > 0$ существуют $N$ и параметры, что

$$
\sup_{x\in K}\Big|f(x) - \sum_{j=1}^{N}v_j\,\sigma(\langle w_j, x\rangle + b_j)\Big| < \varepsilon .
$$

*Идея для $d = 1$ и сигмоиды.* $\sigma(k(x - t))$ при больших $k$ — почти ступенька в точке $t$. Разность двух ступенек —
«прямоугольник» на $[t_1, t_2]$. Сумма прямоугольников с высотами $f(t_j)$ — кусочно-постоянное приближение $f$, сколь угодно
точное при мелком разбиении (равномерная непрерывность $f$ на компакте).

**Чего теорема не говорит:** сколько нужно нейронов (может быть экспоненциально много), какие получатся веса (могут быть
огромными), как их найти обучением.

**Преимущество глубины** (Telgarsky, 2016). «Зуб» $g(x) = 2\min(x, 1 - x)$ на $[0, 1]$ реализуется двумя ReLU. Его $k$-кратная
композиция $g^{(k)}$ имеет $2^{k}$ линейных кусков, а сеть глубины $O(k)$ использует $O(k)$ нейронов. Любая сеть глубины 2
должна иметь не меньше $\sim 2^{k}$ нейронов, чтобы приблизить $g^{(k)}$ (каждый ReLU-нейрон добавляет не больше одного излома).
Глубокие сети представляют некоторые функции **экспоненциально компактнее** — интуиция «последовательного
преобразования признакового пространства» (слайд лекции).

</details>

<details>
<summary><b>4. Backpropagation</b></summary>

**4.1. Цепное правило на графе**

Если $f$ зависит от $x$ через промежуточные переменные $u_1, \dots, u_k$, то

$$
\frac{\partial f}{\partial x} = \sum_{j=1}^{k}\frac{\partial f}{\partial u_j}\frac{\partial u_j}{\partial x} :
$$

вдоль пути производные **перемножаются**, по разным путям — **складываются**. Пример со слайда: $f = (x + y)(y - z)$ при
$x = 1, y = 2, z = 3$; $g = x + y = 3$, $h = y - z = -1$; $\frac{\partial f}{\partial y} = h\cdot1 + g\cdot1 = -1 + 3 = 2$, $\frac{\partial f}{\partial x} = h = -1$,
$\frac{\partial f}{\partial z} = -g = -3$ — ровно `y.grad = 2`, `x.grad = -1`, `z.grad = -3` в PyTorch.

**4.2. Матричная форма для MLP**

Слои $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$, $a^{(l)} = f(z^{(l)})$, $a^{(0)} = x$, $L = L(a^{(n)}, y)$. Обозначим $\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$ (вектор-столбец). Тогда:

$$
\delta^{(n)} = \nabla_{a^{(n)}}L \odot f'(z^{(n)}), \qquad
\delta^{(l)} = \big(W^{(l+1)\top}\delta^{(l+1)}\big)\odot f'(z^{(l)}),
$$

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)}a^{(l-1)\top}, \qquad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)} .
$$

*Вывод.* $z_i^{(l+1)} = \sum_j W_{ij}^{(l+1)}a_j^{(l)} + b_i$ и $a_j^{(l)} = f(z_j^{(l)})$, поэтому $z_j^{(l)}$ влияет на $L$ через все $z_i^{(l+1)}$:
$\delta_j^{(l)} = \sum_i\delta_i^{(l+1)}W_{ij}^{(l+1)}f'(z_j^{(l)})$ — это и есть $(W^\top\delta)\odot f'$. Для весов: $\frac{\partial z_i^{(l)}}{\partial W_{ij}^{(l)}} = a_j^{(l-1)}$. Для батча
из $B$ объектов $\frac{\partial L}{\partial W} = \frac{1}{B}\Delta^{(l)\top}A^{(l-1)}$ (суммы по батчу — одно матричное умножение).

Два «потока» на каждом нейроне (слайд «прохождение градиента через гейты»): градиент для **распространения дальше**
($W^\top\delta$) и для **корректировки весов** ($\delta a^\top$). При прохождении через нейрон градиент умножается на $f'(z)$.

**4.3. Пример с лекции (2 скрытых слоя, ReLU)**

$x = (1, 2)$, $y = 3$, $W_1 = \begin{pmatrix}1 & 1\\-1 & 1\end{pmatrix}$, $W_2 = \begin{pmatrix}1 & -1\\1 & -2\end{pmatrix}$, $v = (1, 2)$, $L = \frac12(\hat y - y)^2$.
Прямой ход: $a = (3, 1)$, $b = (2, 1)$, $\hat y = 4$, $L = 0.5$. Обратный: $\partial L/\partial\hat y = 1$; $\partial L/\partial v = b = (2, 1)$;
$\partial L/\partial b = v = (1, 2)$; $\partial L/\partial W_2 = (\partial L/\partial b)\,a^\top = \begin{pmatrix}3 & 1\\6 & 2\end{pmatrix}$; $\partial L/\partial a = W_2^\top(1, 2)^\top = (3, -5)$;
$\partial L/\partial W_1 = (3, -5)^\top x^\top = \begin{pmatrix}3 & 6\\-5 & -10\end{pmatrix}$. Шаг с $\eta = 0.1$: $w_{14} = 1 - 0.1\cdot(-10) = 2$.

**4.4. Сложность и автодифференцирование**

| Способ | Идея | Стоимость градиента $f: \mathbb{R}^n\to\mathbb{R}$ | Проблемы |
|---|---|---|---|
| Конечные разности | $\frac{f(w + h e_i) - f(w)}{h}$ | $n + 1$ вычислений $f$ | погрешность, дорого — только для проверки (gradcheck) |
| Символьное | формулы производных | — | «взрыв выражений» |
| **Forward mode AD** | несём производные вместе со значениями (дуальные числа) | $n$ проходов | выгодно при малом числе входов |
| **Reverse mode AD** = backprop | прямой проход с сохранением значений, обратный — vector-Jacobian products | **$\approx$ 2–3 прямых прохода** | память на хранение активаций |

**Принцип «дешёвого градиента»:** градиент скалярной функции по миллионам параметров стоит лишь константу прямых проходов.
Цена — **память**: нужно хранить промежуточные значения прямого хода (отсюда gradient checkpointing — пересчитывать часть
активаций вместо хранения).

</details>

<details>
<summary><b>5. Softmax + кросс-энтропия: градиент и устойчивость</b></summary>

$L = -\ln p_y$, $p_k = \frac{e^{a_k}}{\sum_j e^{a_j}}$ ⇒ $L = -a_y + \ln\sum_j e^{a_j}$, и

$$
\frac{\partial L}{\partial a_k} = -[k = y] + \frac{e^{a_k}}{\sum_j e^{a_j}} = p_k - [k = y] .
$$

Градиент ограничен по модулю единицей — не взрывается и не затухает из-за насыщения softmax.

**Устойчивое вычисление (log-sum-exp):** при $a_k = 1000$ $e^{1000}$ переполняется, при $a_k = -1000$ получаем $\ln 0$. Вынесем
$m = \max_k a_k$:

$$
\ln\sum_k e^{a_k} = m + \ln\sum_k e^{a_k - m} ,
$$

все показатели $\le 0$ (нет переполнения), а слагаемое с $a_k = m$ равно 1 (сумма $\ge 1$, логарифм конечен). Поэтому в PyTorch
`CrossEntropyLoss` принимает **логиты**, а softmax внутри — ставить softmax перед ней ошибка (двойной softmax «сплющивает»
распределение, сеть учится хуже). Аналогично `BCEWithLogitsLoss` вместо `Sigmoid + BCELoss`:
$-y\ln\sigma(a) - (1-y)\ln(1-\sigma(a)) = \max(a, 0) - ya + \ln(1 + e^{-|a|})$.

</details>

<details>
<summary><b>6. Затухание и взрыв градиента</b></summary>

$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_n}\prod_{l=2}^{n}\frac{\partial h_l}{\partial h_{l-1}}$ — произведение $n - 1$ якобианов. Если их нормы ≈ $q$, градиент масштабируется как $q^{n}$:
$0.9^{50} \approx 0.005$ (затухание), $1.1^{50} \approx 117$ (взрыв).

- **Сигмоида:** $\sigma' \le \frac14$, каждый слой режет градиент минимум в 4 раза: через 10 слоёв — в $10^6$ раз. В насыщении
  ($|z|$ велико) $\sigma' \approx 0$.
- **ReLU:** производная 1 на активной части — градиент проходит без затухания; но при $z < 0$ путь обрывается. **Мёртвый
  нейрон** — выход 0 на всех объектах выборки ⇒ градиент по его весам всегда 0, он не восстановится (причины: большой шаг,
  сильно отрицательное смещение).
- Борьба: правильная инициализация, нормализация, residual-связи, клиппинг градиента, LSTM/GRU для RNN (темы
  [15](../15_nn_training/README.md), [16](../16_cnn/README.md), [20](../20_rnn/README.md)).

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему без нелинейности глубина бесполезна?</summary>

композиция линейных отображений линейна.

</details>

<details>
<summary>Зачем глубина, если хватает двух слоёв?</summary>

теорема не ограничивает размер; глубокие сети представляют функции экспоненциально компактнее.

</details>

<details>
<summary>Почему не ставят softmax перед `CrossEntropyLoss`?</summary>

он уже внутри (log-sum-exp); двойной softmax ухудшает обучение.

</details>

<details>
<summary>Чем плоха сигмоида в скрытых слоях?</summary>

$\sigma'\le\frac14$, насыщение → затухание градиента; выход не центрирован.

</details>

<details>
<summary>Сколько стоит backprop?</summary>

константа прямых проходов (reverse-mode AD), но нужна память на активации.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- XOR одним нейроном невозможен: условия $b\le0$, $w_1+b>0$, $w_2+b>0$, $w_1+w_2+b\le0$ противоречивы.
- Универсальная аппроксимация: сигмоиды → ступеньки → кусочно-постоянное приближение; нужна неполиномиальная активация.
- Устойчивая кросс-энтропия: $\ln\sum e^{a_k}=m+\ln\sum e^{a_k-m}$, $m=\max a$.
- Forward-mode AD выгоден при малом числе входов, reverse-mode (backprop) — при скалярном выходе и миллионах параметров.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
import torch

# Пример с лекции: f = (x + y)(y - z); пути к y складываются: -1 + 3 = 2
x, y, z = (torch.tensor([v], requires_grad=True) for v in (1., 2., 3.))
f = (x + y) * (y - z); f.backward()
print("df/dx, df/dy, df/dz =", x.grad.item(), y.grad.item(), z.grad.item())

# Backprop руками vs autograd: сеть 2-2-2-1 с ReLU (слайды «Backprop руками»)
xin = torch.tensor([1., 2.])
W1 = torch.tensor([[1., 1.], [-1., 1.]], requires_grad=True)
W2 = torch.tensor([[1., -1.], [1., -2.]], requires_grad=True)
v = torch.tensor([1., 2.], requires_grad=True)
a = torch.relu(W1 @ xin); b = torch.relu(W2 @ a); y_hat = v @ b
loss = 0.5 * (y_hat - 3) ** 2; loss.backward()
d_yhat = (y_hat - 3).item()                                # dL/dŷ = 1
d_b = d_yhat * v.detach() * (W2 @ a > 0)                   # через ReLU'
d_a = (W2.detach().T @ d_b) * (W1 @ xin > 0)
print("W1.grad", W1.grad.tolist(), "== вручную", torch.outer(d_a, xin).tolist())
print("W2.grad", W2.grad.tolist(), "== вручную", torch.outer(d_b, a.detach()).tolist())
```

Результат запуска:

```text
df/dx, df/dy, df/dz = -1.0 2.0 -3.0
W1.grad [[3.0, 6.0], [-5.0, -10.0]] == вручную [[3.0, 6.0], [-5.0, -10.0]]
W2.grad [[3.0, 1.0], [6.0, 2.0]] == вручную [[3.0, 1.0], [6.0, 2.0]]
```

```python
# Один нейрон не решает XOR, MLP с одним скрытым слоем — решает
import torch.nn as nn
torch.manual_seed(0)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]); Y = torch.tensor([0., 1., 1., 0.])
for name, model in [("один нейрон", nn.Linear(2, 1)),
                    ("MLP 2-8-1", nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)))]:
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(2000):
        loss = nn.functional.binary_cross_entropy_with_logits(model(X).squeeze(1), Y)  # логиты + BCEWithLogits
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"{name:12} предсказания: {(model(X).squeeze(1) > 0).int().tolist()}  loss={loss.item():.3f}")
```

Результат запуска:

```text
один нейрон  предсказания: [0, 0, 0, 0]  loss=0.693
MLP 2-8-1    предсказания: [0, 1, 1, 0]  loss=0.000
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Учебник ML Яндекса: [Полносвязные нейросети](https://education.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami) · [Метод обратного распространения ошибки](https://education.yandex.ru/handbook/ml/article/metod-obratnogo-rasprostraneniya-oshibki)
- [CS231n: Backprop notes](https://cs231n.github.io/optimization-2/) · [Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/) · курс [cs231n.stanford.edu](https://cs231n.stanford.edu/)
- Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/) · Goodfellow et al., [Deep Learning](http://www.deeplearningbook.org/)
- [Automatic differentiation in ML: a survey](https://arxiv.org/abs/1502.05767) · [Обзор функций активации (Jagtap, Karniadakis)](https://arxiv.org/pdf/2209.02681.pdf)
- [DL2 HSE, неделя 1 «DL 1 compressed»](https://github.com/thecrazymage/DL2_HSE/tree/main/week_01)
- [YouTube-канал Олега Стефанова](https://youtube.com/@stepoleggg)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Лекция_1_Нейронные_сети_и_backpropagation.pptx` (DEEP LEARNING, день 1, 56 слайдов).
- `Повторение.pptx` (день 11), слайды 1–15.

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Cybenko G. *Approximation by superpositions of a sigmoidal function* (1989); Hornik K. *Approximation capabilities of multilayer feedforward networks* (1991)
- Pinkus A. *Approximation theory of the MLP model in neural networks* — http://www2.math.technion.ac.il/~pinkus/papers/neural.pdf
- Telgarsky M. *Benefits of depth in neural networks* (2016)
- Rumelhart D., Hinton G., Williams R. *Learning representations by back-propagating errors* (1986)
- Baydin A. et al. *Automatic differentiation in machine learning: a survey* — https://arxiv.org/abs/1502.05767
- Choromanska A. et al. *The Loss Surfaces of Multilayer Networks* (2015); Kaplan J. et al. *Scaling Laws for Neural Language Models* (2020)
- Prince S. *Understanding Deep Learning* — https://udlbook.github.io/udlbook/ · Goodfellow I. et al. *Deep Learning* — http://www.deeplearningbook.org/

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL, лекция 1: нейросети и backpropagation · DL: повторение (обзор курса)
- **Видео и внешние курсы:** [Нейросеть, собранная в Factorio (YouTube)](https://youtu.be/Q4iPxj4Y7yI) · [Stanford CS231n — CNN for Visual Recognition](https://cs231n.stanford.edu/)

</details>

---

[← 13. Рекомендательные системы](../13_recommender_systems/README.md) · [Оглавление курса](../../README.md) · [15. Обучение нейросетей: инициализация, оптимизаторы, регуляризация, нормализация →](../15_nn_training/README.md)
