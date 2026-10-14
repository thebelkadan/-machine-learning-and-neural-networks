# 17. Семантическая сегментация

[Шпаргалка (ноутбук)](../../cheatsheet.ipynb) · [Оглавление курса](../../README.md)

**Суть.** Класс для **каждого пикселя**: выход сети — логиты $C\times H\times W$, маска — argmax по классам. Разрешение приходится ронять
(контекст, скорость), а потом возвращать: **энкодер-декодер**, апсемплинг (интерполяция, транспонированная свёртка), **skip-связи**
(FCN — сумма, **U-Net — конкатенация**), atrous-свёртки и ASPP (DeepLab). Метрики — **IoU / Dice / mIoU** (pixel accuracy врёт при
дисбалансе), loss — **CE + Dice**.

| Термин | Определение |
|---|---|
| Semantic / instance / panoptic | класс пикселя / маска каждого объекта / объекты + фон семантикой |
| IoU, Dice | $\frac{TP}{TP+FP+FN}$, $\frac{2TP}{2TP+FP+FN}$; $\text{Dice}=\frac{2\,\text{IoU}}{1+\text{IoU}}$ |
| mIoU | среднее IoU по классам; пересечения и объединения копим по датасету |
| Soft Dice loss | $1-\frac{2\sum py+\varepsilon}{\sum p+\sum y+\varepsilon}$ на вероятностях |
| Транспонированная свёртка | обучаемый апсемплинг; $k$ кратно $s$, иначе шахматные артефакты |
| Atrous (dilated) | дырки в ядре: больше рецептивное поле без потери разрешения |
| Output stride | во сколько раз карта признаков меньше входа (32 → 8 с atrous) |

$$
\text{IoU}=\frac{|P\cap G|}{|P\cup G|},\qquad \text{Dice}=\frac{2|P\cap G|}{|P|+|G|}=\frac{2\,\text{IoU}}{1+\text{IoU}},\qquad
H_{out}^{\text{ConvT}}=(H_{in}-1)s-2p+k+\mathrm{op}
$$

## Конспект

Разделы раскрываются по клику.

<details>
<summary><b>1. Постановка</b></summary>

**Сегментация** — класс для **каждого пикселя**: ответ — карта $H\times W$ с номерами классов (от 2 — объект/фон — до десятков; «фон» —
такой же класс).

| Задача | Что даёт |
|---|---|
| Классификация | один ярлык на картинку |
| Детекция | рамка + класс для каждого объекта (тема [18](../18_detection/README.md)) |
| **Семантическая сегментация** | класс каждого пикселя; объекты одного класса **не различает** (две машины — один цвет) |
| **Instance** | маска каждого объекта отдельно |
| **Panoptic** | объекты (*things*) по отдельности + фон (*stuff*: дорога, небо) семантикой |

**Где нужна:** важна **граница** (контур опухоли для облучения, замена фона в видеозвонке, дефект детали), **площадь** (посевы,
вырубки, пожары со спутника; доля поражённой ткани), **проходимость** (полоса и тротуар для автопилота, газон для робота, пол для AR).
Общее: нужна не рамка, а попиксельная форма.

**Что предсказывает сеть:** тензор логитов $C\times H\times W$; softmax по оси классов, argmax → маска $H\times W$. Разметка — та же карта с
номерами классов. Loss — кросс-энтропия, усреднённая по пикселям. Два класса — можно один канал и сигмоида. По сути
$H\times W$ задач классификации, но **не независимых** — соседние пиксели связаны, это и ловит CNN.

**Данные:** PASCAL VOC 2012 (21 класс), Cityscapes (30 классов, 5000 точных масок), ADE20K (150 классов), COCO-Stuff; медицинские —
свои, маленькие и закрытые. Точная маска кадра Cityscapes — **~1.5 часа** работы разметчика (грубая — 7 минут; рамка для детекции —
секунды) → данных всегда мало, разметка шумная на границах → важны аугментации и предобученный бэкбон.

**Дисбаланс классов:** в Cityscapes дорога и здания на **четыре порядка** чаще прицепа или светофора.

**Наивный подход — классифицировать окно вокруг каждого пикселя** — столько прогонов, сколько пикселей, одни и те же свёртки
пересчитываются, окно не видит контекста. Нужно считать признаки **один раз на всю картинку для всех пикселей** → полностью
свёрточная сеть.

</details>

<details>
<summary><b>2. Метрики</b></summary>

**Pixel accuracy врёт:** модель, не нашедшая ничего, набирает 96.9% (лекция) / 99.3% (семинар), потому что фон занимает почти весь кадр.

| Метрика | Формула | Смысл |
|---|---|---|
| Pixel accuracy | $\frac{TP + TN}{N}$ | не использовать как основную |
| **IoU** (Jaccard) | $\frac{\lvert P\cap G\rvert}{\lvert P\cup G\rvert} = \frac{TP}{TP + FP + FN}$ | стандарт натуральных сцен (VOC, Cityscapes) |
| **Dice** | $\frac{2\lvert P\cap G\rvert}{\lvert P\rvert + \lvert G\rvert} = \frac{2TP}{2TP + FP + FN}$ | = F1 по пикселям; стандарт медицины (MICCAI, V-Net) |
| **mIoU** | среднее IoU **по классам** | редкий класс весит как дорога |
| Frequency-weighted IU | среднее с весами частот | мягче, «красивее» |

- $\text{Dice} = \frac{2\,\text{IoU}}{1 + \text{IoU}}$ ⇒ Dice ≥ IoU; на одной маске метрики эквивалентны, **средние ранжируют по-разному**;
  не сравнивайте свой Dice с чужим IoU (вывод (см. раздел «1. IoU, Dice и их связь»)).
- **TN не входит** ни в IoU, ни в Dice — дисбаланс им не страшен.
- **mIoU честно:** копим пересечения и объединения **по всему датасету**, делим в конце; ignore/void-пиксели выбрасываем;
  на кадрах без класса IoU = 0/0 не определён — считать его нулём значит занизить метрику. Семинар: 4 способа усреднения
  одного предсказания дали 0.79 / 0.62 / 0.32 / 0.46 (почему (см. раздел «2. mIoU: способы усреднения дают разные числа»)).

```python
import torch

def iou_dice(pred, target, eps=1e-7):          # bool-маски [B, H, W]
    inter = (pred & target).sum((1, 2)).float()
    union = (pred | target).sum((1, 2)).float()
    size = pred.sum((1, 2)) + target.sum((1, 2))
    return inter / (union + eps), 2 * inter / (size + eps)

class MeanIoU:                                  # накопление по датасету
    def __init__(self, n_classes, ignore_index=255):
        self.inter = torch.zeros(n_classes); self.union = torch.zeros(n_classes); self.ign = ignore_index
    def update(self, pred, target):             # pred, target: [B, H, W] номера классов
        valid = target != self.ign
        for c in range(len(self.inter)):
            p, t = (pred == c) & valid, (target == c) & valid
            self.inter[c] += (p & t).sum(); self.union[c] += (p | t).sum()
    def compute(self):
        present = self.union > 0
        return (self.inter[present] / self.union[present]).mean().item()
```

</details>

<details>
<summary><b>3. Функции потерь</b></summary>

| Loss | Суть | Нюанс |
|---|---|---|
| **Кросс-энтропия** по пикселям | база | при дисбалансе почти не наказывает за потерянный мелкий объект; вероятности объекта «осторожные» |
| Взвешенная CE / `pos_weight` | вес класса ∝ 1/частота | для 0.7% объекта `pos_weight` ~100 |
| **Dice loss** $= 1 - \text{soft Dice}$ | оптимизирует ту величину, которой меряют; TN не входит | шумит на маленьких батчах, плохо, когда класса в батче нет |
| **Focal loss** | $-(1 - p_t)^\gamma\ln p_t$ — гасит вклад лёгких пикселей фона | из детекции (RetinaNet) |
| **CE + Dice** | на практике почти всегда сумма | |

**Эксперимент семинара** (объект — 0.7% кадра, U-Net, 14 эпох, тест при пороге 0.5):

| Loss | IoU | Dice | accuracy | макс. вероятность |
|---|---|---|---|---|
| BCE | **0.000** | 0.000 | 0.9926 | 0.443 |
| Dice | 0.600 | 0.750 | 0.9964 | 1.000 |
| BCE + Dice | 0.571 | 0.727 | 0.9959 | 1.000 |

BCE **нашёл** объект (на картах вероятностей видно пятно), но ни один пиксель не перешёл порог 0.5: BCE на сильном дисбалансе
систематически недооценивает вероятность объекта. С порогом 0.25 — IoU 0.485 на тесте. **Порог — гиперпараметр:** подбирать на
валидации, докладывать на тесте (0.5 редко оптимален; у Dice-моделей вероятности вытолкнуты к 0/1, и порог почти ничего не меняет).

```python
import torch.nn.functional as F

def bce_dice_loss(logits, target, eps=1.0):            # [B, 1, H, W]
    bce = F.binary_cross_entropy_with_logits(logits, target)     # BCE получает ЛОГИТЫ
    p = torch.sigmoid(logits)                                       # Dice — ВЕРОЯТНОСТИ (без порога!)
    inter = (p * target).sum((1, 2, 3))
    dice = (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + (1 - dice).mean()                                  # Dice по каждой картинке, потом среднее
```

</details>

<details>
<summary><b>4. Архитектуры: как вернуть разрешение</b></summary>

**Почему разрешение приходится ронять**

Свёртка $3\times3$, 64→64 канала на кадре $1024\times2048$ — $7.7\cdot10^{10}$ умножений на **один** слой (и память под активации). Без пулинга
рецептивное поле растёт на 2 пикселя за слой — чтобы увидеть полкадра, нужны сотни слоёв; а «дорога» и «тротуар» отличаются
только **контекстом**. Значит, ронять придётся; весь сюжет — как вернуть разрешение и не потерять границы.

**Полностью свёрточная сеть (FCN):** полносвязные слои классификатора заменяются свёртками $1\times1$ — выход карта $C\times H\times W$, вход любого
размера. **Энкодер** — обычный (предобученный) бэкбон; **декодер** — новая часть, возвращает разрешение.

**Способы апсемплинга**

| Способ | Обучаемые параметры | Особенность |
|---|---|---|
| Nearest | нет | «кубики» |
| Bilinear | нет | размытие; `align_corners` меняет результат |
| Max unpooling | нет | значения по запомненным индексам пулинга, остальное 0 (SegNet); нужен симметричный декодер |
| **Транспонированная свёртка** | да | каждый вход кладёт на выход ядро × значение; вклады суммируются |
| Bilinear + Conv | да (у свёртки) | без шахматных артефактов — частый выбор сейчас |

**ConvTranspose2d:** $H_{\text{out}} = (H_{\text{in}} - 1)s - 2p + k + \mathrm{op}$ (op — `output_padding`); удвоение — $k = 4, s = 2, p = 1$ или $k = 2, s = 2$; padding здесь
**отрезает** от выхода. **Шахматные артефакты:** $k$ должно делиться на $s$ ($k = 3, s = 2$ — регулярная сетка, рваная граница маски,
просадка IoU) (почему (см. раздел «4. Транспонированная свёртка»)).

```python
nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)       # ×2 без артефактов
nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(in_ch, out_ch, 3, padding=1))
```

**FCN, DeconvNet, SegNet**

- **FCN** (Long, Shelhamer, Darrell, CVPR 2015 — начало современной сегментации): **skip-связи** складывают предсказания с разных глубин.
  Глубокие слои знают, **что** на картинке (разрешение 1/32), ранние — **где** границы. VOC 2011: FCN-32s 59.4 → FCN-16s **62.4** (+pool4) →
  FCN-8s 62.7 (+pool3).
- **DeconvNet** (2015): декодер — зеркало VGG-16, unpooling + транспонированные свёртки; тяжёлый.
- **SegNet**: в декодере только unpooling по индексам + обычные свёртки; индексы — почти бесплатная память (встраиваемые устройства).
- Развилка: передавать декодеру **индексы** (дёшево, грубо) или **карты признаков** (дорого, точно) — второй путь = U-Net.

**U-Net (Ronneberger et al., MICCAI 2015)**

Симметричные энкодер и декодер, между ними — **конкатенация** карт признаков того же разрешения.
- **Конкатенация, а не сумма:** декодер сам решает, что взять из ранних и глубоких слоёв; цена — вдвое больше каналов на входе блока.
- В статье: свёртки без padding ($572 \to 388$, «copy and crop»); **повышенные веса на границах** между соседними клетками (иначе слипшиеся
  клетки — одно пятно); 30 размеченных снимков + **упругие деформации**.
- Сегодня U-Net собирают с готовым энкодером (ResNet, EfficientNet); собственный — если картинки не похожи на фото.
- Базовая архитектура для медицины и задач, где мало данных.

```python
import torch.nn as nn

class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                         nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        self.enc1, self.enc2 = DoubleConv(in_ch, base), DoubleConv(base, 2 * base)
        self.bottom = DoubleConv(2 * base, 4 * base)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = DoubleConv(4 * base, 2 * base)            # 2b (up) + 2b (skip) после cat
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = DoubleConv(2 * base, base)
        self.head = nn.Conv2d(base, out_ch, 1)                # логиты, без сигмоиды

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); b = self.bottom(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))   # skip-связь: конкатенация по каналам
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
```

**Что дают skip-связи (семинар):** на 14-й эпохе IoU 0.596 против 0.367 без skip — но это во многом **скорость сходимости**; к 28-й эпохе
0.625 против 0.550 (val), на тесте 0.618 против 0.527; ошибок в полосе ±2 пикселя от границы 12.3% против 16.4%; не найден объект на 14
против 20 картинок из 120. Урок: сравнивать архитектуры на сошедшихся моделях.

**Dilated (atrous) свёртки и DeepLab**

- **Atrous-свёртка:** «дырки» между весами — параметров и вычислений столько же, рецептивное поле больше, разрешение не падает.
- **Output stride:** после пяти пулингов карта в 32 раза мельче (маска «ватная», как FCN-32s). Убираем последние пулинги, ставим
  $d = 2, 4, 8$ → output stride 8, карта в 4 раза детальнее (цена — память и время).
- **ASPP:** одну карту пропускаем через несколько atrous-свёрток с разными $d$ и склеиваем — несколько масштабов сразу.
- **DeepLab:** v1–v2 — atrous, ASPP и **CRF** поверх выхода (подтягивал границы к краям яркости); v3 — ASPP с BN и image pooling, CRF
  убрали (хорошо обученной сети постобработка не нужна); **v3+** — лёгкий декодер и depthwise separable свёртки.
- Рекомендация: быстрый сильный бейзлайн на фото — **DeepLabV3+ с ResNet-50**; мало данных, не похожих на фото — **U-Net**.

**Что ещё**

- **PSPNet** (2017) — пирамидальный пулинг ($1\times1, 2\times2, 3\times3, 6\times6$) — несколько масштабов через пулинг.
- **HRNet** (2019) — параллельные ветки разного разрешения с постоянным обменом; высокое разрешение не теряется.
- **SegFormer** (2021), **Mask2Former** (2022) — трансформерный энкодер, маски как набор запросов.
- **SAM** (Segment Anything, 2023) — обучена на 1.1 млрд масок, выделяет объект по подсказке (точка, рамка, грубая маска), класса не
  предсказывает — инструмент разметки и заготовка для пайплайнов.

Линия за десять лет: научились возвращать разрешение → не терять его → перестали предсказывать пиксели по отдельности.

</details>

<details>
<summary><b>5. Аугментации: маска едет вместе с картинкой</b></summary>

- Геометрия: отражения, повороты, сдвиги, масштаб, кропы; фотометрия: яркость, контраст, шум (**маску не трогают**);
  **упругая деформация** — снимок и маска деформируются одним полем смещений.
- Важно: Геометрическое преобразование применяется к паре **синхронно**, одними и теми же параметрами. Семинар: при независимых «монетках»
  рассинхрон в 96 парах из 200, IoU маски с правильной — 0.039. Ошибка **молчит**: формы сходятся, loss падает, качество не растёт.
- Важно: **Маску интерполируем только nearest**: bilinear по номерам классов даёт несуществующие классы.
- Библиотеки (`albumentations`, `torchvision.transforms.v2`) объявляют маску отдельным типом цели — сами синхронизируют и берут nearest.

```python
import albumentations as A
aug = A.Compose([A.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0)), A.HorizontalFlip(), A.ElasticTransform(p=0.2),
                 A.RandomBrightnessContrast(), A.Normalize(), ])
out = aug(image=image, mask=mask)          # один бросок «монетки» на пару, маска — nearest
```

</details>

<details>
<summary><b>6. Что делать на практике</b></summary>

**Рабочий порядок:** готовая реализация (`segmentation_models_pytorch`) + предобученный энкодер; обучение на случайных кропах (батч больше,
аугментация бесплатно); loss **CE + Dice**; на валидации мерить **mIoU**, а не loss; порог подбирать на валидации.

**Типичные грабли:** билинейный ресайз маски; не та нормировка на инференсе; забытые ignore-пиксели; тюнинг архитектуры, когда проблема
в разметке границ; сравнение своего Dice с чужим IoU.

**Порядок улучшений:** данные и аугментации → функция потерь и порог → разрешение входа → и только потом архитектура.

**Где сегментация ломается:** ряды одинаковых объектов, близкие классы (диван/кресло), всадник и лошадь, кот за сеткой — это
неоднозначность разметки и редкие ракурсы, лечится данными.

**Главная мысль лекции:** сеть легко узнаёт, **что** перед ней, и с трудом — **где** именно это кончается.

**ДЗ 2: от тензоров к U-Net (Oxford-IIIT Pet)**

Животное vs фон, 556 фото (256 train / 100 val / 200 test), сетка $128\times160$:
оси тензоров (`[H, W, 3] → [3, H, W]` через `permute`, не `reshape`); многоклассовое декодирование (softmax по оси классов, argmax → `[B, H, W]`);
бинарное — один логит + сигмоида + порог (argmax по одному каналу всегда 0!); `DoubleConv`; U-Net с **билинейным** апсемплингом и
конкатенацией (каналы после `cat` = сумма); `prepare_pair` (картинка bilinear, маска nearest); loss **BCE + soft Dice** (Dice по каждой
картинке, потом среднее); метрика Dice по бинарным маскам (обе пустые → 1). Опыт 1: запомнить 4 картинки (проверка исправности),
опыт 2: 20 эпох с выбором лучшей эпохи по val, один раз — тест.

</details>

**Математика и выводы**

<details>
<summary><b>1. IoU, Dice и их связь</b></summary>

Для предсказанной маски $P$ и разметки $G$ (одного класса): $TP = |P\cap G|$, $FP = |P\setminus G|$, $FN = |G\setminus P|$.

$$
\text{IoU} = \frac{|P\cap G|}{|P\cup G|} = \frac{TP}{TP + FP + FN}, \qquad
\text{Dice} = \frac{2|P\cap G|}{|P| + |G|} = \frac{2TP}{2TP + FP + FN} .
$$

**Dice = F1 по пикселям:** $\text{prec} = \frac{TP}{|P|}$, $\text{rec} = \frac{TP}{|G|}$, $F_1 = \frac{2\,\text{prec}\cdot\text{rec}}{\text{prec} + \text{rec}} = \frac{2TP^2/(|P||G|)}{TP(|P| + |G|)/(|P||G|)} = \frac{2TP}{|P| + |G|}$.

**Связь.** Обозначим $a = TP$, $b = FP + FN$. Тогда $\text{IoU} = \frac{a}{a + b}$, $\text{Dice} = \frac{2a}{2a + b}$. Из первого $b = a\frac{1 - \text{IoU}}{\text{IoU}}$, подставляем:

$$
\text{Dice} = \frac{2a}{2a + a(1 - \text{IoU})/\text{IoU}} = \frac{2\,\text{IoU}}{1 + \text{IoU}}, \qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}} .
$$

Функция $x \mapsto \frac{2x}{1+x}$ монотонна и вогнута ⇒ на **одной** маске метрики ранжируют одинаково, $\text{Dice} \ge \text{IoU}$; IoU 0.5 = Dice 0.667,
«Dice 0.85» = IoU 0.74. **Средние** же ранжируют по-разному: из-за вогнутости Dice растягивает низ шкалы (IoU $0.1\to0.2$ даёт $+0.15$
Dice, IoU $0.9\to1.0$ — только $+0.05$), поэтому провальный кадр весит в среднем Dice больше. Пример лекции: модели A (IoU 0.5, 0.5) и
B (0.1, 0.95): mIoU 0.500 vs 0.525 (лучше B), mDice 0.667 vs 0.578 (лучше A).

**TN не входит ни в IoU, ни в Dice** — поэтому они не «ломаются» от дисбаланса, а pixel accuracy $\frac{TP + TN}{N}$ — ломается
(предсказание «всё фон» при объекте в 0.7% кадра: accuracy 0.993, IoU 0).

</details>

<details>
<summary><b>2. mIoU: способы усреднения дают разные числа</b></summary>

$\text{IoU}_c = \frac{\sum_{\text{img}}|P_c\cap G_c|}{\sum_{\text{img}}|P_c\cup G_c|}$ — **накопление по датасету** (стандарт VOC/Cityscapes), $\text{mIoU} = \frac1K\sum_c\text{IoU}_c$.
Усреднение IoU **по картинкам** — другое число, и на картинках без класса $\frac00$ не определено. Семинар (одно и то же
предсказание): $0/0 \to 1$ — 0.793; пропуск — 0.620; $0/0 \to 0$ — 0.318; накопление — **0.459**. Разброс больше, чем у большинства
архитектурных улучшений. Правило: копить пересечения и объединения в буферы, делить в конце; ignore-пиксели (void) исключать.

</details>

<details>
<summary><b>3. Функции потерь при дисбалансе</b></summary>

**Почему BCE «осторожничает».** BCE — собственное правило (тема [02](../02_classification_metrics/README.md)): минимум ожидаемого
штрафа — при истинной условной вероятности. Если признаков недостаточно, чтобы уверенно отличить пиксель объекта, оптимальный
прогноз близок к **априорной доле** объекта (0.7%), и вероятности на объекте остаются ниже 0.5. Семинар: максимальная вероятность BCE
на всём тесте — 0.443 ⇒ IoU@0.5 = 0, но с порогом 0.25 — 0.48. Модель **нашла** объект, «неправ» порог.

**Soft Dice loss** (Milletari et al., V-Net, 2016): подставляем вероятности $p_i$ вместо бинарной маски:

$$
L_{\text{Dice}} = 1 - \frac{2\sum_i p_iy_i + \varepsilon}{\sum_i p_i + \sum_i y_i + \varepsilon} .
$$

Градиент по $p_j$ (обозначим $I = \sum p_iy_i$, $U = \sum p_i + \sum y_i$):
$\frac{\partial L}{\partial p_j} = -\frac{2y_jU - 2I}{U^2}$. Он нормирован на размер объекта — потеря мелкого объекта стоит столько же,
сколько крупного; фон (TN) в знаменатель не входит. Минусы: шумит на маленьких батчах; если класса в батче нет ($\sum y = 0$),
потеря сводится к $1 - \frac{\varepsilon}{\sum p + \varepsilon}$ и гонит все вероятности к нулю — поэтому Dice **не используют в одиночку**:
стандарт — **CE + Dice**. Считать Dice по каждой картинке, затем усреднять по батчу (сумма по всему батчу даст другое
взвешивание — крупные объекты сильнее влияют).

**Взвешенная CE:** вес класса $w_c \propto 1/\text{freq}_c$ (или `pos_weight` в BCE ≈ отношение фон/объект — для 0.7% это ~140).

**Focal loss** (Lin et al., 2017): $\text{FL}(p_t) = -\alpha_t(1 - p_t)^\gamma\ln p_t$, где $p_t$ — вероятность правильного класса. Множитель
$(1 - p_t)^\gamma$ ($\gamma \approx 2$) «гасит» вклад лёгких, уже хорошо классифицированных пикселей фона: при $p_t = 0.9$ вес падает в 100 раз,
при $p_t = 0.99$ — в $10^4$ раз. Основная масса градиента приходится на трудные примеры. Широко используется в детекции (RetinaNet).

</details>

<details>
<summary><b>4. Транспонированная свёртка</b></summary>

Свёртку можно записать как $y = Cx$ с разреженной матрицей $C$ (каждая строка — ядро в своей позиции). Её градиент по входу —
умножение на $C^\top$. **Транспонированная свёртка** — это прямое применение $C^\top$: каждый входной элемент «раскладывает» на выход
ядро, умноженное на своё значение, вклады складываются. Это **не обратная** операция к свёртке (название deconvolution неудачно),
а та же схема связей в обратную сторону.

Пример слайда: вход $\begin{pmatrix}0&1\\2&3\end{pmatrix}$, ядро $\begin{pmatrix}0&1\\2&3\end{pmatrix}$, шаг 1 → $\begin{pmatrix}0&0&1\\0&4&6\\4&12&9\end{pmatrix}$.

**Размер выхода** (обращение формулы свёртки):

$$
H_{\text{out}} = (H_{\text{in}} - 1)s - 2p + d(k - 1) + \mathrm{op} + 1 ,
$$

где $\mathrm{op}$ — `output_padding`. Удвоение: $k = 2, s = 2, p = 0$ или $k = 4, s = 2, p = 1$ (или $k = 3, s = 2, p = 1, op = 1$ — с артефактами). `output_padding` нужен, потому что
у свёртки с шагом несколько входных размеров дают один и тот же выход (округление вниз) — обратное отображение неоднозначно.
Веса `ConvTranspose2d` имеют форму $(C_{\text{in}}, C_{\text{out}}, k, k)$ — оси переставлены относительно `Conv2d`.

**Шахматные артефакты**

Каждый вход кладёт на выход окно $k\times k$, соседние входы отстоят на $s$. Число вкладов, получаемых выходным пикселем по одной оси,
зависит от его позиции по модулю $s$: если $s\mid k$, каждый внутренний пиксель покрыт ровно $k/s$ окнами (равномерно); иначе —
$\lfloor k/s\rfloor$ или $\lceil k/s\rceil$, чередуясь. В 2D это даёт шахматную решётку (при $k = 3, s = 2$ — 1, 2 или 4 вклада). Семинар: на
ровном входе с одинаковыми весами $k = 3, s = 2$ даёт центр с разбросом 1…4, $k = 4, s = 2$ — ровно 4. Лечение: $k$ кратно $s$ или
**интерполяция + обычная свёртка** (Odena et al., Distill 2016).

**Билинейная интерполяция:** значение в точке $(x, y)$ между узлами $(0,0),(1,0),(0,1),(1,1)$:
$f = f_{00}(1-x)(1-y) + f_{10}x(1-y) + f_{01}(1-x)y + f_{11}xy$ — без обучаемых параметров. **Маску интерполировать только nearest**:
линейная комбинация номеров классов даёт несуществующие классы (семинар: из классов {0, 3} билинейно получились 1 и 2 у 15 пикселей).

</details>

<details>
<summary><b>5. Atrous-свёртки и output stride</b></summary>

Atrous (dilated) свёртка с расширением $d$: $y[i] = \sum_u w[u]\,x[i + d\,u]$ — ядро $k$ «растянуто» до $d(k-1)+1$ без новых весов и без
потери разрешения. **Output stride** — во сколько раз карта признаков меньше входа (у классификационного ResNet — 32). DeepLab: убираем
последние понижения разрешения, а следующим свёрткам ставим $d = 2, 4, \dots$ — рецептивное поле прежнее, output stride 8, карта в 4 раза
детальнее (цена — память и время на большей карте).

**Gridding-проблема:** при одинаковых $d$ в нескольких слоях подряд выход зависит только от входов на решётке с шагом $d$ — часть пикселей
не используется. Лечат разными $d$ (например, 1, 2, 3 — Hybrid Dilated Convolution).

**ASPP:** параллельно atrous-свёртки с $d = 6, 12, 18$ (при output stride 16) + $1\times1$ + глобальный пулинг (image pooling), склейка по
каналам — несколько масштабов контекста сразу. PSPNet делает то же пирамидальным пулингом (сетки $1\times1, 2\times2, 3\times3, 6\times6$).

</details>

<details>
<summary><b>6. U-Net: веса на границах</b></summary>

Из статьи (Ronneberger et al., 2015): чтобы соприкасающиеся клетки не сливались, пиксели узких промежутков между объектами получают
больший вес в функции потерь:

$$
w(x) = w_c(x) + w_0\exp\Big(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\Big),
$$

$d_1, d_2$ — расстояния до ближайшей и второй ближайшей клетки, $w_c$ балансирует частоты классов, $w_0 = 10$, $\sigma \approx 5$ пикселей.
Свёртки без padding: вход $572\times572$ → выход $388\times388$; карты энкодера при склейке обрезаются по центру («copy and crop»);
предсказание большого изображения — перекрывающимися тайлами (overlap-tile). Упругие деформации как аугментация превращают
30 размеченных снимков в практически неограниченную выборку.

**Конкатенация vs сумма (FCN):** конкатенация даёт декодеру **выбор**, что взять из ранних и глубоких признаков (дальнейшие
свёртки — обучаемая смесь), сумма навязывает фиксированное смешивание и требует одинаковой размерности; цена конкатенации —
вдвое больше каналов на входе блока декодера.

</details>

<details>
<summary><b>Вопросы с собеседований</b></summary>

Сначала ответьте сами, потом раскройте ответ.

<details>
<summary>Почему pixel accuracy бесполезна на Cityscapes?</summary>

фон/дорога занимают почти весь кадр; TN «бесплатны».

</details>

<details>
<summary>Конкатенация U-Net vs сумма FCN?</summary>

декодер сам выбирает, что взять; цена — вдвое больше каналов.

</details>

<details>
<summary>`ConvTranspose2d(kernel_size=3, stride=2)`</summary>

шахматная решётка на маске; брать $k=4$ или интерполяцию + свёртку.

</details>

<details>
<summary>Почему Dice loss не используют в одиночку?</summary>

шумит на маленьких батчах, ломается без класса в батче → CE + Dice.

</details>

<details>
<summary>Маску ресайзят bilinear</summary>

несуществующие классы; только nearest. Аугментации — синхронно для пары.

</details>

</details>

<details>
<summary><b>Дополнительно: детали</b></summary>

- Dice — это F1 по пикселям; средние mIoU и mDice ранжируют модели по-разному (Dice растягивает низ шкалы).
- BCE — собственное правило: при дисбалансе оптимальные вероятности ≈ доле объекта → объект может не перейти порог 0.5.
- Транспонированная свёртка — умножение на $C^\top$ (градиент свёртки), не обратная операция.
- Output stride 8: убрать последние понижения разрешения и поставить dilation 2, 4 — поле то же, карта детальнее.

</details>

<details>
<summary><b>Примеры кода (с результатами запуска)</b></summary>

Запустить их можно в ноутбуке [`cheatsheet.ipynb`](../../cheatsheet.ipynb).

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# Pixel accuracy врёт: объект — 0.7% кадра, предсказание «всё фон»
target = torch.zeros(64, 64, dtype=torch.bool); target[30:35, 30:36] = True
for name, pred in [("всё фон", torch.zeros_like(target)), ("сдвиг на 2 px", torch.roll(target, 2, 1)), ("идеально", target)]:
    tp = (pred & target).sum().item(); fp = (pred & ~target).sum().item(); fn = (~pred & target).sum().item()
    acc = (pred == target).float().mean().item()
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0; dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    print(f"{name:14} accuracy={acc:.4f}  IoU={iou:.3f}  Dice={dice:.3f}  2IoU/(1+IoU)={2 * iou / (1 + iou):.3f}")

# Транспонированная свёртка: пример слайда и шахматные артефакты
x = torch.tensor([[[[0., 1.], [2., 3.]]]])
print("ConvT 2×2→3×3:", F.conv_transpose2d(x, x).squeeze().int().tolist())
ones = torch.ones(1, 1, 8, 8)
for k in (3, 4):
    y = F.conv_transpose2d(ones, torch.ones(1, 1, k, k), stride=2)[0, 0, k:-k, k:-k]
    print(f"k={k}, stride=2: число вкладов во внутренние пиксели от {int(y.min())} до {int(y.max())}")
```

Результат запуска:

```text
всё фон        accuracy=0.9927  IoU=0.000  Dice=0.000  2IoU/(1+IoU)=0.000
сдвиг на 2 px  accuracy=0.9951  IoU=0.500  Dice=0.667  2IoU/(1+IoU)=0.667
идеально       accuracy=1.0000  IoU=1.000  Dice=1.000  2IoU/(1+IoU)=1.000
ConvT 2×2→3×3: [[0, 0, 1], [0, 4, 6], [4, 12, 9]]
k=3, stride=2: число вкладов во внутренние пиксели от 1 до 4
k=4, stride=2: число вкладов во внутренние пиксели от 4 до 4
```

```python
# Мини-U-Net на синтетике с сильным дисбалансом: BCE против BCE + Dice
def make_data(n, rng):
    X = torch.rand(n, 1, 32, 32, generator=rng) * 0.3; Y = torch.zeros(n, 1, 32, 32)
    for i in range(n):
        cy, cx, r = torch.randint(4, 28, (2,), generator=rng).tolist() + [3]
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        Y[i, 0][m] = 1; X[i, 0][m] += 0.25
    return X, Y
g = torch.Generator().manual_seed(0)
Xtr, Ytr = make_data(256, g); Xte, Yte = make_data(64, g)
print(f"доля пикселей объекта: {Ytr.mean().item():.3%}")

def block(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU())
class TinyUNet(nn.Module):
    def __init__(s, b=8):
        super().__init__(); s.e1, s.e2, s.mid = block(1, b), block(b, 2 * b), block(2 * b, 4 * b)
        s.u2, s.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), block(4 * b, 2 * b)
        s.u1, s.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), block(2 * b, b); s.head = nn.Conv2d(b, 1, 1)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); m = s.mid(F.max_pool2d(e2, 2))
        d2 = s.d2(torch.cat([s.u2(m), e2], 1)); return s.head(s.d1(torch.cat([s.u1(d2), e1], 1)))

def soft_dice(logits, y, eps=1.0):
    p = torch.sigmoid(logits); inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()

for name, loss_fn in [("BCE", lambda z, y: F.binary_cross_entropy_with_logits(z, y)),
                      ("BCE + Dice", lambda z, y: F.binary_cross_entropy_with_logits(z, y) + soft_dice(z, y))]:
    torch.manual_seed(0); net = TinyUNet(); opt = torch.optim.Adam(net.parameters(), 3e-3)
    for epoch in range(6):
        for b in torch.randperm(256).split(32):
            loss = loss_fn(net(Xtr[b]), Ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); p = torch.sigmoid(net(Xte)).detach(); pred = p > 0.5; t = Yte.bool()
    iou = (pred & t).sum().item() / max((pred | t).sum().item(), 1)
    print(f"{name:11} IoU@0.5={iou:.3f}  max p на тесте={p.max().item():.2f}")
```

Результат запуска:

```text
доля пикселей объекта: 2.832%
BCE         IoU@0.5=0.810  max p на тесте=0.91
BCE + Dice  IoU@0.5=0.852  max p на тесте=0.99
```

</details>

<details>
<summary><b>Что почитать / посмотреть</b></summary>

- Статьи: [FCN](https://arxiv.org/abs/1411.4038) · [U-Net](https://arxiv.org/abs/1505.04597) · [SegNet](https://arxiv.org/abs/1511.00561) · [DeepLabV3](https://arxiv.org/abs/1706.05587) · [DeepLabV3+](https://arxiv.org/abs/1802.02611) · [SegFormer](https://arxiv.org/abs/2105.15203) · [Segment Anything](https://arxiv.org/abs/2304.02643) · [Focal Loss](https://arxiv.org/abs/1708.02002) · [V-Net](https://arxiv.org/abs/1606.04797)
- [Deconvolution and Checkerboard Artifacts (distill)](https://distill.pub/2016/deconv-checkerboard/) · [Convolution arithmetic](https://arxiv.org/abs/1603.07285)
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) · [albumentations](https://albumentations.ai/)
- [DL2 HSE, неделя 6: Segmentation and Detection](https://github.com/thecrazymage/DL2_HSE/tree/main/week_06) · [CS231n](https://cs231n.stanford.edu/)

</details>

<details>
<summary><b>Источники курса</b></summary>

- `Лекция_4_Семантическая_сегментация.pptx` (DEEP LEARNING, день 4, 44 слайда).
- `Семинар_4_Сегментация,_U_Net,_метрики_IoU_и_Dice.ipynb`: синтетические фигуры с дисбалансом, метрики руками, ловушки mIoU, апсемплинг (interpolate, MaxUnpool, ConvTranspose), шахматные артефакты, U-Net с нуля, BCE vs Dice vs BCE+Dice, подбор порога, абляция skip-связей, синхронные аугментации.
- `task_for_students_2.ipynb` — ДЗ-2 «От тензоров к U-Net» (Oxford-IIIT Pet).

</details>

<details>
<summary><b>Литература к выводам</b></summary>

- Long J., Shelhamer E., Darrell T. *Fully Convolutional Networks for Semantic Segmentation* (CVPR 2015)
- Ronneberger O., Fischer P., Brox T. *U-Net* (MICCAI 2015); Badrinarayanan V. et al. *SegNet* (TPAMI 2017)
- Chen L.-C. et al. *DeepLab* (TPAMI 2018); *Rethinking Atrous Convolution* (2017); *DeepLabV3+* (ECCV 2018)
- Zhao H. et al. *PSPNet* (2017); Wang J. et al. *HRNet* (TPAMI 2020); Xie E. et al. *SegFormer* (NeurIPS 2021); Kirillov A. et al. *Segment Anything* (ICCV 2023)
- Milletari F. et al. *V-Net* (Dice loss, 3DV 2016); Lin T.-Y. et al. *Focal Loss* (ICCV 2017)
- Odena A. et al. *Deconvolution and Checkerboard Artifacts* — https://distill.pub/2016/deconv-checkerboard/
- Dumoulin V., Visin F. *A guide to convolution arithmetic* — https://arxiv.org/abs/1603.07285
- Cordts M. et al. *Cityscapes* (CVPR 2016); Zhou B. et al. *ADE20K* (CVPR 2017)

</details>

<details>
<summary><b>Материалы курса и ссылки</b></summary>

- **Лекции курса** (файлы в репозиторий не включены): DL, лекция 4: семантическая сегментация
- **Семинары:** [Семинар 4: сегментация, U-Net, IoU и Dice](../../materials/seminars/17_Семинар_4_Сегментация_UNet.ipynb)
- **Домашние задания:** [DL, ДЗ 2: от тензоров к U-Net (задание)](../../materials/homework/17_DL_ДЗ_2_От_тензоров_к_UNet.ipynb)
- **Видео и внешние курсы:** [Stanford CS231n — CNN for Visual Recognition](https://cs231n.stanford.edu/) · [DL2 HSE (thecrazymage/DL2_HSE)](https://github.com/thecrazymage/DL2_HSE)

</details>

---

[← 16. Свёрточные сети (CNN)](../16_cnn/README.md) · [Оглавление курса](../../README.md) · [18. Детекция объектов →](../18_detection/README.md)
