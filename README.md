# W2 · CLIP zero-shot demo（P3）

不训练任何参数，用 OpenAI CLIP 的预训练权重在 CIFAR-10 测试集上做零样本分类，
并测出**「prompt 措辞」能带来多少准确率波动**。

上游计划：`W2详细计划（10.20-10.26）` D 节。本仓库对应产物 **P3**，也是 W2 三个 P 级产物的最后一个。

---

## 1. 结论先行

| 指标 | 实测值 |
| --- | --- |
| 最佳 prompt 准确率 | **89.5%**（`a photo of a {class}, a type of object`） |
| 随机基线（10 类） | 10.0% |
| 5 组 prompt 极差 | **5.4 个百分点**（84.1% ~ 89.5%） |
| 95% 置信区间 | ±0.60pp（n = 10000） |
| 类间最大落差 | **18pp**（frog 78% ↔ truck/ship/horse 96%） |

三句话总结：

1. **prompt 措辞真的影响结果**：极差 5.4pp，是 2 倍置信区间（1.2pp）的 4 倍以上，不是噪声。
2. **但「写得更详细更好」是错的**：手写的每类定制长句（84.1%）是 5 组里**最差**的，
   因为 CLIP 的文本塔是在短 caption 上训的，长句偏离训练分布。
3. **误差的大头不是 prompt，是类别本身**：prompt 只能动 5.4pp，而类别之间天然差 18pp。

> ⚠️ **89.5% 这个绝对分数不可外推。** CIFAR-10 是干净网图，很可能出现在 CLIP 的 4 亿图文对
> 预训练数据里（contamination）。本 demo 能支撑的是「相对排序」，不能支撑「CLIP 真实泛化能力」。

---

## 2. 环境

| 项 | 版本 / 说明 |
| --- | --- |
| Python | 3.10.19（conda 环境 `pytorch`） |
| PyTorch | 2.9.0+cu130，CUDA 13.0，RTX 5060 Laptop GPU |
| clip | `clip 1.0`（OpenAI 官方实现） |
| 权重 | `ViT-B/32`，151,277,313 参数，353,976,522 字节 |
| 数据 | 复用 W1 已下载的 CIFAR-10（`d:\learning project\W1 InfoNCE\data`），不重新下载 |

安装：

```powershell
$env:HTTPS_PROXY="http://127.0.0.1:7897"      # 本机代理端口是 7897，不是 7890
& "D:\anaconda\envs\pytorch\python.exe" -m pip install ftfy regex tqdm "git+https://github.com/openai/CLIP.git"
```

权重（约 338 MB，迅雷/aria2c 多线程比单连接快得多）：

```
https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

- 保存到 `weights/ViT-B-32.pt`（**文件名一字不差**，`clip.load` 只认这个名字）
- **不要**放 `~/.cache/clip`：那是 clip 硬编码的默认目录，指错不会报错，只会静默重下 350MB

---

## 3. 跑起来

```powershell
cd "d:\learning project\W2 CLIP demo"

# 默认：全测试集 10000 张，约 30 秒
& "D:\anaconda\envs\pytorch\python.exe" demo_clip.py

# 快速版：每类 50 张共 500 张，约 5 秒（但抽样噪声大，结论只能看趋势）
& "D:\anaconda\envs\pytorch\python.exe" demo_clip.py --n-per-class 50

# 没有 GPU 时
& "D:\anaconda\envs\pytorch\python.exe" demo_clip.py --device cpu
```

耗时构成（10000 张）：加载权重约 5s，图像编码 25s，5 组文本编码 <1s。

---

## 4. 期望输出（照这个对数字）

```
[load] ViT-B/32 ready in 4.98s on cuda
[data] 10000 PIL images (32x32)  labels (10000,)  class counts [1000]*10
[feat] image feats (10000, 512)  mean L2 norm 1.0000  (25.2s)
[eval] #1 a photo of a {class}                       acc =  88.8%
[eval] #2 a photo of a {class}, a type of object     acc =  89.5%
[eval] #3 {class} (bare word)                        acc =  87.4%
[eval] #4 a blurry photo of a {class}                acc =  87.8%
[eval] #5 per-class custom sentence                  acc =  84.1%
[ctrl] shuffled-label acc (#1 a photo of a {class}) = 9.7%  (expect ~10%)
[PASS] zero-shot demo done
```

末段 `SUMMARY` 应打印：

```
random baseline        : 10.0%
shuffled-label control : 9.7%  (must be ~10%)
best prompt            : #2 a photo of a {class}, a type of object
best acc               : 89.5% +/- 0.60pp (95% CI)
worst prompt           : #5 per-class custom sentence -> 84.1%
spread across prompts  : 5.4 pp (> 2x CI, meaningful)
gain over random       : 79.5 pp (8.9x)
worst classes          : ['frog', 'deer']
```

### 自检要点

| 检查 | 期望 | 不符意味着什么 |
| --- | --- | --- |
| `[load]` 是否出现 `Downloading` | 不出现 | `WEIGHTS_DIR` 指错目录了 |
| `mean L2 norm` | 1.0000 | 忘了 L2 归一化，余弦相似度失效 |
| `shuffled-label acc` | 约 10% | 若仍 80%+ = 索引/标签错位，预测与标签没对齐 |
| 某组 acc | 约 10% 附近 | 该组文本编码没生效（tokenize 后被 pad 淹没） |
| `[PASS]` | 出现 | 断言未过会非零退出 |

`shuffled-label` 是**阴性对照**：把标签随机打乱后准确率必须塌回 10%。
这是排除「管线 bug（索引错位、把训练集当测试集）」的唯一手段——
**没有阴性对照的高分，等于没有证据。**

---

## 5. 产物

```
runs/
├── W2_clip_1_prompt_acc.png      主图 · prompt 措辞对比（带 95% CI）
├── W2_clip_2_per_class_acc.png   副图 · 最佳 prompt 的逐类准确率
└── clip_results.json             全部数字（含 ci95_halfwidth、shuffled_label_acc）
```

一图一 claim（两个独立结论，拆成两个 PNG，不拼盘）：

**图 1 · `W2_clip_1_prompt_acc.png`**
> 同一个模型、同一批图，只改一句 prompt，准确率就差 5.4 个百分点。
> 排序：#2 (89.5) > #1 (88.8) > #4 (87.8) > #3 (87.4) > #5 (84.1)。
> 读法：误差棒是 95% CI（±0.60pp），横轴从 80% 起——因为 5 组都挤在 84~89.5%，
> 从 0 起画差异只有几像素宽。随机基线 10% 在轴范围外，所以写进轴标签而不是画线。

**图 2 · `W2_clip_2_per_class_acc.png`**
> prompt 引起的波动只有 5.4pp，而类别本身造成 18pp 落差（frog 78% ↔ truck 96%）。
> 掉分集中在**需要纹理细节**的类（frog、deer）；高分类是轮廓/背景高度特征化的
> （船在水上、马在草地、飞机在天上）。
> 注意：单类 n=1000，CI 约 ±2.6pp，所以 `deer 80` 与 `frog 78` 的差在噪声内，
> 只能说"同处最低一档"；但 `78 vs 96` 远超噪声。

---

## 6. 结果明细

| # | prompt 模板 | 准确率 | 95% CI |
| --- | --- | --- | --- |
| ② | `a photo of a {class}, a type of object` | **89.5%** | ±0.60 |
| ① | `a photo of a {class}` | 88.8% | ±0.60 |
| ④ | `a blurry photo of a {class}` | 87.8% | ±0.60 |
| ③ | `{class}`（裸词） | 87.4% | ±0.59 |
| ⑤ | 每类定制长句 | 84.1% | ±0.69 |
| — | 随机猜 | 10.0% | — |
| — | 阴性对照（标签打乱） | 9.7% | — |

逐类（最佳 prompt ②）：

```
truck 96  ship 96  horse 96  airplane 93  automobile 91
bird 88   cat 88   dog 88   deer 80    frog 78
```

主要混淆（每类 1000 张）：

| 真值 → 预测 | 次数 | 原因 |
| --- | --- | --- |
| deer → horse | 8 | 四足、棕色、野外背景，轮廓几乎同构 |
| frog → cat | 7 | 绿色小目标，32×32 下细节丢光 |
| truck → automobile | 6 | 类别语义本身重叠（"automobile"不含"有无货箱"） |
| deer → dog | 3 | |
| frog → bird | 3 | |

---

## 7. 三个非预期结论

**① 绝对分数比计划书预估高 20 个点（89% vs 60~70%）**
CIFAR-10 是干净网图，极可能出现在 CLIP 的预训练数据里（contamination），
同时「32×32 上采样掉点」远小于预期。所以这个数字只能当「管线正确」的证据。

**② 定制长句反而最差（84.1%，比 `a photo of a {class}` 低 4.7pp）**
CLIP 的文本塔在平均约 10 词的短 caption 上训练，
`a photo of a deer, a wild animal with antlers in the forest` 这类长句偏离训练分布，
文本特征反而变差。**「prompt 写得更详细更好」是错的直觉。**

**③ 样本量比模型关键——这是这次最重要的方法论教训**
500 张时，同一个 prompt 只换一批图就从 90.2% 掉到 87.8%（差 2.4pp），
而 ① 与 ② 之间只差 0.2pp → **那个样本量下 prompt 结论无从谈起**。
扩到全测试集后 σ 从 1.3pp 降到 0.3pp，5.4pp 的极差才显著。
**先算 CI 再比大小，别拿点估计当结论。**

---

## 8. 设计决策（为什么这么写）

| 决策 | 原因 |
| --- | --- |
| 默认全测试集而非 500 张 | 500 张的抽样噪声（±1.3pp）会淹没 prompt 差异（见结论③） |
| 加 95% CI 误差棒 | 没有区间时，"89.5 vs 88.8" 会被误读成"#2 更好" |
| 加标签打乱的阴性对照 | 90% 这个高分首先该被怀疑，对照是唯一能排除管线 bug 的手段 |
| 主图横轴从 80% 起 | 差异只有 5pp，从 0 起画肉眼分不清；牺牲"零基线"换"分辨率" |
| 随机基线写进轴标签而非画线 | 10% 在轴范围外，`axvline` 不会绘制但图例会留下"幽灵线"，反而误导 |
| 10000 张分批编码（64/批） | 一张 224×224 float32 约 0.56MB，10000 张 = 5.6GB，一次堆起来会吃爆内存 |
| 两个结论拆成两个 PNG | 一图一 claim；拼盘会让每张只占半幅，数字看不清 |
| 权重放项目目录并显式传 `download_root` | 放 `~/.cache` 会脱离项目、换机器就没了；W3 的 SAM 还有 2.4GB |

---

## 9. 文件结构

```
W2 CLIP demo/
├── demo_clip.py      主脚本（零样本分类 + prompt 对比 + 出图，五层注释）
├── PROGRESS.md       W2 全量进度记录（A/B 的实测数字 + D 的完整记录）
├── README.md         本文件
├── .gitignore        忽略 weights/ 与 runs/
├── weights/          ViT-B-32.pt（353MB，不入库）
└── runs/             产物（不入库）
```

---

## 10. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `RuntimeError: Dataset not found` | `DATA_ROOT` 没指到 W1 的 `data/` 目录 |
| `RuntimeError: CUDA out of memory` | 加 `--n-per-class 100`，或用 `--device cpu` |
| 加载时卡住不动、无输出 | 在静默重下 350MB——`WEIGHTS_DIR` 指错了 |
| 参数量不是 151,277,313 | 权重文件不完整，重下 `ViT-B-32.pt` |
| 某组准确率只有 10% 左右 | 文本编码没生效，检查 `clip.tokenize` 与 `encode_text` 是否同空间 |
| 控制台报 `UnicodeEncodeError` | Windows 控制台 GBK；设 `$env:PYTHONIOENCODING="utf-8"` |

---

## 11. 与 W1 的接口

W1 的 InfoNCE 是**图-图**对比（同图两视图为正样本），本 demo 的 CLIP 是**图-文**对比（同一图文配对为正样本）。
骨架相同，只换了「正样本对的定义」——这是 W2 的核心命题。

W2 的全量记录（含 A 训练曲线、B t-SNE 的数字）在本仓库的 `PROGRESS.md`；
W2-A / W2-B 的**代码**仍留在 W1 仓库的 `src/`（`train.py`、`plot_curve.py`、`tsne_vis.py`），
因为它们直接复用 W1 的 `model.py` / `loss.py` / `data.py`，不复制一份出来。

## 12. 未做（明确划出边界）

- CLIP 论文那套「80 个 prompt 模板取平均」的 ensemble（预期还能再涨 1~3pp）
- 线性 probe / few-shot 对比（本 demo 只做 zero-shot）
- 224 分辨率的替代数据集（计划书列为可选，非必要）
