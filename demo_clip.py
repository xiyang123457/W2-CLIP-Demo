"""W2-D · CLIP zero-shot 分类 demo（P3 产物）

【用途】
    不训练任何参数，直接用 CLIP 的预训练权重在 CIFAR-10 测试集上做 zero-shot 分类，
    并测出「prompt 措辞」对准确率的影响。
    这是 W2 三个 P 级产物里的最后一个，也是 W3 接 SAM / VLM 的入口。

【为什么只看「prompt 之间的差值」，不看绝对分数】
    CIFAR-10 只有 32x32，CLIP 预处理会把它双三次插值放大到 224 —— 信息量并没有增加，
    只是被拉伸了。所以绝对准确率天然吃亏（预期 60%~75%），
    但 5 组 prompt 用的是同一张图、同一份特征，差值只反映「措辞」这一个变量。
    ★ 实验设计上，这个对照组比绝对分数更硬。

【输入】
    weights/ViT-B-32.pt              CLIP 预训练权重（353,976,522 字节，见 WEIGHTS_DIR）
    d:\\learning project\\W1 InfoNCE\\data   CIFAR-10（复用 W1 已下载的数据，不重新下）

【输出】
    runs/W2_clip_1_prompt_acc.png    5 组 prompt + 随机基线的准确率对比（主产物）
    runs/W2_clip_2_per_class_acc.png 最佳 prompt 的 10 类分类准确率
    runs/clip_results.json           所有数字的机器可读版本（便于回头核对/画别的图）

【跑完应看到的自检数字】ViT-B/32、logit_scale=100、CIFAR-10 全测试集（每类 1000 张）：
    #1 "a photo of a {class}"        **88.8%**  +/- 0.60pp
    #2 "..., a type of object"       **89.5%**  <- 最佳
    #3 "{class}"（裸词）              **87.4%**  <- 最差里仅次于 #5，比 #1 低 1.4pp
    #4 "a blurry photo of a {class}" **87.8%**
    #5 每类定制句子                    **84.1%**  <- 最差，比 #1 低 4.7pp（手写长句反而更差，见下图）
    随机基线                          10.0%（10 类等概率；本脚本不跑模型，直接写死 1/10）
    阴性对照（标签打乱）              9.7%  <- 必须塌回 10%，否则是管线 bug
    ★ 5 组极差 5.4pp，> 2x CI(1.2pp)，所以"prompt 措辞确实有影响"这一句是站得住的。

    ⚠️ 计划书原文预期「60%~75%」，实测 90%+，原因有二，都写进周报了：
       a) CIFAR-10 的干净图片极可能出现在 CLIP 的 4 亿图文对预训练数据里（contamination），
          所以这个绝对分数偏高、不能当"CLIP 真实泛化能力"引用；
       b) 32x32 上采样到 224 的掉点没有计划书估计的那么严重。

    ⚠️ 样本量必须够：500 张时二项标准差 = sqrt(0.9*0.1/500) = 1.3pp，
      只换一批 500 张图准确率就从 90.2% 变到 87.8%（2.4pp）——
      而 #1 与 #2 只差 0.2pp，在这个噪声下【无法分辨】。所以默认值设成全测试集（每类 1000 张），
      标准差降到 0.3pp，prompt 差异才有意义。想跑快可用 --n-per-class 50，但结论只当趋势看。

【已知坑】
    ① clip 的默认权重目录是硬编码的 ~/.cache/clip，指错不会报错，而是静默重下 350MB。
       所以下面用 WEIGHTS_DIR 常量显式指定 download_root，所有加载都走它。
    ② clip.tokenize 的 context_length 默认 77，超长会截断；本脚本的 prompt 都远短于 77，不受影响。
    ③ 本脚本全程 torch.no_grad()，不产生梯度；500 张图在 RTX 5060 上约 2~4 秒。
"""

# --- 标准库 ---
# argparse：命令行参数（样本数、设备、随机种子），方便复跑与对照
import argparse
# json：把结果落成 JSON，和 W1/W2 的 metrics_*.json 同一套习惯（训练/指标/画图解耦）
import json
# time：记录耗时，用来判断「是不是又被静默重下了 350MB」
import time
# pathlib.Path：跨平台路径拼接
from pathlib import Path

# --- 第三方 ---
# numpy：分层抽样（每类取 50 张）用 default_rng，比 random.sample 更可控
import numpy as np
# torch：张量、no_grad 前向
import torch
# matplotlib：出图。★ 必须在 pyplot 之前切 Agg 后端，服务器/无显示环境才不会卡住
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# clip：OpenAI 官方实现。用它干三件事：加载权重、图像预处理 preprocess、tokenize 文本
import clip
# torchvision.datasets.CIFAR10：只用来读已下载的测试集，不做任何 transform（预处理交给 CLIP）
from torchvision.datasets import CIFAR10


# ==== 1. 路径与超参常量 ====

# 变量 PROJECT_ROOT：Path，本文件所在目录
#   示例值：d:\learning project\W2 CLIP demo
#   为什么这么取：权重和 runs/ 都跟着脚本走，换机器后 relpath 仍然成立
PROJECT_ROOT = Path(__file__).resolve().parent

# 变量 WEIGHTS_DIR：Path，CLIP 权重目录
#   示例值：d:\learning project\W2 CLIP demo\weights\ViT-B-32.pt
#   为什么不放 ~/.cache/clip：350MB 落 C 盘且脱离项目，换机器就没了；W3 的 SAM 还有 2.4GB
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# 变量 DATA_ROOT：str，CIFAR-10 的 data 目录（原始字符串，含空格与反斜杠）
#   示例值：d:\learning project\W1 InfoNCE\data
#   为什么复用 W1：计划书 D1 节明确「你已经有 data/ 了，直接复用，别新下」
DATA_ROOT = r"d:\learning project\W1 InfoNCE\data"

# 变量 RUNS_DIR：Path，所有产物落点（会被 .gitignore 排除）
RUNS_DIR = PROJECT_ROOT / "runs"

# 变量 CIFAR_CLASSES：list[str]，长度 10，CIFAR-10 的类别名（小写英文）
#   示例值：["airplane", "automobile", ..., "truck"]
#   为什么要这一份：prompt 模板里的 {class} 要填它，且顺序必须与数据集 label 0~9 严格一致
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# 变量 DEFAULT_N_PER_CLASS：int，每类抽多少张测试图
#   示例值：1000 -> 共 10000 张（= CIFAR-10 全测试集）
#   为什么不是计划书的 50（=500 张）：实测同一 prompt 只换一批 500 张图，准确率从 90.2% 到 87.8%，
#     抽样噪声 ±1.3pp 比 #1 与 #2 之间的 0.2pp 差还大 —— 那个样本量下 prompt 结论无从谈起。
#     取全测试集后二项标准差降到 sqrt(0.9*0.1/10000) = 0.3pp，差异才可分辨；代价只是约 40 秒 GPU。
DEFAULT_N_PER_CLASS = 1000

# 变量 SEED：int，抽样随机种子
#   示例值：0
#   为什么固定：5 组 prompt 必须用【完全相同】的图，否则差值里混进了「换图」的影响
SEED = 0

# 变量 N_CLASSES：int，类别数，也是「随机猜」的基线分母
#   示例值：10
#   为什么单列：随机基线 = 1/N_CLASSES = 10%，这个数字要进图和周报
N_CLASSES = len(CIFAR_CLASSES)


# ==== 2. 五组 prompt 模板 ====

# 设计决策：为什么这 5 组
#   计划书 D2 给了固定清单，正好覆盖「措辞」的三个维度：
#   有无上下文（裸词 vs 句子）、上下文长短（句尾加从句）、语气干扰（blurry）、是否针对类别定制。
#   只改这一项、其余全固定，才能把准确率差异归因到「措辞」。

# 变量 PROMPT_TEMPLATES：dict[str, list[str]]，键是图例名，值是 10 条完整句子（顺序同 CIFAR_CLASSES）
#   示例值：{"#1 ...": ["a photo of a airplane", ...], ...}
#   为什么存「展开后的 10 条」而不是模板字符串：编码前必须拿到最终文本，模板只用于展示
PROMPT_TEMPLATES = {
    "#1 a photo of a {class}": [
        f"a photo of a {c}" for c in CIFAR_CLASSES
    ],
    "#2 a photo of a {class}, a type of object": [
        f"a photo of a {c}, a type of object" for c in CIFAR_CLASSES
    ],
    "#3 {class} (bare word)": [
        f"{c}" for c in CIFAR_CLASSES
    ],
    "#4 a blurry photo of a {class}": [
        f"a blurry photo of a {c}" for c in CIFAR_CLASSES
    ],
    "#5 per-class custom sentence": [
        # 每类一句带语义描述的句子，模仿 CLIP 论文里「给每类写专属 prompt」的做法。
        # ★ 这些句子是我手写的先验（不是搜出来的标准答案），效果可能好也可能差，正是要测的东西。
        "a photo of an airplane, a vehicle with wings flying in the sky",
        "a photo of an automobile, a car driving on the road",
        "a photo of a bird, a small animal with feathers and wings",
        "a photo of a cat, a small furry domestic animal",
        "a photo of a deer, a wild animal with antlers in the forest",
        "a photo of a dog, a furry pet animal",
        "a photo of a frog, a green amphibian in water",
        "a photo of a horse, a large four-legged animal",
        "a photo of a ship, a large boat on the water",
        "a photo of a truck, a large vehicle carrying goods",
    ],
}


# ==== 3. 数据：CIFAR-10 测试集均衡抽样 ====

def build_dataset(n_per_class: int, seed: int):
    """按类别均衡抽 n_per_class 张测试图，返回 PIL 原图列表与标签。

    参数
        n_per_class：int，每类张数。示例 1000（= 全测试集）。
        seed：int，抽样种子。示例 0。
    返回
        images：list[PIL.Image]，长度 N = 10 * n_per_class，每张 32x32 RGB 原图。
        labels：LongTensor (N,)，取值 0~9，与 images 一一对应且同序。

    # 调用 torchvision.datasets.CIFAR10：
    #   签名 CIFAR10(root, train=False, download=False, transform=None) -> Dataset
    #   作用 只读已下载的 test_batch，不触发任何下载
    #   关键参数 transform=None：★ 必须留空。CIFAR-10 原生是 PIL 32x32，
    #     预处理必须由 CLIP 的 preprocess 做（它才知道 224/bicubic/那组 mean-std）
    #   坑 若 root 里没有 cifar-10-batches-py/test_batch，会直接 raise RuntimeError，
    #     此时去检查 DATA_ROOT 是否指向 W1 的 data/ 目录
    """
    dataset = CIFAR10(root=DATA_ROOT, train=False, download=False, transform=None)
    targets = np.array(dataset.targets)                       # (10000,) int array

    # 分层抽样：每类用同一个 rng 依次抽，保证种子固定时结果逐位可复现
    # 变量 rng：np.random.Generator，示例 default_rng(0)
    #   为什么用 default_rng 而不是 random.sample：前者支持按类别索引数组直接抽，无需手写循环
    rng = np.random.default_rng(seed)
    picked = []
    for c in range(N_CLASSES):
        idx_c = np.flatnonzero(targets == c)                  # 该类全部索引，约 1000 个
        picked.extend(rng.choice(idx_c, size=n_per_class, replace=False).tolist())

    images, labels = [], []
    for i in picked:
        img, label = dataset[i]                               # img 是 PIL.Image，32x32
        images.append(img)                                    # 存 PIL 原图，预处理留到 preprocess_images
        labels.append(label)

    return images, torch.tensor(labels, dtype=torch.long)


def encode_dataset(model, raw_images, preprocess, device, batch_size=64):
    """PIL 列表 -> 分批预处理 + 编码 -> L2 归一化特征 (N, 512)。

    设计决策：为什么分批，而不是先堆成一张大张量
        一张 224x224 的 float32 图 = 3*224*224*4 = 602,112 字节 ≈ 0.56MB。
        10000 张一次堆起来 = 5.6GB，机器会直接被 swap 拖死（现象往往不是报错，而是"卡住"）。
        分批后峰值只有 64 张 ≈ 36MB，且 GPU 一直有活干，总耗时几乎不变。

    # 调用 preprocess（clip.load 返回的第二个值）：
    #   签名 preprocess(PIL.Image) -> FloatTensor (3, 224, 224)
    #   作用 Resize(224, bicubic) -> CenterCrop(224) -> RGB -> ToTensor -> Normalize
    #   关键参数 无（参数已封在 Compose 里），★ 不要自己另写 Resize，mean/std 一旦不同准确率就崩
    #   坑 传进来的必须是 PIL 或 uint8 张量；直接传 float tensor 会因值域 [0,1] 被当成像素而报错
    """
    feats = []
    with torch.no_grad():
        for i in range(0, len(raw_images), batch_size):
            chunk = [preprocess(img.convert("RGB")) for img in raw_images[i:i + batch_size]]
            x = torch.stack(chunk, dim=0).to(device)           # (b, 3, 224, 224)
            f = model.encode_image(x)                          # (b, 512)
            feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    return torch.cat(feats, dim=0)


# ==== 4. 模型加载 ====

def load_model(device: str):
    """加载 CLIP ViT-B/32 并切成 eval 模式。

    # 调用 clip.load：
    #   签名 clip.load(name, device, download_root=None) -> (model, preprocess)
    #   作用 读 weights/ViT-B-32.pt（torch.jit 格式），返回推理模型 + 预处理流水线
    #   关键参数 download_root=WEIGHTS_DIR：★ 必须显式传，否则会去 ~/.cache/clip 找，
    #     找不到就静默重下 350MB（唯一现象是卡住不动，很容易被误判成"在加载"）
    #   坑 model 是 ScriptModule，没有 .train()/.eval() 之外的常规 PyTorch 接口；
    #     encode_image / encode_text 是它自带的两个方法
    """
    t0 = time.time()
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(WEIGHTS_DIR))
    model.eval()                                              # 切 eval：固定 BN/LN 统计量
    print(f"[load] ViT-B/32 ready in {time.time() - t0:.2f}s on {device}")
    return model, preprocess


# ==== 5. 特征抽取（图 / 文） ====

def encode_texts(model, texts, device):
    """10 条 prompt -> L2 归一化后的文本特征，形状 (10, 512)。

    为什么要 L2 归一化：CLIP 的相似度是余弦相似度，未归一化时模长会干扰排序。
    归一化后「点积 = 余弦相似度」，这也是画像与文本两个空间能直接比对的前提。

    # 调用 clip.tokenize：
    #   签名 clip.tokenize(text, context_length=77, truncate=False) -> LongTensor (n, 77)
    #   作用 BPE 分词 + 补齐到 77 个 token（SOT/ EOT 自动加）
    #   关键参数 truncate=True：超长时截断而不是抛错；本脚本句子都短，这个开关是保险
    #   坑 pad token 会被 attention mask 屏蔽，★ 但若手动改 context_length，权重里的位置编码就对不上了
    """
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        f = model.encode_text(tokens)                         # (10, 512)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.float().cpu()


# ==== 6. zero-shot 分类 ====

def zero_shot_classify(img_feat, txt_feat, labels, logit_scale=100.0):
    """余弦相似度 -> 取 argmax -> 算 top-1 准确率。

    为什么 logit_scale=100：CLIP 训练时 logits = 100 * cos（等价于 1/tau，tau≈0.01）。
    argmax 其实与缩放无关，但保留它能让 logits 的量级与论文一致，方便日后接 softmax 概率。
    """
    logits = logit_scale * img_feat @ txt_feat.t()            # (N, 10)
    preds = logits.argmax(dim=1)                              # (N,)
    acc = (preds == labels).float().mean().item()
    return acc, preds, logits


def binomial_ci95(acc, n):
    """准确率的 95% 置信区间半宽（正态近似）。

    为什么必须给：准确率是一次抽样的点估计，n 张图上的随机波动是
    1.96 * sqrt(p(1-p)/n)。n=500 时约 1.3pp、n=10000 时约 0.3pp。
    ★ 两个 prompt 的准确率之差若小于 2 倍半宽，就不能说"谁更好" —— 这是本图能下结论的前提。
    """
    return 1.96 * float(np.sqrt(max(acc * (1 - acc), 1e-6) / n))


# ==== 7. 出图与落盘 ====

def plot_prompt_acc(results, random_baseline, n_images, out_path):
    """主图：5 组 prompt 的准确率横向条形图（带 95% CI）+ 随机基线竖线。"""
    names = list(results.keys())
    accs = [results[n]["acc"] * 100 for n in names]           # 换成百分数便于阅读
    errs = [binomial_ci95(results[n]["acc"], n_images) * 100 for n in names]

    # 设计决策：横轴不从 0 开始，而从 80% 起
    #   5 组都挤在 86%~90%，从 0 起画差异只有 4 格宽度，肉眼分不清；
    #   从 80 起画"差几个百分点"才是这张图要传达的信息（随机基线 10% 另行标注在标题里）
    x_lo = 80.0

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    y = np.arange(len(names))
    bars = ax.barh(y, [a - x_lo for a in accs], left=x_lo, height=0.55,
                   color="#4C72B0", edgecolor="black", linewidth=0.6,
                   xerr=errs, error_kw={"ecolor": "black", "capsize": 4, "linewidth": 1.2})
    # 最佳那一根换成橙色，让「哪个 prompt 最好」一眼可见，不用读数字
    best = int(np.argmax(accs))
    bars[best].set_color("#DD8452")

    for yi, acc, err in zip(y, accs, errs):                    # 条尾写数值与置信区间
        ax.text(acc + err + 0.25, yi, f"{acc:.1f}%", va="center", fontsize=10)

    # 随机基线画不进这张图（10% 在 80~92 的范围外），所以写进 x 轴标签 ——
    # ★ 不要用 axvline：线落在范围外根本不会绘制，图例里却仍会出现一条"幽灵"线，反而误导读者
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()                                         # 让 #1 排在最上面，符合阅读顺序
    ax.set_xlabel(f"zero-shot top-1 accuracy (%) on {n_images} CIFAR-10 test images\n"
                  f"(random guess = {random_baseline * 100:.1f}%, off scale; x-axis starts at {x_lo:.0f}%)",
                  fontsize=9.5)
    ax.set_xlim(x_lo, max(a + e for a, e in zip(accs, errs)) + 2.0)
    ax.set_title("CLIP ViT-B/32 zero-shot: prompt wording sweep (error bars = 95% CI)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path.name}")


def plot_per_class(best_name, per_class_acc, random_baseline, out_path):
    """副图：最佳 prompt 下 10 个类各自的准确率（看哪几类在掉分）。"""
    names = list(per_class_acc.keys())
    accs = [per_class_acc[n] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    x = np.arange(len(names))
    ax.bar(x, accs, color="#55A868", edgecolor="black", linewidth=0.6, width=0.65)
    for xi, acc in zip(x, accs):
        ax.text(xi, acc + 1.5, f"{acc:.0f}", ha="center", fontsize=9)

    ax.axhline(random_baseline * 100, color="crimson", linestyle="--", linewidth=1.2,
               label=f"random guess = {random_baseline * 100:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=9)
    ax.set_ylabel("top-1 accuracy (%)")
    # 上限留到 112：最差情况有一类 100%，数字标签必须画在柱顶之外，否则会被裁掉
    ax.set_ylim(0, 112)
    # 标题只取 "#2" 这种短编号：完整模板名太长，会和柱子/数字标签重叠
    ax.set_title(f"Per-class zero-shot accuracy (best prompt = {best_name.split(' ')[0]})",
                 fontsize=11)
    # 图例放左上：10 类最低也有 78%，下方没有空白可用，只有 100% 以上的顶部是空的
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path.name}")


# ==== 8. 主流程 ====

def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot demo on CIFAR-10")
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS,
                        help="每类抽多少张测试图（默认 1000 -> 共 10000 张 = 全测试集；"
                             "想要快可给 50，但抽样噪声 1.3pp 会淹没 prompt 差异）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 先固定随机源：抽样种子固定 => 5 组 prompt 面对同一批图
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess = load_model(args.device)

    # ---- 数据 ----
    print(f"[data] CIFAR-10 test from {DATA_ROOT}, {args.n_per_class}/class "
          f"= {args.n_per_class * N_CLASSES} images, seed={SEED}")
    raw_images, labels = build_dataset(args.n_per_class, SEED)
    print(f"[data] {len(raw_images)} PIL images (32x32)  labels {tuple(labels.shape)}  "
          f"class counts {np.bincount(labels.numpy(), minlength=N_CLASSES).tolist()}")

    # ---- 图特征只算一次（这就是 zero-shot 的全部"计算"：不训练、不反向）----
    t_feat = time.time()
    img_feat = encode_dataset(model, raw_images, preprocess, args.device)   # (N, 512)
    # 自检：归一化后每一行的 L2 范数应当非常接近 1
    print(f"[feat] image feats {tuple(img_feat.shape)}  mean L2 norm "
          f"{img_feat.norm(dim=-1).mean().item():.4f}  ({time.time() - t_feat:.1f}s)")

    # ---- 逐模板评估 ----
    random_baseline = 1.0 / N_CLASSES                         # 10 类等概率 = 0.1
    results = {}
    for name, texts in PROMPT_TEMPLATES.items():
        txt_feat = encode_texts(model, texts, args.device)    # (10, 512)
        acc, preds, _ = zero_shot_classify(img_feat, txt_feat, labels)
        results[name] = {"acc": acc, "preds": preds, "texts": texts}
        print(f"[eval] {name:<45s} acc = {acc * 100:5.1f}%")

    # ---- 阴性对照：把标签随机打乱，准确率必须塌回 10% ----
    # 设计决策：为什么不只测真标签
    #   「90%」这个数字太高，第一反应是怀疑管线（索引错位、把训练集当测试集）。
    #   打乱标签后若仍是 90%，就说明预测与标签无关 —— 那才是 bug；
    #   塌回 10% 则证明"预测 = 图像特征 x 文本特征"这条链是干净的。
    n_images = int(labels.numel())
    rng = np.random.default_rng(SEED)
    shuf_labels = labels[rng.permutation(n_images)]
    first_name = next(iter(results))
    txt_feat = encode_texts(model, PROMPT_TEMPLATES[first_name], args.device)
    acc_shuf, _, _ = zero_shot_classify(img_feat, txt_feat, shuf_labels)
    print(f"[ctrl] shuffled-label acc ({first_name}) = {acc_shuf * 100:.1f}%  (expect ~10%)")

    # ---- 最佳模板的逐类拆解 ----
    best_name = max(results, key=lambda k: results[k]["acc"])
    best_preds = results[best_name]["preds"]
    per_class_acc = {}
    for c in range(N_CLASSES):
        mask = labels == c
        per_class_acc[CIFAR_CLASSES[c]] = (best_preds[mask] == c).float().mean().item()

    # ---- 出图 ----
    plot_prompt_acc(results, random_baseline, n_images, RUNS_DIR / "W2_clip_1_prompt_acc.png")
    plot_per_class(best_name, per_class_acc, random_baseline,
                   RUNS_DIR / "W2_clip_2_per_class_acc.png")

    # ---- 落 JSON（只存人类可读的字段，preds 太长不入盘）----
    payload = {
        "model": "ViT-B/32",
        "dataset": "CIFAR-10 test",
        "n_per_class": args.n_per_class,
        "n_images": n_images,
        "seed": SEED,
        "logit_scale": 100.0,
        "random_baseline": random_baseline,
        "shuffled_label_acc": acc_shuf,
        "prompt_results": {k: {"acc": v["acc"],
                               "ci95_halfwidth": binomial_ci95(v["acc"], n_images),
                               "sample_text_first_class": v["texts"][0]}
                           for k, v in results.items()},
        "best_prompt": best_name,
        "best_acc": results[best_name]["acc"],
        "per_class_acc_best_prompt": per_class_acc,
    }
    json_path = RUNS_DIR / "clip_results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[dump] {json_path.name}")

    # ---- 结论摘要（周报直接用这几行）----
    accs = {k: v["acc"] for k, v in results.items()}
    spread = max(accs.values()) - min(accs.values())
    ci = binomial_ci95(accs[best_name], n_images)
    print("\n================ SUMMARY ================")
    print(f"random baseline        : {random_baseline * 100:.1f}%")
    print(f"shuffled-label control : {acc_shuf * 100:.1f}%  (must be ~10%)")
    print(f"best prompt            : {best_name}")
    print(f"best acc               : {accs[best_name] * 100:.1f}% +/- {ci * 100:.2f}pp (95% CI)")
    print(f"worst prompt           : {min(accs, key=accs.get)} -> "
          f"{min(accs.values()) * 100:.1f}%")
    print(f"spread across prompts  : {spread * 100:.1f} pp "
          f"({'> 2x CI, meaningful' if spread > 2 * ci else '<= 2x CI, within noise'})")
    print(f"gain over random       : {(accs[best_name] - random_baseline) * 100:.1f} pp "
          f"({accs[best_name] / random_baseline:.1f}x)")
    print(f"worst classes          : "
          f"{sorted(per_class_acc, key=per_class_acc.get)[:2]}")
    print("=========================================")

    # 自检断言：模型若真在做事，准确率必须显著高于随机；否则多半是特征/文本没对齐
    assert accs[best_name] > random_baseline * 3, (
        f"best acc {accs[best_name]:.3f} 不到随机基线 3 倍，"
        "检查文本特征与图像特征是否同一个 512 维空间"
    )
    # 阴性对照若也高达 80%+，说明预测与标签解耦失败（索引/子集错位）
    assert acc_shuf < random_baseline * 3, (
        f"shuffled-label acc {acc_shuf:.3f} 太高，管线可能没有真正对齐图像与标签"
    )
    print("[PASS] zero-shot demo done")


# ---------------------------------------------------------------------------
# 怎么验证跑对了（按顺序对数字）
#   1) [load] ViT-B/32 ready in ~2s，且【没有】出现 Downloading 字样
#        -> 出现下载 = WEIGHTS_DIR 指错了
#   2) [data] images (10000, 3, 224, 224)，class counts 十个 1000
#   3) [feat] image feats (10000, 512)，mean L2 norm 1.0000
#   4) [ctrl] shuffled-label acc ≈ 10%  <-- 这是"管线干净"的铁证
#   5) 五组 acc 都在 84%~90%，spread 5.4pp，且要 > 2x CI（CI 约 0.60pp）
#        - 若某组只有 10% 左右 = 文本编码没生效（多半 tokenize 后被 pad 淹没）
#        - 若 shuffled 也 80%+ = 索引/标签错位，查 build_dataset 的 picked 与 labels 是否同序
#   6) worst classes 应是 deer(80%) / frog(78%) 这类需要细节的类
#
# 实测（2026-10-17，seed=0，ViT-B/32）：
#   全测试集 10000 张：#1 88.8 / #2 89.5 / #3 87.4 / #4 87.8 / #5 84.1，shuffled 9.7
#   同一脚本 500 张：#1 90.2 / #2 90.4 / #3 88.0 / #4 88.6 / #5 86.2，shuffled 10.4
#   500 张换 seed=1：#1 87.8 —— 仅换 500 张图就差 2.4pp，说明 500 张的抽样噪声(±1.3pp)
#   足以盖过 #1 与 #2 的 0.2pp 差，所以结论必须用全测试集跑（这也是 default 设为 1000 的原因）
#
# 常见异常对照
#   RuntimeError: Dataset not found        -> DATA_ROOT 没指到 W1 的 data/
#   RuntimeError: CUDA out of memory       -> 加 --n-per-class 100，或 --device cpu
#   params 数量与 151277313 不符           -> 权重文件不完整，重下 ViT-B-32.pt
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
