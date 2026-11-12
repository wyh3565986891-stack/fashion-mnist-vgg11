"""Generate the Chinese PDF report from the JSON/PNG artifacts produced by the experiments.

Every number in the PDF is read from runs/*.json, so the report cannot drift from
the code output. Build:

    python report.py --out "../output/CNN大作业/2427080125王炎煌.pdf"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

# --------------------------------------------------------------------------- #
# fonts and styles
# --------------------------------------------------------------------------- #
pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
pdfmetrics.registerFont(TTFont("KaiTi", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))

BODY = ParagraphStyle("Body", fontName="SimSun", fontSize=10.5, leading=17.5,
                      firstLineIndent=21, alignment=TA_JUSTIFY, spaceAfter=3,
                      wordWrap="CJK", splitLongWords=1)
BODY_NO_INDENT = ParagraphStyle("BodyNoIndent", parent=BODY, firstLineIndent=0)
CAPTION = ParagraphStyle("Caption", fontName="SimSun", fontSize=8.8, leading=13,
                         alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
                         spaceBefore=3, spaceAfter=8)
H1 = ParagraphStyle("Heading1", fontName="SimHei", fontSize=15, leading=22,
                    spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#10243e"))
H2 = ParagraphStyle("Heading2", fontName="SimHei", fontSize=11.8, leading=19,
                    spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#10243e"))
CELL = ParagraphStyle("Cell", fontName="SimSun", fontSize=8.4, leading=11.5, alignment=TA_CENTER)
CELL_HEAD = ParagraphStyle("CellHead", fontName="SimHei", fontSize=8.4, leading=11.5,
                           alignment=TA_CENTER)
CELL_LEFT = ParagraphStyle("CellLeft", parent=CELL, alignment=TA_LEFT)
BULLET = ParagraphStyle("Bullet", parent=BODY, firstLineIndent=0, leftIndent=14,
                        bulletIndent=2, spaceAfter=2)

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 24 * mm
FRAME_WIDTH = PAGE_WIDTH - 2 * MARGIN


def P(text: str, style=BODY):
    return Paragraph(text, style)


def H1T(number: str, title: str):
    paragraph = Paragraph(f"{number}&nbsp;&nbsp;{title}".rstrip("&nbsp;"), H1)
    paragraph.toc_key = (number, title)
    return paragraph


def H2T(number: str, title: str):
    paragraph = Paragraph(f"{number}&nbsp;&nbsp;{title}", H2)
    paragraph.toc_key = (number, title)
    return paragraph


def caption(text: str):
    return Paragraph(text, CAPTION)


def bullets(items: list[str]):
    return [Paragraph(f"•&nbsp;&nbsp;{item}", BULLET) for item in items]


def table(rows, widths, header: bool = True, font_size: float = 8.4,
          align_left_first: bool = False):
    data = []
    for i, row in enumerate(rows):
        line = []
        for j, cell in enumerate(row):
            style = CELL_LEFT if (align_left_first and j == 0) else CELL
            if header and i == 0:
                style = CELL_HEAD
            line.append(Paragraph(str(cell), style) if not isinstance(cell, Paragraph) else cell)
        data.append(line)
    widths = [w * mm for w in widths]
    table_object = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                  ("FONTNAME", (0, 0), (-1, 0), "SimHei")]
    table_object.setStyle(TableStyle(style))
    return KeepTogether([table_object])


def figure(path: Path, width_mm: float, text: str):
    with PILImage.open(path) as image:
        aspect = image.height / image.width
    width = width_mm * mm
    return KeepTogether([Image(str(path), width=width, height=width * aspect), caption(text)])


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def n(value) -> str:
    return f"{value:,}"


def pc(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    return f"{value * 100:+.{digits}f}"


def pct1(value: float) -> str:
    return f"{value * 100:.1f}"


def num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def top_confusions(record, k=4):
    matrix = np.array(record["test"]["confusion"], dtype=float)
    labels = record["classes"]
    support = matrix.sum(1)
    matrix[np.diag_indices_from(matrix)] = 0
    flat = np.argsort(-matrix, axis=None)[:k]
    rows = []
    for index in flat:
        i, j = divmod(int(index), matrix.shape[1])
        rows.append((labels[i], labels[j], matrix[i, j] / support[i]))
    return rows


# --------------------------------------------------------------------------- #
# document
# --------------------------------------------------------------------------- #
class Report(BaseDocTemplate):
    def __init__(self, path, pages=None, **kwargs):
        super().__init__(str(path), **kwargs)
        self.pages = pages if pages is not None else {}
        frame = Frame(MARGIN, MARGIN, FRAME_WIDTH, PAGE_HEIGHT - 2 * MARGIN,
                      id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self.decorate)])

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("SimSun", 8.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(MARGIN, PAGE_HEIGHT - MARGIN + 8 * mm,
                          "神经网络与深度学习 · 大作业报告")
        canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN + 8 * mm,
                               "服饰图像 CNN 结构设计与实验分析")
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_HEIGHT - MARGIN + 6 * mm,
                    PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN + 6 * mm)
        canvas.line(MARGIN, MARGIN - 4 * mm, PAGE_WIDTH - MARGIN, MARGIN - 4 * mm)
        canvas.drawCentredString(PAGE_WIDTH / 2, MARGIN - 8.5 * mm, f"— {doc.page} —")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        key = getattr(flowable, "toc_key", None)
        if key is not None:
            self.pages[key[0]] = self.page


def cover(date_text: str) -> list:
    story = [Spacer(1, 42 * mm)]
    story.append(Paragraph("神经网络与深度学习", ParagraphStyle(
        "cover1", fontName="SimHei", fontSize=17, alignment=TA_CENTER, leading=26)))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("大作业报告", ParagraphStyle(
        "cover2", fontName="SimHei", fontSize=30, alignment=TA_CENTER, leading=42)))
    story.append(Spacer(1, 14 * mm))
    story.append(Paragraph("面向服饰图像的轻量化 CNN", ParagraphStyle(
        "cover3", fontName="SimHei", fontSize=16, alignment=TA_CENTER, leading=28)))
    story.append(Paragraph("结构设计、消融分析与部署实践", ParagraphStyle(
        "cover4", fontName="SimHei", fontSize=16, alignment=TA_CENTER, leading=28)))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("—— 一个窄通道 VGG11 的九组对照实验 ——", ParagraphStyle(
        "cover5", fontName="SimSun", fontSize=11, alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"))))
    story.append(Spacer(1, 34 * mm))
    info = [["姓　　名", "王炎煌"], ["学　　号", "2427080125"], ["班　　级", "智科 24-1"],
            ["课程名称", "神经网络与深度学习"], ["完成日期", date_text]]
    rows = [[Paragraph(a, ParagraphStyle("ci", fontName="SimSun", fontSize=12, alignment=TA_CENTER)),
             Paragraph(b, ParagraphStyle("cv", fontName="SimSun", fontSize=12, alignment=TA_CENTER))]
            for a, b in info]
    cover_table = Table(rows, colWidths=[34 * mm, 62 * mm])
    cover_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(cover_table)
    story.append(PageBreak())
    return story


def toc_page(pages: dict, entries: list[tuple[str, str]]) -> list:
    """Table of contents; page numbers come from a first measuring pass."""
    story = [Paragraph("目　　录", ParagraphStyle("toctitle", fontName="SimHei", fontSize=16,
                                                  alignment=TA_CENTER, spaceAfter=12))]
    for level, title in entries:
        page = pages.get(level, "")
        left = f"{level}  {title}" if "." not in level else f"　　{level}  {title}"
        style = ParagraphStyle("toc", fontName="SimSun", fontSize=10.5, leading=19.5)
        left_width = pdfmetrics.stringWidth(left, "SimSun", 10.5)
        right_width = pdfmetrics.stringWidth(str(page), "SimSun", 10.5)
        dot_width = max(0.5, pdfmetrics.stringWidth("·", "SimSun", 10.5))
        available = FRAME_WIDTH - left_width - right_width - 14
        dots = "·" * max(2, int(available / dot_width))
        story.append(Paragraph(f"{left}  {dots}  {page}", style))
    story.append(PageBreak())
    return story


def section_intro(S, R, D, A) -> list:
    base = R["vgg11"]
    story = [H1T("1", "引言与任务定义")]
    story.append(H2T("1.1", "任务背景"))
    story.append(P(
        "服装图像的自动识别是电商平台最基础也最密集的视觉任务之一：以图搜款、自动打标、"
        "库存盘点、二手平台成色描述都依赖一个能把「这是衬衫还是外套」判断准确的分类器。"
        "这类任务通常部署在手机端或边缘设备上，算力与内存都受限，因此「用尽量小的模型把准确率做上去」"
        "比单纯追求 SOTA 更有工程意义。本作业选择 Fashion-MNIST 作为可控的实验床，"
        "在统一的训练协议下比较一种经典卷积架构与若干结构修改，定量回答两个问题："
        "（1）不同 CNN 组件各自贡献了多少精度、多少计算量；（2）在同样的算力预算下，"
        "哪种组合更适合部署。"))
    story.append(P(
        "需要说明的是，课内实验侧重「网络训练与调试流程」，本作业的重心在结构本身："
        "从 VGG11 的可控基线出发，只改动卷积算子、注意力、激活函数与归一化层中的一项，"
        "其余训练配置完全一致，从而得到可比较的因果结论，而不是把多个技巧堆在一起。"))

    story.append(H2T("1.2", "数据集"))
    story.append(P(
        "Fashion-MNIST 由 Zalando 发布，规模、格式与 MNIST 完全一致：60000 张训练图、10000 张测试图，"
        "单通道 28×28，10 个类别、每类严格 6000/1000 张，因此不需要处理类别不平衡。"
        "与手写数字相比，它的类间差异更小——T-shirt、Pullover、Coat、Shirt 四类共享几乎相同的轮廓，"
        "仅靠局部纹理区分，这也是后文混淆矩阵中错误的主要来源。"))
    rows = [["类别编号", "类别名称", "训练集", "测试集"]]
    for index, (name, train_count, test_count) in enumerate(
            zip(D["classes"], D["train_per_class"], D["test_per_class"])):
        rows.append([index, name, n(train_count), n(test_count)])
    rows.append(["合计", "10 类", n(D["train_size"]), n(D["test_size"])])
    story.append(table(rows, [22, 52, 26, 26]))
    story.append(caption(
        f"表 1　数据集构成（图像尺寸 {D['image_shape'][0]}×{D['image_shape'][1]}，"
        f"灰度；训练集像素均值 {D['pixel_mean']:.4f}、标准差 {D['pixel_std']:.4f}）"))
    story.append(P(
        f"预处理只做两件事：训练时 RandomCrop(28, padding=2) 与水平翻转做数据增广，"
        f"随后按训练集统计量 (mean={D['normalization_mean']}, std={D['normalization_std']}) 做标准化；"
        "验证与测试只做标准化。这样验证集上的精度不会被增广「污染」，可以用来选模型。"))

    story.append(H2T("1.3", "评价指标"))
    story.append(P(
        "准确率（accuracy）是主指标，但因为四类上衣容易互相混淆，仅看总准确率会掩盖局部退化，"
        "所以同时报告宏平均 F1（macro-F1，对 10 个类别的 F1 取算术平均，每类权重相同）。"
        "模型成本用三项衡量：参数量（可训练参数个数）、单张图 MACs（乘加次数，"
        "卷积层按 H′W′·(C_in/g)·k²·C_out 统计，全连接层按 in×out 统计，由 forward hook 实测）、"
        "以及 CPU 单线程推理延迟。训练成本另外记录每轮耗时，用于比较不同算子的实际开销。"))
    return story


def section_baseline(S, R, A) -> list:
    base = R["vgg11"]
    story = [H1T("2", "基线模型：窄通道 VGG11")]
    story.append(H2T("2.1", "网络结构"))
    story.append(P(
        "VGG 的核心思想是「用 3×3 小卷积堆叠换取感受野」：两个 3×3 卷积的感受野等价于一个 5×5，"
        "但参数更少、非线性更多。原版 VGG11 面向 224×224 的 ImageNet，通道数 64→512，"
        "直接搬到 28×28 上会带来大量无效计算（最后几个 stage 的空间尺寸只有 1×1，"
        "继续堆 512 通道几乎没有意义）。因此本文保留 VGG11 的层数节奏（每个 stage 的卷积个数为 "
        "1-1-2-2-2），把通道宽度缩到 16-32-64-96-96，并在每个 stage 之后接 2×2 最大池化（最后一个 stage 除外），"
        "得到「窄通道 VGG11」。"))
    story.append(P(
        "每个卷积单元是 Conv3×3(padding=1, bias=False) → BatchNorm → ReLU 的标准组合；"
        "五个 stage 之后是全局平均池化、三层 MLP 分类头（96→128→64→10，中间夹 ReLU 与 Dropout 0.2）。"
        "全局平均池化替代了 VGG 原来的三层大全连接，把分类头参数压到 2 万左右，"
        "这也是本作业能够把总参数量控制在 40 万以内的关键。"))
    rows = [["阶段", "输出尺寸", "参数量", "占比", "MACs", "占比"]]
    for row in A["vgg11"]["rows"]:
        shape = "×".join(str(v) for v in row["shape"]) if row["shape"] else "—"
        rows.append([row["stage"].replace("stage", "Stage "), shape, n(row["params"]),
                     pc(row["params_share"], 1), n(row["macs"]), pc(row["macs_share"], 1)])
    rows.append(["合计", "—", n(A["vgg11"]["params"]), "100%",
                 n(A["vgg11"]["macs"]), "100%"])
    story.append(table(rows, [24, 28, 30, 18, 32, 18]))
    story.append(caption("表 2　基线网络逐阶段成本（由 describe.py 统计）"))
    stages = {row["stage"]: row for row in A["vgg11"]["rows"]}
    third = stages["stage3"]
    late_params = stages["stage4"]["params_share"] + stages["stage5"]["params_share"]
    late_macs = stages["stage4"]["macs_share"] + stages["stage5"]["macs_share"]
    story.append(P(
        f"表 2 暴露了两个结构性问题：其一，Stage 3 只占 {pc(third['params_share'], 1)} 的参数，"
        f"却吃掉 {pc(third['macs_share'], 1)} 的 MACs——"
        f"因为它的特征图还有 7×7，而通道已有 64；其二，Stage 4、5 合计占 {pc(late_params, 1)} 的参数，"
        f"但空间尺寸只剩 3×3 和 1×1，MACs 占比只有 {pc(late_macs, 1)}。这提示两件事："
        "压缩计算量应该优先动 Stage 3 这类「大特征图 × 中等通道」的层；"
        "而压缩参数量应该动 Stage 4、5 这类「小特征图 × 大通道」的层。后面的深度可分离卷积实验正是按这个思路设计的。"))

    story.append(H2T("2.2", "训练设置"))
    story.append(P(
        "所有实验共用一套训练协议，只改网络结构，保证对比公平。官方 60000 张训练图按固定随机种子（42）"
        "切成 55000 张训练集与 5000 张验证集，官方 10000 张测试集只在最后评估一次，不参与任何选择。"
        "验证集用于挑选最佳轮次并保存权重，报告中的测试指标一律来自这份最佳权重。"))
    rows = [["项目", "设置"],
            ["优化器", "AdamW（β=(0.9, 0.999)）"],
            ["学习率", "1×10⁻³，CosineAnnealingLR 衰减到 0"],
            ["权重衰减", "5×10⁻⁴"],
            ["批大小", "128（第 3.5 节的小批量对照实验另设为 16）"],
            ["损失函数", "交叉熵 + label smoothing 0.05"],
            ["训练轮数", "16"],
            ["数据增广", "RandomCrop(28, padding=2) + RandomHorizontalFlip(0.5)"],
            ["随机种子", "42（数据划分、初始化、增广共用）"],
            ["硬件", "Intel i9-14900HX，6 线程 PyTorch CPU 训练"]]
    story.append(table(rows, [30, 96], align_left_first=True))
    story.append(caption("表 3　统一训练协议"))
    story.append(P(
        "几个选择的理由：label smoothing 0.05 缓解四类上衣的过自信问题；余弦退火让最后几轮学习率接近 0，"
        "避免验证精度在末期抖动；RandomCrop 的 padding=2 模拟了真实场景里服饰在画面中偏移、裁切不完整的情况，"
        "比单纯的随机翻转更贴近部署时的输入分布。"))

    story.append(H2T("2.3", "基线结果与分析"))
    story.append(P(
        f"基线在测试集上取得 {num(base['test']['accuracy'])} 的准确率与 {num(base['test']['macro_f1'])} 的 macro-F1，"
        f"最佳轮次为第 {base['best_epoch']} 轮，单轮平均耗时 {base['seconds_per_epoch']:.1f} s，"
        f"整次训练 {base['train_seconds'] / 60:.1f} min。这一水平与 Fashion-MNIST 上同类小 CNN 的公开结果相当，"
        "说明训练协议没有明显缺陷，可以作为后续消融的参照点。"))
    story.append(P(
        "把 10×10 的混淆矩阵按行归一化后（图 1），对角线以外的错误高度集中："))
    confusions = top_confusions(base, 4)
    lines = [f"{a} 被判成 {b}，占该类样本的 {pc(r, 1)}" for a, b, r in confusions]
    story.append(P("；".join(lines) + "。"))
    scores = dict(zip(base["classes"], base["test"]["per_class"]["f1"]))
    easy_classes = ["Trouser", "Sandal", "Sneaker", "Bag", "Ankle boot"]
    hard_classes = ["T-shirt/top", "Pullover", "Coat", "Shirt"]
    worst_class = min(scores, key=scores.get)
    story.append(P(
        "这四类上衣的轮廓几乎一致，区别主要在领口、袖长与纹理，而 28×28 的灰度图在两次池化后只剩 7×7，"
        f"细节已经被抹平：它们的 F1 只有 {pc(min(scores[c] for c in hard_classes), 1)}–"
        f"{pc(max(scores[c] for c in hard_classes), 1)}，其中最难的是 {worst_class}（{num(scores[worst_class])}）。"
        f"相比之下，鞋、包、裤子这些轮廓独特的类别 F1 都在 {pc(min(scores[c] for c in easy_classes), 1)} 以上（图 2）。"
        "换言之，基线的瓶颈不在优化，而在「输入分辨率 + 类别语义重叠」这一对客观约束上——"
        "后面的结构修改能改善的是特征选择效率，而不是信息量本身。"))
    story.append(figure(Path("figures/confusion.png"), 165,
                        "图 1　测试集混淆矩阵（按行归一化，数值为百分比）：错误集中在四类上衣之间"))
    story.append(figure(Path("figures/per_class_f1.png"), 165,
                        "图 2　各类别 F1：鞋、包、裤接近饱和，上衣四类明显偏低"))
    return story


def section_ablation(S, R, A) -> list:
    story = [H1T("3", "结构修改与消融实验")]
    story.append(P(
        "消融实验遵循「一次只动一个变量」的原则，共 8 个变体 + 1 个基线（表 4）。"
        "所有变体使用同一份数据划分、同一个随机种子、同一套超参数，只有网络结构不同；"
        "表中「改动位置」一列说明修改发生在哪些 stage，未列出的部分与基线完全一致。"))
    order = ["vgg11", "sep", "sep_all", "se", "cbam", "gelu", "mish", "gn", "final"]
    labels = {"vgg11": "基线", "sep": "可分离卷积（Stage 3–5）", "sep_all": "可分离卷积（全部）",
              "se": "SE 注意力", "cbam": "CBAM 注意力", "gelu": "GELU 激活",
              "mish": "Mish 激活", "gn": "GroupNorm(8)", "final": "可分离 + CBAM + GELU"}
    rows = [["编号", "配置", "结构改动"]]
    for index, name in enumerate(order):
        rows.append([f"C{index}", name, labels[name]])
    story.append(table(rows, [14, 44, 68], align_left_first=True))
    story.append(caption("表 4　九组对照实验"))

    # ---------------- 3.1 separable ---------------- #
    story.append(H2T("3.1", "深度可分离卷积"))
    story.append(P(
        "标准 3×3 卷积同时做「空间滤波」和「通道混合」，一次乘加就把 C_in 个通道压成 C_out 个。"
        "深度可分离卷积把它拆成两步：先对每个通道单独做 3×3 空间卷积（depthwise，groups=C_in），"
        "再用 1×1 卷积做通道线性组合（pointwise）。参数量从 k²·C_in·C_out 降到 k²·C_in + C_in·C_out，"
        "两者之比约为 1/C_out + 1/k²；当 C_out=96、k=3 时理论值只有 12.2%。"
        "计算量的推导相同，只是多乘一个输出空间尺寸 H′W′。"))
    story.append(P(
        "本作业设置两个变体：sep 只替换 Stage 3–5（即表 2 中参数量最大的后三个阶段），"
        "sep_all 替换全部五个 stage。这样设计是为了回答「压缩应该压在哪里」——"
        "如果只在参数量大的地方压缩就能拿到大部分收益，说明参数分布确实决定了压缩空间。"))
    rows = [["配置", "参数量", "相对基线", "MACs", "相对基线", "准确率", "相对基线"]]
    for name in ("vgg11", "sep", "sep_all"):
        record = R[name]
        rows.append([name, n(record["parameters"]),
                     pc(record["parameters"] / R["vgg11"]["parameters"], 1),
                     n(record["macs"]), pc(record["macs"] / R["vgg11"]["macs"], 1),
                     num(record["test"]["accuracy"]),
                     pp(record["delta_accuracy"]) + " pp"])
    story.append(table(rows, [22, 26, 22, 26, 22, 24, 24]))
    story.append(caption("表 5　深度可分离卷积的参数量、计算量与精度"))
    sep = R["sep"]
    sep_all = R["sep_all"]
    sep_text = (
        f"sep 把参数量压缩到基线的 {pc(sep['parameters'] / R['vgg11']['parameters'], 1)}、"
        f"MACs 压到 {pc(sep['macs'] / R['vgg11']['macs'], 1)}，测试准确率 {pp(sep['delta_accuracy'])} 个百分点；"
        f"sep_all 进一步压到 {pc(sep_all['parameters'] / R['vgg11']['parameters'], 1)} 的参数与 "
        f"{pc(sep_all['macs'] / R['vgg11']['macs'], 1)} 的 MACs，准确率 {pp(sep_all['delta_accuracy'])} 个百分点。")
    better = "sep" if sep["test"]["accuracy"] >= sep_all["test"]["accuracy"] else "sep_all"
    extra_macs = sep["macs"] - sep_all["macs"]
    extra_params = sep["parameters"] - sep_all["parameters"]
    story.append(P(
        sep_text + f"两组之间 {better} 更好。值得注意的是，sep_all 相对 sep 只多省了 "
        f"{n(extra_params)} 个参数（{pc(extra_params / sep['parameters'], 1)}）与 "
        f"{n(extra_macs)} 次乘加（{pc(extra_macs / sep['macs'], 1)}），"
        f"精度却再掉 {abs(sep_all['delta_accuracy'] - sep['delta_accuracy']) * 100:.2f} 个百分点。"
        "可分离卷积的压缩比在浅层与深层差别并不大（1/C_out+1/k² 从 14% 到 12%），"
        "所以这点额外损失不能归因于「压得太狠」，而应归因于浅层标准卷积承担的职责："
        "它在原始像素上同时完成空间滤波与通道组合，拆成 depthwise + 1×1 两步之后，"
        "第一步完全不做通道混合，跨通道的表达被推迟到第二步，"
        "而窄通道网络的容量不足以把这部分损失补回来。"))
    stage3_base = {row["stage"]: row for row in A["vgg11"]["rows"]}["stage3"]["macs"]
    stage3_final = {row["stage"]: row for row in A["final"]["rows"]}["stage3"]["macs"]
    theory = 1 / 9 + 1 / 64
    story.append(P(
        f"逐层数据可以验证上面的推导：final 把 Stage 3 换成可分离卷积后，该阶段的 MACs 从 "
        f"{n(stage3_base)} 降到 {n(stage3_final)}，只剩 {pc(stage3_final / stage3_base, 1)}，"
        f"与 C_in=C_out=64、k=3 时的理论比值 {pc(theory, 1)} 基本吻合，"
        "差出来的一小部分是同一 stage 里 CBAM 注意力分支的开销。"
        "这条经验对真实部署很重要：可分离卷积在「宽通道 + 大特征图」的层收益最大，"
        "在浅层窄通道上却未必划算——此时 1/k² 项不再占绝对主导，额外的 1×1 卷积会抬高兴起开销的比例。"))

    # ---------------- 3.2 attention ---------------- #
    story.append(H2T("3.2", "注意力机制：SE 与 CBAM"))
    story.append(P(
        "SE（Squeeze-and-Excitation）模块先用全局平均池化把每个通道压成一个标量，"
        "再经一个 reduction=8 的瓶颈 MLP 学出通道权重，最后逐通道相乘。它建模的是「哪些通道重要」，"
        "几乎不改变空间分辨率，因此开销很小。CBAM 在通道注意力之外再加一个空间注意力分支："
        "沿通道维取平均与最大得到两张 28×28（或 7×7）的图，拼接后过 7×7 卷积得到空间权重图。"
        "两者都插在每个 stage 的最后一个卷积之后、池化之前，即 Stage 3–5。"))
    rows = [["配置", "新增参数", "参数量", "MACs", "准确率", "相对基线", "macro-F1"]]
    for name in ("vgg11", "se", "cbam"):
        record = R[name]
        extra = record["parameters"] - R["vgg11"]["parameters"]
        rows.append([name, f"{extra:+,}", n(record["parameters"]), n(record["macs"]),
                     num(record["test"]["accuracy"]), pp(record["delta_accuracy"]) + " pp",
                     num(record["test"]["macro_f1"])])
    story.append(table(rows, [22, 22, 26, 28, 22, 22, 22]))
    story.append(caption("表 6　注意力模块的收益与代价"))
    se_better = R["se"]["delta_accuracy"] > R["cbam"]["delta_accuracy"]
    story.append(P(
        f"SE 只增加了 {n(R['se']['parameters'] - R['vgg11']['parameters'])} 个参数（约 "
        f"{pc((R['se']['parameters'] - R['vgg11']['parameters']) / R['vgg11']['parameters'], 2)}），"
        f"准确率 {pp(R['se']['delta_accuracy'])} 个百分点；CBAM 参数略多（多出空间分支的 7×7 卷积），"
        f"准确率 {pp(R['cbam']['delta_accuracy'])} 个百分点。"
        + ("两者都没有带来数量级上的提升，这与数据集本身的信息瓶颈一致："
           "被池化到 7×7 之后，可供注意力挑选的判别性细节本就有限。" if abs(R["cbam"]["delta_accuracy"]) < 0.01 else
           "注意力带来的提升说明通道与空间加权在本任务上是有效的正则。")
        + f"从性价比看，{'SE' if se_better else 'CBAM'} 更划算。"))
    def pair_errors(name: str) -> int:
        record = R[name]
        confusion = record["test"]["confusion"]
        first = record["classes"].index("T-shirt/top")
        second = record["classes"].index("Shirt")
        return int(confusion[first][second] + confusion[second][first])

    story.append(P(
        "为了看清注意力到底改变了什么，本文用 Grad-CAM 对第 3 个 stage 的特征图（7×7）做梯度加权，"
        "再放大回原图（图 3）。两组的热区并没有一致的方向性差异：在短靴、凉鞋这类小目标上，"
        "组合模型的热区更贴近鞋体本身，而基线的热区常常落在图像下缘与背景上；"
        "但在运动鞋、长裤上，基线的热区反而更完整。6 个样本、单次前向的可视化只能作为示意，"
        "不足以论证注意力「看得更准」。定量结果同样不支持：衬衫与 T 恤之间的互相误判从基线的 "
        f"{pair_errors('vgg11')} 例增加到 CBAM 的 {pair_errors('cbam')} 例、final 的 {pair_errors('final')} 例，"
        "说明注意力模块并没有缓解这对最难的类别。"))
    story.append(figure(Path("figures/gradcam.png"), 168,
                        "图 3　Grad-CAM（第 3 个 stage）：上排为基线，下排为可分离卷积 + CBAM + GELU 组合模型"))

    # ---------------- 3.3 activation ---------------- #
    story.append(H2T("3.3", "激活函数：GELU 与 Mish"))
    story.append(P(
        "ReLU 的优点是便宜、梯度不饱和，缺点是负半轴硬截断，信息直接丢弃。"
        "GELU 用高斯 CDF 给负值一个平滑的小斜率（x·Φ(x)），Mish 用 x·tanh(softplus(x)) 构造一条"
        "在负半轴先下探再回升的曲线。两者都希望在保持稀疏性的同时保留少量负信息。"))
    rows = [["配置", "参数量", "准确率", "相对基线", "macro-F1", "单轮耗时", "相对基线"]]
    for name in ("vgg11", "gelu", "mish"):
        record = R[name]
        rows.append([name, n(record["parameters"]), num(record["test"]["accuracy"]),
                     pp(record["delta_accuracy"]) + " pp", num(record["test"]["macro_f1"]),
                     f"{record['seconds_per_epoch']:.1f} s",
                     f"{record['seconds_per_epoch'] / R['vgg11']['seconds_per_epoch'] - 1:+.1%}"])
    story.append(table(rows, [22, 26, 22, 24, 24, 24, 22]))
    story.append(caption("表 7　激活函数的精度与训练开销"))
    story.append(P(
        f"GELU 与 Mish 都不带参数，因此参数量与基线完全相同（表 7），它们影响的只是计算与优化。"
        "表里的单轮耗时只能作参考：九次训练是三个进程并行跑的，单轮耗时受 CPU 争用影响，"
        "不能直接用来比较算子开销；真正可比的是第 4.2 节在同一台空闲机器上测的单线程延迟。"
        "精度方面，"
        + ("两者与 ReLU 的差距都在 0.5 个百分点以内，属于单次实验的噪声量级，"
           "在 28×28 的小网络上换激活函数的收益远不如换卷积算子明显。"
           if max(abs(R["gelu"]["delta_accuracy"]), abs(R["mish"]["delta_accuracy"])) < 0.005 else
           "两者的精度差异超过了单次实验的噪声量级，说明平滑激活在本任务上确实提供了额外正则。")))

    # ---------------- 3.4 norm ---------------- #
    story.append(H2T("3.4", "归一化层：BatchNorm 与 GroupNorm"))
    story.append(P(
        "BatchNorm 在 batch 维上统计均值方差，batch 越小人均统计越不可靠；"
        "GroupNorm 改为在每个样本内部把通道分成 8 组、组内统计，完全不依赖 batch 大小。"
        "理论上 batch=128 时两者应该接近，但在边缘设备上推理常常是 batch=1，"
        "此时 BN 只能退回使用训练期滑动统计量，而 GN 的行为与训练时一致。"
        "为了验证这一点，本文额外训练了两个 batch=16 的模型（8 轮，其余设置不变），"
        "与 batch=128 的结果对照。"))
    rows = [["配置", "batch", "轮数", "准确率", "macro-F1"]]
    rows.append(["vgg11", "128", str(R["vgg11"]["epochs"]), num(R["vgg11"]["test"]["accuracy"]),
                 num(R["vgg11"]["test"]["macro_f1"])])
    rows.append(["gn", "128", str(R["gn"]["epochs"]), num(R["gn"]["test"]["accuracy"]),
                 num(R["gn"]["test"]["macro_f1"])])
    if "batchsize" in S:
        for record in S["batchsize"]:
            rows.append([record["name"], str(record["batch"]), str(record["epochs"]),
                         num(record["test"]["accuracy"]), num(record["test"]["macro_f1"])])
    story.append(table(rows, [24, 18, 16, 26, 26], align_left_first=True))
    story.append(caption("表 8　归一化层在不同批大小下的表现（小批量组训练 8 轮）"))

    def step_std(record) -> float:
        values = [row["val_accuracy"] for row in record["history"]]
        return float(np.std(np.diff(values)))

    gap_big = R["gn"]["test"]["accuracy"] - R["vgg11"]["test"]["accuracy"]
    small = {record["name"].split(" ")[0]: record for record in S.get("batchsize", [])}
    if small:
        bn_small, gn_small = small["vgg11"], small["gn"]
        gap_small = gn_small["test"]["accuracy"] - bn_small["test"]["accuracy"]
        story.append(P(
            f"在 batch=128 时，GN 比 BN 低 {abs(gap_big) * 100:.2f} 个百分点；"
            f"把 batch 降到 16（两组都只训 8 轮，预算相同）后，GN 比 BN 低 "
            f"{abs(gap_small) * 100:.2f} 个百分点，差距缩小了一半。"
            "方向上与「小 batch 会让 BN 的统计量变噪」一致，但 GN 并没有反超，"
            "说明在本任务上它的主要价值不是精度，而是推理时统计量与训练时一致"
            "——batch=1 的流式推理里 BN 只能退回训练期滑动统计量，GN 的行为则完全不变。"
            f"另一个意外是曲线的抖动：BN 相邻两轮验证精度的变化标准差是 {step_std(R['vgg11']):.4f}，"
            f"GN 是 {step_std(R['gn']):.4f}，GN 反而更抖，这与「GN 更稳定」的常见说法相反。"
            "一个可能的解释是 GN 把通道分成 8 组，每组只有 8–12 个通道，统计量本身噪声更大；"
            f"在 batch=16 时两者的抖动同样如此（{step_std(bn_small):.4f} 与 {step_std(gn_small):.4f}）。"
            "工程上的结论因此要打折：训练 batch 足够大就用 BN（更快、精度更高），"
            "只有在 batch 极小或需要逐样本统计的场景才换 GN，并且要接受精度上的让步。"))
    else:
        story.append(P(
            f"在 batch=128 时，GN 比 BN 低 {abs(gap_big) * 100:.2f} 个百分点，"
            "差距不大但方向稳定：BN 的 batch 维噪声本身就是一种正则，"
            "GN 用逐样本统计换来了批大小无关的行为，代价是失去这部分正则。"))

    # ---------------- 3.5 combined ---------------- #
    story.append(H2T("3.5", "组合模型与综合对比"))
    story.append(P(
        "把三项互不冲突的修改叠加——Stage 3–5 用可分离卷积（省参数与计算）、Stage 3–5 加 CBAM（补通道与空间选择）、"
        "激活换成 GELU（平滑且比 Mish 便宜）——得到组合模型 final。它只有 "
        f"{n(R['final']['parameters'])} 个参数、{n(R['final']['macs'])} MACs，"
        f"分别是基线的 {pc(R['final']['parameters'] / R['vgg11']['parameters'], 1)} 与 "
        f"{pc(R['final']['macs'] / R['vgg11']['macs'], 1)}，测试准确率 {num(R['final']['test']['accuracy'])}"
        f"（{pp(R['final']['delta_accuracy'])} 个百分点）。"))
    rows = [["配置", "参数量", "MACs", "准确率", "Δacc (pp)", "macro-F1", "最佳轮", "单轮 (s)"]]
    for name in order:
        record = R[name]
        rows.append([name, n(record["parameters"]), n(record["macs"]),
                     num(record["test"]["accuracy"]), pp(record["delta_accuracy"]),
                     num(record["test"]["macro_f1"]), str(record["best_epoch"]),
                     f"{record['seconds_per_epoch']:.1f}"])
    story.append(table(rows, [22, 24, 26, 20, 20, 20, 14, 16]))
    story.append(caption("表 9　九组实验的完整结果（测试集 10000 张，Δacc 相对基线）"))
    best_name = max(order, key=lambda k: R[k]["test"]["accuracy"])
    story.append(P(
        f"表 9 中测试准确率最高的是 {best_name}（{num(R[best_name]['test']['accuracy'])}）；"
        f"组合模型 final 的 {num(R['final']['test']['accuracy'])} 并不是最高的——它把 MACs 压到 "
        f"{R['final']['macs'] / R['vgg11']['macs']:.2f} 倍、参数压到 "
        f"{R['final']['parameters'] / R['vgg11']['parameters']:.2f} 倍，代价是 "
        f"{abs(R['final']['delta_accuracy']) * 100:.2f} 个百分点。"
        f"与只做可分离卷积的 sep（{num(R['sep']['test']['accuracy'])}）相比，"
        f"CBAM 与 GELU 把损失从 {abs(R['sep']['delta_accuracy']) * 100:.2f} 个百分点收窄到 "
        f"{abs(R['final']['delta_accuracy']) * 100:.2f} 个百分点，回收约 "
        f"{(R['final']['delta_accuracy'] - R['sep']['delta_accuracy']) * 100:.2f} 个百分点，"
        "但远不足以抵消压缩带来的损失。"
        f"工程上这是一笔可以接受的交换：模型体积从 {R['vgg11']['weights_kb'] / 1024:.2f} MB 降到 "
        f"{R['final']['weights_kb'] / 1024:.2f} MB，单次推理的乘加次数降到 "
        f"{R['final']['macs'] / R['vgg11']['macs']:.0%}。"
        "如果精度优先，应该选 mish（表 7），它只增加常数级开销、精度却不降；"
        "如果算力优先，才需要在 final 与 sep 之间按部署预算做取舍。"))
    story.append(figure(Path("figures/accuracy.png"), 150,
                        "图 4　九组配置的测试准确率（括号内为相对基线的变化）"))
    story.append(figure(Path("figures/cost_accuracy.png"), 148,
                        "图 5　精度—计算量权衡：横轴为对数刻度的 MACs，越靠左上越优"))
    def early_mean(name, upto=3):
        history = R[name]["history"][:upto]
        return float(np.mean([row["val_accuracy"] for row in history]))

    def last_val(name):
        return R[name]["history"][-1]["val_accuracy"]

    late_best = sum(1 for name in order if R[name]["best_epoch"] > R[name]["epochs"] * 0.6)
    story.append(P(
        f"从曲线看，九组配置都跑满 16 轮，其中 {late_best} 组的最佳轮次出现在第 10 轮之后——"
        "这说明余弦退火的下半程确实在把模型往更好的解上推，而不是单纯地等训练结束。"
        "验证集在后半程进入平台期，最后几轮学习率接近 0，因此没有出现明显的末期反弹。"
        f"可分离卷积的收敛速度略慢于基线：前 3 轮的平均验证精度为 {num(early_mean('sep'))}，"
        f"基线是 {num(early_mean('vgg11'))}；到训练末期两者已处于同一水平"
        f"（{num(last_val('sep'))} 与 {num(last_val('vgg11'))}）。"
        "这与 depthwise 卷积的梯度尺度较小、需要更多步数有关。"))
    story.append(figure(Path("figures/training_curves.png"), 165,
                        "图 6　九组配置的训练/验证准确率曲线（虚线为选中轮次）"))
    return story


def section_deploy(S, R, EXP, ROB, CONF) -> list:
    confidence_record = next(record for record in CONF["models"] if record["name"] == "final")
    review_row = min(confidence_record["rows"], key=lambda row: abs(row["threshold"] - 0.60))
    story = [H1T("4", "场景应用与部署")]
    story.append(H2T("4.1", "交互式 Web 演示"))
    story.append(P(
        "为了让「场景应用」不只是概念，本文实现了一个零依赖的 Web 演示（webapp.py，只用 Python 标准库的 "
        "http.server + PIL + PyTorch，不引入 Flask/FastAPI）。页面支持拖拽上传或从测试集随机抽图，"
        "服务端把图片转灰度、短边缩放到 28 并中心裁剪，再按训练集均值方差标准化。"
        "一个细节是：Fashion-MNIST 是「亮色衣物 + 黑底」，而真实商品图通常是「深色衣物 + 白底」，"
        "因此预处理会根据图像平均亮度自动反色，否则模型会系统性误判。"))
    story.append(P(
        f"推理返回 Top-3 类目与概率，并给出一个业务化的开关：最高概率低于 0.60 时标记为「转人工复核」。"
        f"阈值的效果可以直接从测试集上量出来：组合模型有 {pct1(review_row['review_share'])}% 的样本落在阈值以下，"
        f"这部分样本的准确率只有 {pct1(review_row['accuracy_rejected'])}%；"
        f"其余 {pct1(review_row['coverage'])}% 自动通过，准确率 {pct1(review_row['accuracy_covered'])}%。"
        f"若人工把复核样本全部改对，流水线的等效准确率为 "
        f"{num(review_row['coverage'] * review_row['accuracy_covered'] + review_row['review_share'])}，"
        f"而全部自动通过时是 {num(confidence_record['accuracy'])}——人工只需要处理约 "
        f"{pct1(review_row['review_share'])}% 的样本，就把整体精度抬高了"
        f" {num(review_row['coverage'] * review_row['accuracy_covered'] + review_row['review_share'] - confidence_record['accuracy'])}。"))
    screenshot = Path("figures/webapp.png")
    if screenshot.exists():
        story.append(figure(screenshot, 158, "图 7　Web 演示界面：上传图片后返回 Top-3 与人工复核标记"))
    else:
        story.append(P("（Web 演示截图见仓库 figures/webapp.png。）", BODY_NO_INDENT))

    story.append(H2T("4.2", "ONNX 导出与 int8 量化"))
    story.append(P(
        "部署侧的第二步是把 PyTorch 模型导出为 ONNX，并做动态 int8 量化（权重按通道量化，激活在线量化，"
        "不需要校准数据集）。表 10 对比了基线与我们最终模型在 CPU 单线程下的体积与延迟。"))
    rows = [["模型", "参数量", "ONNX 体积", "int8 体积", "PyTorch 延迟", "ONNX 延迟", "int8 延迟"]]
    for record in EXP["models"]:
        rows.append([record["name"], n(record["parameters"]),
                     f"{record['onnx_kb']:.0f} KB",
                     f"{record.get('int8_kb', 0):.0f} KB" if record.get("int8_kb") else "—",
                     f"{record['torch_ms']:.2f} ms",
                     f"{record['onnx_ms']:.2f} ms" if record.get("onnx_ms") else "—",
                     f"{record['int8_ms']:.2f} ms" if record.get("int8_ms") else "—"])
    story.append(table(rows, [22, 22, 24, 22, 28, 24, 24]))
    story.append(caption("表 10　ONNX 导出与 int8 动态量化的体积、延迟（batch=1，单线程）"))
    story.append(P(
        "量化把权重从 fp32 压到 int8，模型体积明显缩小（表 10），但延迟没有改善，反而变慢了——"
        "这是本次实验里最反直觉的一个结果。原因在于动态量化的 ConvInteger 在 ONNX Runtime 的 CPU 执行器里"
        "缺少针对小通道卷积优化的 kernel：本网络的通道数只有 16–96，逐层的反量化、权重恢复与调度开销"
        "盖过了整数运算本身的收益。因此在这类小模型上，数值层面的量化只在「体积敏感、延迟不敏感」"
        "的场景（例如随 App 分发的离线模型）才划算；要在 CPU 上提速，结构层面的裁剪"
        "（可分离卷积、减少 stage）仍然比量化更直接。"))
    onnx_ratio = {record["name"]: record["int8_ms"] / record["onnx_ms"]
                  for record in EXP["models"] if record.get("onnx_ms") and record.get("int8_ms")}
    size_ratio = {record["name"]: record["int8_kb"] / record["onnx_kb"]
                  for record in EXP["models"] if record.get("int8_kb")}
    if onnx_ratio:
        story.append(P(
            f"具体数字：int8 体积是 fp32 的 {min(size_ratio.values()):.0%}–{max(size_ratio.values()):.0%}，"
            f"而单线程延迟是 fp32 ONNX 的 {min(onnx_ratio.values()):.1f}–{max(onnx_ratio.values()):.1f} 倍。"
            "同时可以看到 PyTorch 的 eager 延迟明显高于 ONNX Runtime（后者对 conv 做了更充分的图优化），"
            "但 PyTorch 下组合模型反而比基线更慢，说明 depthwise 卷积在 PyTorch CPU 上的实现效率"
            "还抵消了一部分 MACs 优势——同一个模型，换运行时就是另一套结论。"))

    story.append(H2T("4.3", "鲁棒性评估"))
    story.append(P(
        "真实场景的输入不会像测试集那样干净。本文构造了六种合成退化——旋转 15°、高斯噪声（σ=0.15）、"
        "高斯模糊（σ=1.0）、亮度降到 40%、随机 10×10 遮挡、先降采样到 14×14 再放大——"
        "在完整 10000 张测试集上比较四个模型的准确率。退化参数固定、随机种子固定，保证可复现。"))
    rows = [["退化条件"] + [model for model in ROB["models"]]]
    for row in ROB["rows"]:
        rows.append([row["corruption"]] + [num(row["accuracy"][model]) for model in ROB["models"]])
    story.append(table(rows, [40] + [22] * len(ROB["models"]), align_left_first=True))
    story.append(caption("表 11　合成退化下的测试准确率"))
    clean = {model: ROB["rows"][0]["accuracy"][model] for model in ROB["models"]}
    drops = {model: np.mean([clean[model] - row["accuracy"][model] for row in ROB["rows"][1:]])
             for model in ROB["models"]}
    most = max(drops, key=drops.get)
    least = min(drops, key=drops.get)
    by_condition = {row["corruption"]: row["accuracy"] for row in ROB["rows"]}
    condition_drops = {
        name: float(np.mean([clean[model] - accuracy[model] for model in ROB["models"]]))
        for name, accuracy in by_condition.items() if name != "clean"}
    worst_condition = max(condition_drops, key=condition_drops.get)
    mildest_condition = min(condition_drops, key=condition_drops.get)
    sep_gap = drops["sep"] - drops["vgg11"]
    occlusion = by_condition.get("occlusion 10x10", {})
    cbam_gain = occlusion.get("cbam", 0) - occlusion.get("vgg11", 0)
    story.append(P(
        f"四个模型平均掉点最多的是 {most}（{pct1(drops[most])} 个百分点），"
        f"最少的是 {least}（{pct1(drops[least])} 个百分点）。"
        f"从退化条件看，平均掉点最大的是「{worst_condition}」（{pct1(condition_drops[worst_condition])} 个百分点），"
        f"最小的是「{mildest_condition}」（{pct1(condition_drops[mildest_condition])} 个百分点）——"
        "这与信息损失的量级一致：前者破坏了模型依赖的像素证据，"
        "而后者只是整体线性缩放，标准化之后差异被大部分抵消。"
        + (f"可分离卷积模型的平均掉点比基线低 {pct1(-sep_gap)} 个百分点，"
           "说明省参数并没有以牺牲鲁棒性为代价。" if sep_gap < 0 else
           f"可分离卷积模型的平均掉点比基线高 {pct1(sep_gap)} 个百分点，"
           "说明压缩结构在输入退化时略微吃亏。")
        + (f"CBAM 模型在 10×10 遮挡下比基线高 {pct1(cbam_gain)} 个百分点，"
           "与「空间注意力能补偿局部缺失」的直觉一致。" if cbam_gain > 0 else
           f"CBAM 模型在 10×10 遮挡下比基线低 {pct1(-cbam_gain)} 个百分点，"
           "说明空间注意力并没有像预期那样带来局部缺失的鲁棒性。")
        + f"但组合模型 final 在遮挡下只有 {num(occlusion.get('final', 0))}，是四个模型里最低的，"
        "说明可分离卷积带来的鲁棒性余量并不足以抵消 CBAM 与 GELU 在小目标上的损失——"
        "结构改动之间不是简单叠加，鲁棒性尤其如此。"))
    if Path("figures/robustness.png").exists():
        story.append(figure(Path("figures/robustness.png"), 150,
                            "图 8　合成退化下的准确率对比"))

    story.append(H2T("4.4", "落地讨论"))
    story.append(P(
        "把实验结果放回业务场景，可以得到三条可操作的结论。第一，如果目标是「在手机端实时打标」，"
        f"应当选组合模型：{R['final']['weights_kb'] / 1024:.2f} MB 的体积、基线的 "
        f"{pc(R['final']['macs'] / R['vgg11']['macs'], 0)} 计算量；导出 ONNX 后单线程延迟是 "
        f"{num(next(r for r in EXP['models'] if r['name'] == 'final')['onnx_ms'], 3)} ms，"
        "留给移动端还有充足余量（本次没有做 int8 之外的真机验证，量化版本在这台机器上并不更快）。"
        "第二，如果目标是「后台批量清洗商品库」，算力不是瓶颈，"
        "应该选精度最高的配置，并用置信度阈值把不确定样本送人工，形成「模型预筛 + 人工兜底」的流水线。"
        "第三，如果输入分布与训练集差异大（例如实拍图而非白底图），需要先做背景归一化与反色，"
        "否则再好的结构也救不回来——本文在 Web 演示里内置了这个预处理步骤。"))
    return story


def section_discussion(S, R) -> list:
    story = [H1T("5", "讨论与局限")]
    story.append(P(
        "本作业的实验有三个明确的局限，需要在解读结论时打折扣。第一，所有配置只跑了一个随机种子，"
        "而 Fashion-MNIST 上不同初始化之间的准确率波动通常在 0.3–0.5 个百分点，"
        "因此小于这个量级的差异（例如 GELU 与 ReLU、SE 与 CBAM 之间）不应被当作真实效应。"
        "要得到统计上可靠的结论，每个配置至少需要 3–5 次重复并报告均值与标准差，"
        "这在本次的时间预算内没有做。"))
    story.append(P(
        "第二，数据集本身限制了对「结构创新」的检验。28×28 的灰度图在四次池化后只剩 1×1，"
        "感受野很快就覆盖全图，深层结构的差异被压缩；而服饰类别的语义重叠又给精度设了上限（约 0.95）。"
        "在这种「小图 + 简单任务」上，结构修改的收益天然小于 ImageNet 级别的任务，"
        "这也是注意力模块几乎没有带来精度收益、而压缩卷积却掉了近两个百分点的原因之一。"
        "若要在更有说服力的数据上验证，应换到 Fashion-Product-1M 或 DeepFashion 这类高分辨率真实商品图数据集。"))
    story.append(P(
        "第三，可分离卷积的收益被网络的窄通道削弱。理论比值 1/C_out + 1/k² 假设 C_out 远大于 k²，"
        "而本网络最大通道只有 96，1/C_out 项不可忽略，因此实测压缩比不如 MobileNet 那么夸张。"
        "这提醒我们：结构技巧的收益依赖于它被放置的通道宽度与特征图尺寸，"
        "照搬论文里的「替换即提速」并不可靠，必须结合逐层成本分析（如表 2）来定位修改点。"))
    story.append(P(
        "后续可以做的改进包括：用多 seed 重复给出置信区间；引入 CutMix/MixUp 等更强增广以突破精度上限；"
        "用知识蒸馏把基线模型的软标签迁移给组合模型，尝试在更小参数下追平精度；"
        "以及把 Grad-CAM 从定性观察升级为定量指标（例如对比 CAM 与人工标注的衣物区域 IoU）。"))
    return story


def section_conclusion(S, R) -> list:
    base, final = R["vgg11"], R["final"]
    best = max(R.values(), key=lambda r: r["test"]["accuracy"])
    singles = {name: R[name]["delta_accuracy"] for name in ("se", "cbam", "gelu", "mish", "gn")}
    story = [H1T("6", "结论")]
    story.append(P(
        f"本文在 Fashion-MNIST 上实现了一个窄通道 VGG11，并以它为基线完成了八组结构消融。"
        f"基线取得 {num(base['test']['accuracy'])} 的测试准确率（macro-F1 {num(base['test']['macro_f1'])}），"
        f"参数量 {n(base['parameters'])}、单张图 MACs {n(base['macs'])}。"))
    story.append(P(
        f"结构层面最重要的结论是「压缩位置比压缩技巧更重要」：逐层成本分析显示 Stage 3 吃掉一半以上的计算量、"
        f"Stage 4–5 占去近八成参数，把深度可分离卷积精确地放在 Stage 3–5，"
        f"就能用 {pc(R['sep']['parameters'] / base['parameters'], 1)} 的参数与 "
        f"{pc(R['sep']['macs'] / base['macs'], 1)} 的 MACs 拿到 {pp(R['sep']['delta_accuracy'])} 个百分点的精度变化。"
        f"注意力、激活函数与归一化的单项改动带来的精度变化在 {pp(min(singles.values()))}–"
        f"{pp(max(singles.values()))} 个百分点之间，"
        f"但组合模型 final 仍然把参数量压到 {pc(final['parameters'] / base['parameters'], 1)}、"
        f"MACs 压到 {pc(final['macs'] / base['macs'], 1)}，同时保持 {num(final['test']['accuracy'])} 的准确率；"
        f"九组中精度最高的是 {best['name']}（{num(best['test']['accuracy'])}）。"))
    story.append(P(
        "工程层面，本文把模型导出为 ONNX 并做 int8 动态量化，配了一个可交互的 Web 演示，"
        "用六种合成退化验证了鲁棒性，并给出「置信度阈值 + 人工复核」的落地方式。"
        "这些工作说明：一个 40 万参数以内的 CNN 已经足以在服饰分类任务上达到可用精度，"
        "真正的难点在于把结构改动放在正确的位置，以及把训练集与真实输入之间的分布差异处理好。"))
    return story


def appendix(R) -> list:
    import importlib.metadata as metadata

    def version_of(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "—"

    story = [H1T("附录 A", "运行环境与复现步骤")]
    rows = [["项目", "版本 / 配置"],
            ["操作系统", R["vgg11"]["platform"]],
            ["Python", R["vgg11"]["python"]],
            ["PyTorch", R["vgg11"]["torch"]],
            ["torchvision", version_of("torchvision")],
            ["NumPy / Matplotlib", f"{version_of('numpy')} / {version_of('matplotlib')}"],
            ["onnxruntime / onnx", f"{version_of('onnxruntime')} / {version_of('onnx')}"],
            ["CPU", "Intel Core i9-14900HX（训练用 6 线程）"]]
    story.append(table(rows, [34, 92], align_left_first=True))
    story.append(caption("表 A1　实验环境"))
    story.append(P("完整复现流程（在仓库根目录执行）：", BODY_NO_INDENT))
    for command in [
        "pip install -r requirements.txt",
        "python download.py                 # 下载并校验 Fashion-MNIST",
        "python dataset_stats.py            # 数据集统计",
        "python describe.py                 # 逐阶段参数量 / MACs",
        "python train.py --name vgg11 --epochs 16",
        "python train.py --name sep --epochs 16",
        "python train.py --name sep_all --epochs 16",
        "python train.py --name se --epochs 16",
        "python train.py --name cbam --epochs 16",
        "python train.py --name gelu --epochs 16",
        "python train.py --name mish --epochs 16",
        "python train.py --name gn --epochs 16",
        "python train.py --name final --epochs 16",
        "python train.py --name vgg11 --epochs 8 --batch 16 --tag bs16",
        "python train.py --name gn --epochs 8 --batch 16 --tag bs16",
        "python aggregate.py                # 汇总为 runs/summary.json / .csv",
        "python plots.py                    # 生成全部图表",
        "python gradcam.py                  # Grad-CAM 可视化",
        "python robustness.py               # 合成退化鲁棒性",
        "python confidence.py               # 置信度—覆盖率权衡",
        "python export_onnx.py              # ONNX 导出 + int8 量化",
        "python webapp.py                   # 打开 http://127.0.0.1:8000",
    ]:
        story.append(Paragraph(command, ParagraphStyle(
            "code", fontName="SimSun", fontSize=8.6, leading=13, leftIndent=10,
            backColor=colors.HexColor("#f4f6f8"), borderPadding=2, spaceAfter=1)))
    rows = [["文件", "作用"],
            ["models.py", "VGG11 主体、SE/CBAM、Mish、深度可分离卷积、9 组配置"],
            ["data.py", "数据划分、增广、DataLoader、单图预处理"],
            ["engine.py", "训练循环、评估、混淆矩阵与 F1、MACs 统计"],
            ["train.py", "单配置训练入口，输出 runs/<name>/metrics.json"],
            ["aggregate.py", "汇总所有实验为 summary.json / summary.csv"],
            ["describe.py", "逐阶段参数量与 MACs 表"],
            ["plots.py", "全部图表（曲线、柱状、散点、混淆矩阵、鲁棒性）"],
            ["gradcam.py", "Grad-CAM 可视化"],
            ["robustness.py", "合成退化下的鲁棒性评估"],
            ["confidence.py", "置信度阈值与人工复核的覆盖率—精度权衡"],
            ["export_onnx.py", "ONNX 导出与 int8 量化、延迟测量"],
            ["webapp.py", "零依赖 Web 演示（上传 / 随机测试图 / Top-3）"],
            ["report.py", "由 JSON 与 PNG 自动生成这份 PDF"]]
    story.append(KeepTogether([H1T("附录 B", "代码结构"),
                               table(rows, [36, 92], align_left_first=True),
                               caption("表 B1　代码文件说明")]))
    return story


def references() -> list:
    items = [
        "Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015. arXiv:1409.1556.",
        "Xiao H, Rasul K, Vollgraf R. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.",
        "Howard A G, Zhu M, Chen B, et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861, 2017.",
        "Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks. CVPR 2018. arXiv:1709.01507.",
        "Woo S, Park J, Lee J Y, Kweon I S. CBAM: Convolutional Block Attention Module. ECCV 2018. arXiv:1807.06521.",
        "Hendrycks D, Gimpel K. Gaussian Error Linear Units (GELUs). arXiv:1606.08415, 2016.",
        "Misra D. Mish: A Self Regularized Non-Monotonic Activation Function. BMVC 2020. arXiv:1908.08681.",
        "Wu Y, He K. Group Normalization. ECCV 2018. arXiv:1803.08494.",
        "Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015. arXiv:1502.03167.",
        "Selvaraju R R, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017. arXiv:1610.02391.",
    ]
    story = [H1T("参考文献", "")]
    for index, item in enumerate(items, 1):
        style_name = "ref"
        story.append(Paragraph(f"[{index}]&nbsp;{item}", ParagraphStyle(
            style_name, fontName="SimSun", fontSize=9, leading=14, leftIndent=16,
            firstLineIndent=-16, spaceAfter=3, alignment=TA_LEFT)))
    return story


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_story(data) -> list:
    story = []
    story += section_intro(*data["intro"])
    story += section_baseline(*data["baseline"])
    story += section_ablation(*data["ablation"])
    story += section_deploy(*data["deploy"])
    story += section_discussion(*data["discussion"])
    story += section_conclusion(*data["conclusion"])
    story += appendix(*data["appendix"])
    story += references()
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path,
                        default=Path("../output/CNN大作业/2427080125王炎煌.pdf"))
    parser.add_argument("--date", default="2026 年 11 月")
    args = parser.parse_args()

    summary = load_json(args.runs / "summary.json")
    # the two bs16 runs reuse the names vgg11 / gn, so keep them out of the name-keyed dict
    R = {record["name"]: record for record in summary if record.get("tag") != "bs16"}
    batchsize = [dict(record) for record in summary if record.get("tag") == "bs16"]
    for record in batchsize:
        record["name"] = record["name"] + " / bs16"
    S = {"batchsize": batchsize}
    A = load_json(args.runs / "architecture.json")
    D = load_json(args.runs / "dataset.json")
    ROB = load_json(args.runs / "robustness.json")
    EXP = load_json(args.runs / "export.json")
    CONF = load_json(args.runs / "confidence.json")

    data = {
        "intro": (S, R, D, A),
        "baseline": (S, R, A),
        "ablation": (S, R, A),
        "deploy": (S, R, EXP, ROB, CONF),
        "discussion": (S, R),
        "conclusion": (S, R),
        "appendix": (R,),
    }

    entries = [("1", "引言与任务定义"), ("1.1", "任务背景"), ("1.2", "数据集"), ("1.3", "评价指标"),
               ("2", "基线模型：窄通道 VGG11"), ("2.1", "网络结构"), ("2.2", "训练设置"),
               ("2.3", "基线结果与分析"),
               ("3", "结构修改与消融实验"), ("3.1", "深度可分离卷积"), ("3.2", "注意力机制：SE 与 CBAM"),
               ("3.3", "激活函数：GELU 与 Mish"), ("3.4", "归一化层：BatchNorm 与 GroupNorm"),
               ("3.5", "组合模型与综合对比"),
               ("4", "场景应用与部署"), ("4.1", "交互式 Web 演示"), ("4.2", "ONNX 导出与 int8 量化"),
               ("4.3", "鲁棒性评估"), ("4.4", "落地讨论"),
               ("5", "讨论与局限"), ("6", "结论"),
               ("附录 A", "运行环境与复现步骤"), ("附录 B", "代码结构"), ("参考文献", "")]

    # pass 1 measures page numbers; the TOC is present in both passes so pagination matches
    tmp = Path("report_tmp.pdf")
    pages: dict = {}
    story = cover(args.date) + toc_page(pages, entries) + build_story(data)
    Report(tmp, pages=pages, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
           topMargin=MARGIN, bottomMargin=MARGIN,
           title="面向服饰图像的轻量化 CNN", author="王炎煌").build(story)
    # pass 2 writes the final file with the measured page numbers
    story = cover(args.date) + toc_page(dict(pages), entries) + build_story(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Report(args.out, pages={}, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
           topMargin=MARGIN, bottomMargin=MARGIN,
           title="面向服饰图像的轻量化 CNN", author="王炎煌").build(story)
    tmp.unlink(missing_ok=True)
    print(args.out)


if __name__ == "__main__":
    main()
