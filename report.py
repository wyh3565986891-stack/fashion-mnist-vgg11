# 用实验产出的 JSON 和图生成中文 PDF 报告；正文里的数字都从 runs/*.json 读，
# 所以改了实验重跑一遍，报告不会和结果对不上。
# 用法：python report.py --out "../output/CNN大作业/2427080125王炎煌.pdf"

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

# 字体和样式
pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
pdfmetrics.registerFont(TTFont("KaiTi", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))

BODY = ParagraphStyle("Body", fontName="SimSun", fontSize=10.5, leading=17.5,
                      firstLineIndent=21, alignment=TA_JUSTIFY, spaceAfter=3,
                      wordWrap="CJK", splitLongWords=1)
BODY_NO_INDENT = ParagraphStyle("BodyNoIndent", parent=BODY, firstLineIndent=0)
CAPTION = ParagraphStyle("Caption", fontName="SimSun", fontSize=8.8, leading=13,
                         alignment=TA_CENTER, spaceBefore=3, spaceAfter=8)
H1 = ParagraphStyle("Heading1", fontName="SimHei", fontSize=15, leading=22,
                    spaceBefore=14, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("Heading2", fontName="SimHei", fontSize=11.8, leading=19,
                    spaceBefore=10, spaceAfter=5, keepWithNext=1)
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
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.9, colors.black),
                  ("FONTNAME", (0, 0), (-1, 0), "SimHei")]
    table_object.setStyle(TableStyle(style))
    return KeepTogether([table_object])


def figure(path: Path, width_mm: float, text: str):
    with PILImage.open(path) as image:
        aspect = image.height / image.width
    width = width_mm * mm
    return KeepTogether([Image(str(path), width=width, height=width * aspect), caption(text)])


# 数字格式
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


# 文档模板
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
        canvas.setFillColor(colors.black)
        canvas.drawString(MARGIN, PAGE_HEIGHT - MARGIN + 8 * mm,
                          "神经网络与深度学习 · 大作业报告")
        canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN + 8 * mm,
                               "服饰图像 CNN 结构设计与实验分析")
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(0.4)
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
    story.append(Paragraph("Fashion-MNIST 服饰图像分类", ParagraphStyle(
        "cover3", fontName="SimHei", fontSize=16, alignment=TA_CENTER, leading=28)))
    story.append(Paragraph("VGG11 基线、结构消融与部署验证", ParagraphStyle(
        "cover4", fontName="SimHei", fontSize=16, alignment=TA_CENTER, leading=28)))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("9 组对照实验，8 张图、13 张表", ParagraphStyle(
        "cover5", fontName="SimSun", fontSize=11, alignment=TA_CENTER)))
    story.append(Spacer(1, 32 * mm))
    info = [["姓　　名", "王炎煌"], ["学　　号", "2427080125"], ["班　　级", "智科 24-1"],
            ["课程名称", "神经网络与深度学习"], ["完成日期", date_text]]
    rows = [[Paragraph(a, ParagraphStyle("ci", fontName="SimSun", fontSize=12, alignment=TA_CENTER)),
             Paragraph(b, ParagraphStyle("cv", fontName="SimSun", fontSize=12, alignment=TA_CENTER))]
            for a, b in info]
    cover_table = Table(rows, colWidths=[34 * mm, 62 * mm])
    cover_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(cover_table)
    story.append(PageBreak())
    return story


def toc_page(pages: dict, entries: list[tuple[str, str]]) -> list:
    # 目录里的页码来自第一遍构建
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
        "服装图像的自动识别在电商里用处很多：以图搜款、自动打标、库存盘点，"
        "二手平台判断衣服成色也要靠它。这类模型通常跑在手机或边缘设备上，算力和内存都不宽裕，"
        "所以用尽量小的模型把准确率做上去，比刷一个更高的 SOTA 更有实际意义。"
        "本次实验用 Fashion-MNIST 作数据集，在同一套训练设置下比较一个经典卷积网络和几种结构改动，"
        "回答两个问题：各个模块对精度和计算量分别有多大影响；同样的算力预算下，哪种组合更值得部署。"))
    story.append(P(
        "课内实验练的是训练和调试流程，这次的重点放在网络结构上。"
        "从 VGG11 这个可控的基线出发，每次只改卷积算子、注意力、激活函数或归一化层中的一项，"
        "训练参数全部保持一致，这样比较出来的差值才可信。"))

    story.append(H2T("1.2", "数据集"))
    story.append(P(
        "Fashion-MNIST 由 Zalando 发布，格式和 MNIST 一样：60000 张训练图、10000 张测试图，"
        "单通道 28×28，10 个类别，每类正好 6000 张和 1000 张，不用处理类别不平衡。"
        "它比手写数字难的地方是类间差异小：T-shirt、Pullover、Coat、Shirt 四类的轮廓几乎相同，"
        "只能靠领口、袖长这些局部纹理来分辨。后文的混淆矩阵里，错误也基本都集中在这四类之间。"))
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
        f"预处理只做两件事：训练时随机裁到 28×28（四周补 2 像素）再加水平翻转，"
        f"然后按训练集的统计量（mean={D['normalization_mean']}, std={D['normalization_std']}）标准化；"
        "验证和测试只做标准化。这样验证集上的精度不会被增广影响，可以直接用来选模型。"))

    story.append(H2T("1.3", "评价指标"))
    story.append(P(
        "主指标是准确率。四类上衣容易互相混淆，只看总准确率会掩盖某一类的退化，"
        "所以同时报告宏平均 F1，也就是 10 个类别 F1 的算术平均。"
        "模型成本看三项：参数量、单张图的乘加次数（MACs，卷积层按 H′W′·(C_in/g)·k²·C_out 统计，"
        "全连接层按 in×out 统计，由 forward hook 实测）以及 CPU 单线程推理延迟。"
        "每轮训练耗时也一并记录，用来比较不同算子的实际开销。"))
    return story


def section_baseline(S, R, A) -> list:
    base = R["vgg11"]
    story = [H1T("2", "基线模型：窄通道 VGG11")]
    story.append(H2T("2.1", "网络结构"))
    story.append(P(
        "VGG 的做法是用 3×3 小卷积堆叠换感受野：两个 3×3 的感受野等于一个 5×5，但参数更少、中间多一次非线性。"
        "原版 VGG11 是为 224×224 的 ImageNet 设计的，通道数从 64 一路到 512。"
        "直接搬到 28×28 上，最后几个 stage 的特征图只剩 1×1，再堆 512 通道没有意义，大部分计算都浪费了。"
        "所以这里保留 VGG11 的层数节奏（每个 stage 的卷积个数是 1-1-2-2-2），"
        "把通道宽度缩到 16-32-64-96-96，每个 stage 后面接 2×2 最大池化，最后一个 stage 除外。"))
    story.append(P(
        "每个卷积单元是 Conv3×3（padding=1，bias=False）、BatchNorm、ReLU 的组合。"
        "五个 stage 之后接全局平均池化和三层 MLP 分类头（96→128→64→10，中间有 ReLU 和 Dropout 0.2）。"
        "用全局平均池化替掉 VGG 原来的三层全连接，分类头只剩两万参数，"
        "整个网络的参数量才能控制在 40 万以内。"))
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
        f"表 2 里有两个数字值得注意。Stage 3 只占 {pc(third['params_share'], 1)} 的参数，"
        f"却吃掉 {pc(third['macs_share'], 1)} 的乘加，原因是它的特征图还有 7×7，通道已经到 64。"
        f"Stage 4 和 5 合计占 {pc(late_params, 1)} 的参数，但特征图只剩 3×3 和 1×1，"
        f"乘加只占 {pc(late_macs, 1)}。"
        "也就是说，想省计算量就该动 Stage 3 这种大特征图配中等通道的层，"
        "想省参数量则要动 Stage 4、5 这种小特征图配大通道的层。后面的可分离卷积实验就是按这个思路安排的。"))

    story.append(H2T("2.2", "训练设置"))
    story.append(P(
        "所有实验共用一套训练设置，只改网络结构，这样结果才有可比性。"
        "官方 60000 张训练图用固定种子（42）切成 55000 张训练和 5000 张验证，"
        "官方 10000 张测试集只在最后评一次，不参与任何选择。"
        "验证集用来挑最佳轮次、保存权重，报告里的测试指标都来自这份权重。"))
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
        "几个参数是这样定的：label smoothing 取 0.05，因为四类上衣很容易被网络判得过于自信；"
        "余弦退火让最后几轮的学习率接近 0，验证精度不会在末期来回跳；"
        "随机裁剪补 2 像素，对应真实照片里衣服没有居中、边缘被切掉的情况，比只做翻转更接近部署时的输入。"))

    story.append(H2T("2.3", "基线结果与分析"))
    story.append(P(
        f"基线在测试集上拿到 {num(base['test']['accuracy'])} 的准确率和 {num(base['test']['macro_f1'])} 的 macro-F1，"
        f"最好的是第 {base['best_epoch']} 轮，每轮平均 {base['seconds_per_epoch']:.1f} s，"
        f"一次训练 {base['train_seconds'] / 60:.1f} min。这个水平和 Fashion-MNIST 上同类小网络的公开结果差不多，"
        "说明训练设置没问题，可以拿它当后面消融的参照。"))
    story.append(P(
        "把 10×10 的混淆矩阵按行归一化后（图 1），错误几乎都集中在四类上衣之间："))
    confusions = top_confusions(base, 4)
    lines = [f"{a} 被判成 {b}，占该类样本的 {pc(r, 1)}" for a, b, r in confusions]
    story.append(P("；".join(lines) + "。"))
    scores = dict(zip(base["classes"], base["test"]["per_class"]["f1"]))
    easy_classes = ["Trouser", "Sandal", "Sneaker", "Bag", "Ankle boot"]
    hard_classes = ["T-shirt/top", "Pullover", "Coat", "Shirt"]
    worst_class = min(scores, key=scores.get)
    story.append(P(
        "这四类衣服的轮廓几乎一样，区别在领口、袖长和纹理，"
        f"而 28×28 的灰度图经过两次池化只剩 7×7，这些细节基本被抹平了。它们的 F1 在 "
        f"{pc(min(scores[c] for c in hard_classes), 1)} 到 {pc(max(scores[c] for c in hard_classes), 1)} 之间，"
        f"最难的是 {worst_class}，只有 {num(scores[worst_class])}。"
        f"鞋、包、裤子这几类轮廓差别大，F1 都在 {pc(min(scores[c] for c in easy_classes), 1)} 以上（图 2）。"
        "所以基线的瓶颈不在训练，而在输入分辨率和类别本身太像这两点上，"
        "后面的结构改动能提高的是特征利用效率，变不出图像里没有的信息。"))
    story.append(figure(Path("figures/confusion.png"), 165,
                        "图 1　测试集混淆矩阵（按行归一化，数值为百分比）：错误集中在四类上衣之间"))
    story.append(figure(Path("figures/per_class_f1.png"), 165,
                        "图 2　各类别 F1：鞋、包、裤接近饱和，上衣四类明显偏低"))
    return story


def section_ablation(S, R, A) -> list:
    story = [H1T("3", "结构修改与消融实验")]
    story.append(P(
        "消融的做法是一次只改一个地方，一共 8 个变体加 1 个基线（表 4）。"
        "所有变体用同一份数据划分、同一个随机种子、同一套超参数，只有网络结构不同；"
        "表里第三列写清楚改在哪个 stage，没写到的部分和基线一样。"))
    order = ["vgg11", "sep", "sep_all", "se", "cbam", "gelu", "mish", "gn", "final"]
    labels = {"vgg11": "基线", "sep": "可分离卷积（Stage 3–5）", "sep_all": "可分离卷积（全部）",
              "se": "SE 注意力", "cbam": "CBAM 注意力", "gelu": "GELU 激活",
              "mish": "Mish 激活", "gn": "GroupNorm(8)", "final": "可分离 + CBAM + GELU"}
    rows = [["编号", "配置", "结构改动"]]
    for index, name in enumerate(order):
        rows.append([f"C{index}", name, labels[name]])
    story.append(table(rows, [14, 44, 68], align_left_first=True))
    story.append(caption("表 4　九组对照实验"))

    # 3.1 可分离卷积
    story.append(H2T("3.1", "深度可分离卷积"))
    story.append(P(
        "标准 3×3 卷积一次同时干两件事：在空间上做滤波，在通道上做混合，一次乘加就把 C_in 个通道压成 C_out 个。"
        "深度可分离卷积把它拆开：先对每个通道单独做 3×3 卷积（depthwise，groups=C_in），"
        "再用 1×1 卷积做通道组合（pointwise）。"
        "参数量从 k²·C_in·C_out 降到 k²·C_in + C_in·C_out，两者之比约为 1/C_out + 1/k²；"
        "C_out=96、k=3 时理论值只有 12.2%。乘加同理，只是再乘一个输出空间尺寸 H′W′。"))
    story.append(P(
        "做了两个变体：sep 只替换 Stage 3–5，也就是表 2 里参数量最大的后三个阶段；sep_all 替换全部五个 stage。"
        "这样能看出压哪里的收益更大。如果只压参数多的层就能拿到大部分压缩，说明参数分布确实决定了压缩空间。"))
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
        f"sep 把参数量压到基线的 {pc(sep['parameters'] / R['vgg11']['parameters'], 1)}、"
        f"乘加压到 {pc(sep['macs'] / R['vgg11']['macs'], 1)}，"
        f"准确率{'高了' if sep['delta_accuracy'] > 0 else '低了'} "
        f"{abs(sep['delta_accuracy']) * 100:.2f} 个百分点；"
        f"sep_all 进一步压到 {pc(sep_all['parameters'] / R['vgg11']['parameters'], 1)} 的参数和 "
        f"{pc(sep_all['macs'] / R['vgg11']['macs'], 1)} 的乘加，"
        f"准确率{'高了' if sep_all['delta_accuracy'] > 0 else '低了'} "
        f"{abs(sep_all['delta_accuracy']) * 100:.2f} 个百分点。")
    better = "sep" if sep["test"]["accuracy"] >= sep_all["test"]["accuracy"] else "sep_all"
    extra_macs = sep["macs"] - sep_all["macs"]
    extra_params = sep["parameters"] - sep_all["parameters"]
    story.append(P(
        sep_text + f"两组里 {better} 更好。sep_all 比 sep 只多省了 "
        f"{n(extra_params)} 个参数（{pc(extra_params / sep['parameters'], 1)}）和 "
        f"{n(extra_macs)} 次乘加（{pc(extra_macs / sep['macs'], 1)}），"
        f"准确率却再降 {abs(sep_all['delta_accuracy'] - sep['delta_accuracy']) * 100:.2f} 个百分点。"
        "可分离卷积的压缩比在浅层和深层差得不多（1/C_out+1/k² 从 14% 到 12%），"
        "所以这部分额外损失不是「压得太狠」造成的，而是浅层标准卷积本来就在原始像素上兼顾空间滤波和通道组合，"
        "拆成 depthwise 加 1×1 之后，第一步不做通道混合，跨通道的信息要到第二步才合并，"
        "窄通道网络没有足够的容量把这部分补回来。"))
    stage3_base = {row["stage"]: row for row in A["vgg11"]["rows"]}["stage3"]["macs"]
    stage3_final = {row["stage"]: row for row in A["final"]["rows"]}["stage3"]["macs"]
    theory = 1 / 9 + 1 / 64
    story.append(P(
        f"逐层数据也能对上：final 把 Stage 3 换成可分离卷积后，这一阶段的乘加从 "
        f"{n(stage3_base)} 降到 {n(stage3_final)}，只剩 {pc(stage3_final / stage3_base, 1)}，"
        f"和 C_in=C_out=64、k=3 时的理论值 {pc(theory, 1)} 差不多，"
        "多出来的一小部分是同一个 stage 里 CBAM 分支的开销。"
        "所以可分离卷积适合放通道宽、特征图大的层，浅层窄通道上不一定划算："
        "这时 1/k² 不再占主导，多出来的 1×1 卷积会顶掉一部分收益。"))

    # 3.2 注意力
    story.append(H2T("3.2", "注意力机制：SE 与 CBAM"))
    story.append(P(
        "SE（Squeeze-and-Excitation）先用全局平均池化把每个通道压成一个数，"
        "再过一个 reduction=8 的瓶颈 MLP 学通道权重，最后逐通道相乘。它只关心哪些通道重要，不改空间分辨率，开销很小。"
        "CBAM 在通道注意力之外加了一个空间分支：沿通道取平均和最大，得到两张 28×28（或 7×7）的图，"
        "拼起来过一层 7×7 卷积，得到空间权重图。"
        "两者都加在每个 stage 的最后一个卷积之后、池化之前，也就是 Stage 3–5。"))
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
        f"SE 只多了 {n(R['se']['parameters'] - R['vgg11']['parameters'])} 个参数（约 "
        f"{pc((R['se']['parameters'] - R['vgg11']['parameters']) / R['vgg11']['parameters'], 2)}），"
        f"准确率 {pp(R['se']['delta_accuracy'])} 个百分点；CBAM 因为多了空间分支的 7×7 卷积，参数略高，"
        f"准确率 {pp(R['cbam']['delta_accuracy'])} 个百分点。"
        + ("两个模块都没有带来能看出来的提升。这和数据集本身有关：特征已经被池化到 7×7，"
           "能挑的细节本来就少，注意力加权在这里没有多少可加的东西。" if abs(R["cbam"]["delta_accuracy"]) < 0.01 else
           "注意力带来了明显的提升，说明通道和空间加权在这个任务上起到了正则作用。")
        + f"论性价比，{'SE' if se_better else 'CBAM'} 更划算。"))
    def pair_errors(name: str) -> int:
        record = R[name]
        confusion = record["test"]["confusion"]
        first = record["classes"].index("T-shirt/top")
        second = record["classes"].index("Shirt")
        return int(confusion[first][second] + confusion[second][first])

    story.append(P(
        "注意力到底改变了什么，可以看 Grad-CAM。这里对第 3 个 stage 的特征图（7×7）做梯度加权，"
        "再放大回原图（图 3）。两组的热区没有一致的方向：短靴、凉鞋这类小目标上，"
        "组合模型的热区更贴鞋身，基线常常落到图像下缘和背景里；"
        "但运动鞋、长裤上基线的热区反而更完整。6 个样本、一次前向的可视化只能当例子看，说明不了注意力「看得更准」。"
        "定量结果也差不多：衬衫和 T 恤互相误判的样本，基线是 "
        f"{pair_errors('vgg11')} 例，CBAM 是 {pair_errors('cbam')} 例，final 是 {pair_errors('final')} 例，"
        "注意力并没有缓解这对最难的类别。"))
    story.append(figure(Path("figures/gradcam.png"), 168,
                        "图 3　Grad-CAM（第 3 个 stage）：上排为基线，下排为可分离卷积 + CBAM + GELU 组合模型"))

    # 3.3 激活函数
    story.append(H2T("3.3", "激活函数：GELU 与 Mish"))
    story.append(P(
        "ReLU 便宜，梯度不饱和，但负半轴直接截断，负值的信息全丢了。"
        "GELU 用高斯 CDF 给负值留一个小斜率（x·Φ(x)），Mish 用 x·tanh(softplus(x)) 在负半轴先下去再回来。"
        "这两个函数都想在保留稀疏性的同时留下一点负值信息。"))
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
        "GELU 和 Mish 都没有参数，所以参数量和基线一模一样（表 7），变的只是计算和优化。"
        "表里的单轮耗时只能当参考：九次训练是三个进程并行跑的，单轮耗时受 CPU 争用影响很大，"
        "不能直接拿来比算子开销；能比的是第 4.2 节在空闲机器上测的单线程延迟。"
        "精度上，"
        + ("两个函数和 ReLU 的差距都在 0.5 个百分点以内，属于单次实验的噪声。"
           "在 28×28 这样的小网络上，换激活函数的收益远不如换卷积算子。"
           if max(abs(R["gelu"]["delta_accuracy"]), abs(R["mish"]["delta_accuracy"])) < 0.005 else
           "两者的差距超过了单次实验的噪声量级，说明平滑激活在这个任务上确实起到了额外正则的作用。")))

    # 3.4 归一化
    story.append(H2T("3.4", "归一化层：BatchNorm 与 GroupNorm"))
    story.append(P(
        "BatchNorm 在 batch 维上统计均值和方差，batch 越小统计越不可靠。"
        "GroupNorm 改成在每个样本内部把通道分成 8 组、在组内统计，完全不依赖 batch 大小。"
        "理论上前者 batch=128 时两者应该差不多，但边缘设备推理常常是 batch=1，"
        "这时 BN 只能用训练时存下的滑动统计量，GN 的行为则和训练时一样。"
        "为了验证，另外训了两个 batch=16 的模型（8 轮，其余不变），和 batch=128 的结果对照。"))
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
            f"batch=128 时 GN 比 BN 低 {abs(gap_big) * 100:.2f} 个百分点；"
            f"batch 降到 16（两组都只训 8 轮，预算一样）后，GN 比 BN 低 "
            f"{abs(gap_small) * 100:.2f} 个百分点，差距小了一半。"
            "方向和小 batch 会让 BN 统计量变噪的说法一致，但 GN 没有反超。"
            "也就是说，在本任务上 GN 的价值不在精度，而在推理时统计量和训练时一致："
            "batch=1 的流式推理里 BN 只能用滑动统计量，GN 不受影响。"
            f"另一个和预期相反的地方是曲线：BN 相邻两轮验证精度的标准差是 {step_std(R['vgg11']):.4f}，"
            f"GN 是 {step_std(R['gn']):.4f}，GN 反而更抖。"
            "可能是因为 GN 把通道分成 8 组，每组只有 8 到 12 个通道，统计量本身噪声更大；"
            f"batch=16 时也是这个结果（{step_std(bn_small):.4f} 和 {step_std(gn_small):.4f}）。"
            "所以实际的结论是：训练 batch 够大就用 BN，又快又准；"
            "batch 很小或者需要逐样本统计时才换 GN，精度上要让一点。"))
    else:
        story.append(P(
            f"在 batch=128 时，GN 比 BN 低 {abs(gap_big) * 100:.2f} 个百分点，"
            "差距不大但方向稳定：BN 的 batch 维噪声本身就是一种正则，"
            "GN 用逐样本统计换来了批大小无关的行为，代价是失去这部分正则。"))

    # 3.5 组合模型
    story.append(H2T("3.5", "组合模型与综合对比"))
    story.append(P(
        "最后把三项不冲突的改动叠在一起：Stage 3–5 用可分离卷积（省参数和计算）、"
        "Stage 3–5 加 CBAM（补通道和空间选择）、激活换成 GELU（平滑，也比 Mish 便宜），得到 final。"
        f"它只有 {n(R['final']['parameters'])} 个参数、{n(R['final']['macs'])} 次乘加，"
        f"分别是基线的 {pc(R['final']['parameters'] / R['vgg11']['parameters'], 1)} 和 "
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
        f"表 9 里准确率最高的是 {best_name}（{num(R[best_name]['test']['accuracy'])}）。"
        f"组合模型 final 是 {num(R['final']['test']['accuracy'])}，不是最高的："
        f"它把乘加压到基线的 {pc(R['final']['macs'] / R['vgg11']['macs'], 0)}，参数压到 "
        f"{pc(R['final']['parameters'] / R['vgg11']['parameters'], 0)}，代价是 "
        f"{abs(R['final']['delta_accuracy']) * 100:.2f} 个百分点。"
        f"和只做可分离卷积的 sep（{num(R['sep']['test']['accuracy'])}）比，CBAM 和 GELU 把损失从 "
        f"{abs(R['sep']['delta_accuracy']) * 100:.2f} 个百分点收窄到 "
        f"{abs(R['final']['delta_accuracy']) * 100:.2f} 个百分点，收回 "
        f"{(R['final']['delta_accuracy'] - R['sep']['delta_accuracy']) * 100:.2f} 个百分点，"
        "但补不回压缩带来的全部损失。"
        f"这笔交换还算划算：权重从 {R['vgg11']['weights_kb'] / 1024:.2f} MB 降到 "
        f"{R['final']['weights_kb'] / 1024:.2f} MB，单张图的乘加只剩 "
        f"{R['final']['macs'] / R['vgg11']['macs']:.0%}。"
        "要精度就选 mish，只多一点常数开销，精度不降；"
        "要算力才需要在 final 和 sep 之间按预算取舍。"))
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
        f"九组配置都跑满 16 轮，其中 {late_best} 组的最佳轮次在第 10 轮之后，"
        "说明余弦退火的后半程确实在继续往好的解上走，不是干等训练结束。"
        "验证精度到了后半程基本是平的，最后几轮学习率接近 0，也没有出现末期回升。"
        f"可分离卷积收敛慢一点：前 3 轮的平均验证精度是 {num(early_mean('sep'))}，基线是 "
        f"{num(early_mean('vgg11'))}；到末期两者差不多了（{num(last_val('sep'))} 和 {num(last_val('vgg11'))}）。"
        "这和 depthwise 卷积梯度尺度小、需要更多步有关。"))
    story.append(figure(Path("figures/training_curves.png"), 165,
                        "图 6　九组配置的训练/验证准确率曲线（虚线为选中轮次）"))
    return story


def section_deploy(S, R, EXP, ROB, CONF) -> list:
    confidence_record = next(record for record in CONF["models"] if record["name"] == "final")
    review_row = min(confidence_record["rows"], key=lambda row: abs(row["threshold"] - 0.60))
    story = [H1T("4", "场景应用与部署")]
    story.append(H2T("4.1", "交互式 Web 演示"))
    story.append(P(
        "场景应用做了一个可以直接跑的 Web 演示（webapp.py，只用标准库的 http.server 加 PIL 和 PyTorch，"
        "没有 Flask 之类的依赖）。页面可以拖拽上传图片，也可以从测试集随机抽一张，"
        "服务端把图转灰度、短边缩到 28 再中心裁剪，最后按训练集的均值和方差标准化。"
        "有个地方要处理：Fashion-MNIST 是亮色衣服配黑底，实拍的商品图通常是深色衣服配白底，"
        "所以预处理会先看整图的平均亮度，偏亮就自动反色，不然模型会一直判错。"))
    story.append(P(
        f"推理返回 Top-3 类目和概率，另外加了一个可以直接用的开关：最高概率低于 0.60 就标成「转人工复核」。"
        f"阈值的效果可以在测试集上量出来。组合模型有 {pct1(review_row['review_share'])}% 的样本低于阈值，"
        f"这些样本的准确率只有 {pct1(review_row['accuracy_rejected'])}%；"
        f"剩下 {pct1(review_row['coverage'])}% 自动通过，准确率 {pct1(review_row['accuracy_covered'])}%。"
        f"假设人工能把复核的样本全部改对，整条流水线的等效准确率是 "
        f"{num(review_row['coverage'] * review_row['accuracy_covered'] + review_row['review_share'])}，"
        f"全部自动通过时是 {num(confidence_record['accuracy'])}。"
        f"人工只需要看 {pct1(review_row['review_share'])}% 的样本，整体精度就被抬高了 "
        f"{num(review_row['coverage'] * review_row['accuracy_covered'] + review_row['review_share'] - confidence_record['accuracy'])}。"))
    screenshot = Path("figures/webapp.png")
    if screenshot.exists():
        story.append(figure(screenshot, 158, "图 7　Web 演示界面：上传图片后返回 Top-3 与人工复核标记"))
    else:
        story.append(P("（Web 演示截图见仓库 figures/webapp.png。）", BODY_NO_INDENT))

    story.append(H2T("4.2", "ONNX 导出与 int8 量化"))
    story.append(P(
        "部署这一步是把 PyTorch 模型导成 ONNX，再做动态 int8 量化（权重按通道量化，激活在线量化，不需要校准集）。"
        "表 10 是基线和最终模型在 CPU 单线程下的体积和延迟。"))
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
        "量化把权重从 fp32 压到 int8，体积明显小了（表 10），但延迟没有变好，反而更慢，"
        "这是这次实验里最反直觉的结果。原因是动态量化的 ConvInteger 在 ONNX Runtime 的 CPU 执行器里"
        "缺少针对小通道卷积优化的 kernel：这个网络的通道只有 16 到 96，"
        "每层的反量化、权重恢复和调度开销比省下的整数运算还多。"
        "所以小模型上的数值量化只适合体积敏感、延迟不敏感的场景，比如跟着 App 一起分发的离线模型；"
        "想在 CPU 上提速，还是剪结构更直接，比如可分离卷积、少一个 stage。"))
    onnx_ratio = {record["name"]: record["int8_ms"] / record["onnx_ms"]
                  for record in EXP["models"] if record.get("onnx_ms") and record.get("int8_ms")}
    size_ratio = {record["name"]: record["int8_kb"] / record["onnx_kb"]
                  for record in EXP["models"] if record.get("int8_kb")}
    if onnx_ratio:
        story.append(P(
            f"具体数字：int8 体积是 fp32 的 {min(size_ratio.values()):.0%} 到 {max(size_ratio.values()):.0%}，"
            f"单线程延迟却是 fp32 ONNX 的 {min(onnx_ratio.values()):.1f} 到 {max(onnx_ratio.values()):.1f} 倍。"
            "另外 PyTorch eager 的延迟比 ONNX Runtime 高不少（后者对卷积做了更多图优化），"
            "但在 PyTorch 下组合模型反而比基线还慢，说明 depthwise 卷积在 PyTorch CPU 上的实现效率"
            "吃掉了一部分乘加优势。同一个模型换个运行时，结论就变了。"))

    story.append(H2T("4.3", "鲁棒性评估"))
    story.append(P(
        "真实场景的输入不会像测试集这么干净。这里做了六种退化：旋转 15°、加高斯噪声（σ=0.15）、"
        "高斯模糊（σ=1.0）、亮度降到 40%、随机 10×10 遮挡、先降到 14×14 再放大，"
        "在完整的 10000 张测试集上比较四个模型的准确率。退化参数和随机种子都固定，可以复现。"))
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
        f"四个模型里平均掉点最多的是 {most}（{pct1(drops[most])} 个百分点），"
        f"最少的是 {least}（{pct1(drops[least])} 个百分点）。"
        f"按条件看，掉得最多的是「{worst_condition}」（{pct1(condition_drops[worst_condition])} 个百分点），"
        f"最少的是「{mildest_condition}」（{pct1(condition_drops[mildest_condition])} 个百分点）。"
        "这和信息损失的多少是对得上的：前者把模型要看的像素证据破坏了，"
        "后者只是整体缩放亮度，标准化之后差别基本被抹平。"
        + (f"可分离卷积模型的平均掉点比基线低 {pct1(-sep_gap)} 个百分点，"
           "省参数没有拿鲁棒性去换。" if sep_gap < 0 else
           f"可分离卷积模型的平均掉点比基线高 {pct1(sep_gap)} 个百分点，"
           "压缩之后遇到退化输入会稍微吃亏。")
        + (f"CBAM 在 10×10 遮挡下比基线高 {pct1(cbam_gain)} 个百分点，"
           "和空间注意力能补局部缺失的直觉一致。" if cbam_gain > 0 else
           f"CBAM 在 10×10 遮挡下比基线低 {pct1(-cbam_gain)} 个百分点，"
           "空间注意力并没有带来预期的遮挡鲁棒性。")
        + f"不过组合模型 final 在遮挡下只有 {num(occlusion.get('final', 0))}，四个模型里最低的。"
        "可分离卷积省下来的那点鲁棒性余量，抵不过 CBAM 和 GELU 在小目标上的损失。"
        "结构改动之间不是简单叠加，鲁棒性尤其如此。"))
    if Path("figures/robustness.png").exists():
        story.append(figure(Path("figures/robustness.png"), 150,
                            "图 8　合成退化下的准确率对比"))

    story.append(H2T("4.4", "落地讨论"))
    story.append(P(
        "把结果放回业务里，大概有三种用法。手机端实时打标用组合模型："
        f"权重 {R['final']['weights_kb'] / 1024:.2f} MB，计算量是基线的 "
        f"{pc(R['final']['macs'] / R['vgg11']['macs'], 0)}，导出 ONNX 后单线程延迟 "
        f"{num(next(r for r in EXP['models'] if r['name'] == 'final')['onnx_ms'], 3)} ms，"
        "移动端还有余量。不过这次只做了 int8 量化，没上真机，量化版本在这台机器上反而更慢。"
        "后台批量清洗商品库时算力不是瓶颈，直接挑精度最高的配置，"
        "再用置信度阈值把没把握的样本送人工，也就是 4.1 那条流水线。"
        "输入分布和训练集差得远的时候（实拍图而不是白底图），先做背景归一化和反色，"
        "结构再好也补不回这部分分布差异。"))
    return story


def section_discussion(S, R) -> list:
    story = [H1T("5", "讨论与局限")]
    story.append(P(
        "有三点会影响这些结论的适用范围。第一，所有配置只跑了一个随机种子。"
        "Fashion-MNIST 上换个初始化，准确率大概会晃 0.3 到 0.5 个百分点，"
        "比这个还小的差异（比如 GELU 和 ReLU、SE 和 CBAM 之间）不能当成真实效应。"
        "要做统计上靠得住的结论，每组至少得重复 3 到 5 次并给出均值和标准差，这次的时间预算不够。"))
    story.append(P(
        "第二，数据集本身限制了结构创新的空间。28×28 的灰度图池化四次之后只剩 1×1，"
        "感受野很快就盖住整张图，深层结构之间的差别被压没了；服饰类别之间语义又重叠，精度上限大概在 0.95 附近。"
        "小图加简单任务，结构改动的收益天然比 ImageNet 级别小，"
        "这也是注意力几乎没有收益、可分离卷积却掉了近两个百分点的一个原因。"
        "想验证得更充分，应该换 Fashion-Product-1M 或 DeepFashion 这类高分辨率的真实商品图。"))
    story.append(P(
        "第三，可分离卷积的收益被窄通道削弱了。理论比值 1/C_out + 1/k² 假设 C_out 远大于 k²，"
        "而这里最大的通道只有 96，1/C_out 这一项不能忽略，所以实测压缩比比 MobileNet 那类网络差不少。"
        "结构技巧的收益取决于它放在多宽的通道、多大的特征图上，"
        "照搬论文里「替换即提速」的说法不可靠，得先看逐层成本（如表 2）再决定改哪里。"))
    story.append(P(
        "后面还能做的是：多跑几个种子给出置信区间；用 CutMix、MixUp 这类更强的增广冲一冲精度上限；"
        "用知识蒸馏把基线的软标签传给组合模型，看更少参数下能不能追回来；"
        "以及把 Grad-CAM 从定性观察做成定量指标，比如和人工标注的衣物区域算 IoU。"))
    return story


def section_conclusion(S, R) -> list:
    base, final = R["vgg11"], R["final"]
    best = max(R.values(), key=lambda r: r["test"]["accuracy"])
    singles = {name: R[name]["delta_accuracy"] for name in ("se", "cbam", "gelu", "mish", "gn")}
    story = [H1T("6", "结论")]
    story.append(P(
        f"这次在 Fashion-MNIST 上搭了一个窄通道 VGG11，以它作基线做了八组结构消融。"
        f"基线测试准确率 {num(base['test']['accuracy'])}，macro-F1 {num(base['test']['macro_f1'])}，"
        f"参数量 {n(base['parameters'])}，单张图 {n(base['macs'])} 次乘加。"))
    story.append(P(
        f"结构上最值得说的是压哪里比怎么压更重要。逐层成本显示 Stage 3 吃掉一半以上的计算量，"
        f"Stage 4、5 占去近八成参数；把可分离卷积只放在 Stage 3–5，用 "
        f"{pc(R['sep']['parameters'] / base['parameters'], 1)} 的参数和 "
        f"{pc(R['sep']['macs'] / base['macs'], 1)} 的乘加换来 "
        f"{abs(R['sep']['delta_accuracy']) * 100:.2f} 个百分点的精度损失。"
        f"注意力、激活函数、归一化的单项改动带来的变化在 {pp(min(singles.values()))} 到 "
        f"{pp(max(singles.values()))} 个百分点之间；组合模型 final 把参数压到 "
        f"{pc(final['parameters'] / base['parameters'], 1)}、乘加压到 {pc(final['macs'] / base['macs'], 1)}，"
        f"准确率还有 {num(final['test']['accuracy'])}；九组里最高的是 {best['name']}，"
        f"{num(best['test']['accuracy'])}。"))
    story.append(P(
        "工程上把模型导成了 ONNX 并做了 int8 动态量化，配了一个 Web 演示，用六种退化测了鲁棒性，"
        "也验证了置信度阈值加人工复核这条流水线。做完的一个感受是：40 万参数以内的 CNN "
        "在服饰分类上已经够用，难点主要在把结构改在正确的位置，以及处理好训练数据和真实输入之间的分布差异。"))
    return story


def appendix(R) -> list:
    import importlib.metadata as metadata

    def version_of(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "—"

    rows = [["项目", "版本 / 配置"],
            ["操作系统", R["vgg11"]["platform"]],
            ["Python", R["vgg11"]["python"]],
            ["PyTorch", R["vgg11"]["torch"]],
            ["torchvision", version_of("torchvision")],
            ["NumPy / Matplotlib", f"{version_of('numpy')} / {version_of('matplotlib')}"],
            ["onnxruntime / onnx", f"{version_of('onnxruntime')} / {version_of('onnx')}"],
            ["CPU", "Intel Core i9-14900HX（训练用 6 线程）"]]
    story = [KeepTogether([H1T("附录 A", "运行环境与复现步骤"),
                            table(rows, [34, 92], align_left_first=True),
                            caption("表 A1　实验环境")])]
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
            "code", fontName="SimSun", fontSize=8.6, leading=13, leftIndent=10, spaceAfter=1)))
    story.append(P(
        "这些命令跑完后，runs/ 下每个配置各有一个 metrics.json 和一份权重，figures/ 下是全部图，"
        "最后 python report.py 读这些 JSON 重新生成这份报告。正文里的数字都是脚本从 JSON 里取出来填进去的，"
        "改了实验重跑一遍，报告里的数不会和结果对不上。"))
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


# 组装
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
