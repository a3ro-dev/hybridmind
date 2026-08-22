"""Render the frozen HybridMind retrieval research note as a polished PDF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "hybridmind-retrieval-research-20260822.pdf"
LEDGER = ROOT / "experiments" / "results" / "claim-ledger-20260822.json"
EXPECTED_LEDGER_SHA256 = "6a12c941f10a538fdb5fd1d35a76385e9ea8a3a177d370d6abda94efc634cca3"

INK = colors.HexColor("#171A1F")
MUTED = colors.HexColor("#5E6875")
BLUE = colors.HexColor("#315CF4")
TEAL = colors.HexColor("#008F7A")
RED = colors.HexColor("#C94848")
AMBER = colors.HexColor("#B36A00")
PAPER = colors.HexColor("#FBFCFE")
PALE = colors.HexColor("#F1F4F8")
LINE = colors.HexColor("#D9E0E8")
WHITE = colors.white


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_fonts() -> None:
    font_root = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("Segoe", str(font_root / "segoeui.ttf")))
    pdfmetrics.registerFont(TTFont("Segoe-Bold", str(font_root / "segoeuib.ttf")))
    pdfmetrics.registerFont(TTFont("Segoe-Italic", str(font_root / "segoeuii.ttf")))
    addMapping("Segoe", 0, 0, "Segoe")
    addMapping("Segoe", 1, 0, "Segoe-Bold")
    addMapping("Segoe", 0, 1, "Segoe-Italic")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Segoe-Bold", fontSize=29,
            leading=33, textColor=INK, alignment=TA_LEFT, spaceAfter=9,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Segoe", fontSize=11.2,
            leading=16, textColor=MUTED, spaceAfter=14,
        ),
        "eyebrow": ParagraphStyle(
            "eyebrow", parent=base["Normal"], fontName="Segoe-Bold", fontSize=8.2,
            leading=10, textColor=BLUE, tracking=1.4, spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Segoe-Bold", fontSize=19,
            leading=23, textColor=INK, spaceBefore=4, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Segoe-Bold", fontSize=12.5,
            leading=15.5, textColor=INK, spaceBefore=8, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Segoe", fontSize=9.25,
            leading=13.3, textColor=INK, spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontName="Segoe", fontSize=7.5,
            leading=10.2, textColor=MUTED,
        ),
        "table": ParagraphStyle(
            "table", parent=base["BodyText"], fontName="Segoe", fontSize=7.3,
            leading=9.4, textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "table_head", parent=base["BodyText"], fontName="Segoe-Bold", fontSize=7.2,
            leading=9, textColor=WHITE,
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["BodyText"], fontName="Segoe-Bold", fontSize=13,
            leading=18, textColor=INK, alignment=TA_LEFT,
        ),
        "metric": ParagraphStyle(
            "metric", parent=base["BodyText"], fontName="Segoe-Bold", fontSize=19,
            leading=21, textColor=BLUE, alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "metric_label", parent=base["BodyText"], fontName="Segoe", fontSize=7.2,
            leading=9, textColor=MUTED, alignment=TA_CENTER,
        ),
        "reference": ParagraphStyle(
            "reference", parent=base["BodyText"], fontName="Segoe", fontSize=7.1,
            leading=9.5, textColor=MUTED, leftIndent=8, firstLineIndent=-8, spaceAfter=3,
        ),
    }


class ResearchTree(Flowable):
    def __init__(self, width: float = 500, height: float = 292) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def _box(self, canvas: Any, x: float, y: float, w: float, h: float, text: str,
             fill: colors.Color, stroke: colors.Color = LINE, font_size: float = 8.5) -> None:
        canvas.setFillColor(fill)
        canvas.setStrokeColor(stroke)
        canvas.roundRect(x, y, w, h, 5, fill=1, stroke=1)
        canvas.setFillColor(INK)
        canvas.setFont("Segoe-Bold", font_size)
        canvas.drawCentredString(x + w / 2, y + h / 2 - font_size / 3, text)

    def draw(self) -> None:
        c = self.canv
        w = self.width
        c.setStrokeColor(colors.HexColor("#98A5B5"))
        c.setLineWidth(1.1)
        top_y = 255
        self._box(c, w / 2 - 64, top_y, 128, 28, "HYBRIDMIND", colors.HexColor("#E8EDFF"), BLUE, 10)
        branch_y = 208
        c.line(w / 2, top_y, w / 2, branch_y + 28)
        c.line(w / 2, branch_y + 28, 128, branch_y + 28)
        c.line(w / 2, branch_y + 28, w - 128, branch_y + 28)
        c.line(128, branch_y + 28, 128, branch_y + 24)
        c.line(w - 128, branch_y + 28, w - 128, branch_y + 24)
        self._box(c, 72, branch_y, 112, 24, "RETRIEVAL", PALE, font_size=8.8)
        self._box(c, w - 184, branch_y, 112, 24, "MEMORY MODEL", PALE, font_size=8.8)

        node_y = 150
        left_x = [28, 99, 170]
        right_x = [w - 223, w - 152, w - 81]
        for center in (128, w - 128):
            c.line(center, branch_y, center, node_y + 34)
            c.line(center - 71, node_y + 34, center + 71, node_y + 34)
        for x, label in zip(left_x, ("dense", "sparse", "graph")):
            c.line(x + 29, node_y + 34, x + 29, node_y + 25)
            self._box(c, x, node_y, 58, 25, label, WHITE, font_size=7.8)
        for x, label in zip(right_x, ("episodic", "semantic", "temporal")):
            c.line(x + 29, node_y + 34, x + 29, node_y + 25)
            self._box(c, x, node_y, 58, 25, label, WHITE, font_size=7.2)

        fuse_y = 104
        for x in [*left_x, *right_x]:
            c.line(x + 29, node_y, x + 29, fuse_y + 32)
        c.line(left_x[0] + 29, fuse_y + 32, right_x[-1] + 29, fuse_y + 32)
        c.line(w / 2, fuse_y + 32, w / 2, fuse_y + 24)
        self._box(c, w / 2 - 70, fuse_y, 140, 24, "candidate fusion", colors.HexColor("#E8F7F3"), TEAL)

        labels = (("reranking / selection", 65), ("evidence grounding", 30), ("answer generation", -5))
        previous_y = fuse_y
        for label, y in labels:
            c.line(w / 2, previous_y, w / 2, y + 23)
            self._box(c, w / 2 - 70, y, 140, 23, label, WHITE, font_size=7.8)
            previous_y = y


class CeilingBars(Flowable):
    def __init__(self, width: float = 500, height: float = 178) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def draw_panel(self, c: Any, x: float, y: float, width: float, title: str,
                   values: list[tuple[str, float, colors.Color]]) -> None:
        c.setFont("Segoe-Bold", 8.5)
        c.setFillColor(INK)
        c.drawString(x, y + 106, title)
        bar_x = x + 72
        bar_w = width - 98
        for index, (label, value, color) in enumerate(values):
            row_y = y + 75 - index * 31
            c.setFont("Segoe", 7.5)
            c.setFillColor(MUTED)
            c.drawRightString(bar_x - 8, row_y + 5, label)
            c.setFillColor(PALE)
            c.roundRect(bar_x, row_y, bar_w, 13, 4, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(bar_x, row_y, bar_w * value, 13, 4, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Segoe-Bold", 7.3)
            c.drawString(bar_x + bar_w * value + 5, row_y + 3, f"{value:.3f}")

    def draw(self) -> None:
        c = self.canv
        half = self.width / 2 - 8
        self.draw_panel(c, 0, 52, half, "MiniLM split - speaker-prefix BM25S",
                        [("pre", 0.588057, MUTED), ("post", 0.630680, BLUE), ("pool oracle", 0.673342, TEAL)])
        self.draw_panel(c, half + 16, 52, half, "BGE-M3 split - speaker-prefix BM25S",
                        [("pre", 0.567986, MUTED), ("MaxSim", 0.597208, BLUE), ("pool oracle", 0.670371, TEAL)])
        c.setFillColor(MUTED)
        c.setFont("Segoe-Italic", 7)
        c.drawString(0, 20, "Separate conversation splits: read each panel as its own fixed-pool ceiling, not as a model leaderboard.")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def metric_card(value: str, label: str, s: dict[str, ParagraphStyle]) -> Table:
    card = Table([[p(value, s["metric"])], [p(label, s["metric_label"])]], colWidths=[42 * mm])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
    ]))
    return card


def result_table(rows: list[list[str]], s: dict[str, ParagraphStyle], widths: list[float]) -> LongTable:
    data = [[p(cell, s["table_head"]) for cell in rows[0]]]
    data.extend([p(cell, s["table"]) for cell in row] for row in rows[1:])
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F7F9FB")))
    table.setStyle(TableStyle(commands))
    return table


def page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.line(doc.leftMargin, 17 * mm, A4[0] - doc.rightMargin, 17 * mm)
    canvas.setFont("Segoe", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 11 * mm, "hybridmind retrieval research - exact evidence, not architecture loyalty")
    canvas.drawRightString(A4[0] - doc.rightMargin, 11 * mm, str(doc.page))
    canvas.restoreState()


def build() -> None:
    if sha256(LEDGER) != EXPECTED_LEDGER_SHA256:
        raise RuntimeError("claim ledger changed after the report was frozen")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if sum((row.get("provider_calls") or 0) for row in ledger["claims"]) != 0:
        raise RuntimeError("report requires zero provider calls in every retained claim")
    register_fonts()
    s = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=19 * mm, bottomMargin=23 * mm,
        title="hybridmind, after contact with evidence",
        author="Akshat / HybridMind research program",
        subject="Evidence-driven long-term memory retrieval mechanism study",
    )
    story: list[Any] = []

    story += [
        Spacer(1, 15 * mm),
        p("RESEARCH NOTE  /  22 AUG 2026", s["eyebrow"]),
        p("hybridmind, after contact with evidence", s["title"]),
        p("an architecture-neutral long-term memory retrieval program, with exact evidence IDs, held-out controls, resource accounting, and zero provider calls", s["subtitle"]),
        Spacer(1, 5 * mm),
    ]
    callout = Table([[p("no SOTA claim. a quantified offline ceiling - and a simpler architecture that earned its place.", s["callout"])]], colWidths=[172 * mm])
    callout.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8EDFF")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B7C4FF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 13),
        ("RIGHTPADDING", (0, 0), (-1, -1), 13),
        ("TOPPADDING", (0, 0), (-1, -1), 13),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
    ]))
    story += [callout, Spacer(1, 8 * mm)]
    cards = Table([[metric_card("39", "Tier-S + Tier-A systems investigated", s),
                    metric_card("1,977", "exact-evidence baseline questions", s),
                    metric_card("0", "provider calls in retained artifacts", s),
                    metric_card("387 + 16", "Python + legacy verification passes", s)]],
                  colWidths=[43 * mm] * 4, hAlign="LEFT")
    cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    story += [cards, Spacer(1, 9 * mm)]
    story += [
        p("the answer first", s["h1"]),
        p("i treated every component as disposable. the conventional candidate generator survived: speaker-prefixed BM25S. MiniLM and BGE-M3 MaxSim both improved selection inside fixed pools, but neither passed an unconditional production gate. learned sparse, general graph promotion, a speaker router, two-field sparse RRF, and low-bit Turbovec defaults did not survive.", s["body"]),
        p("the honest result is narrower than the original ambition and much more useful: candidate generation remains the dominant gap, selectors recover only part of the evidence already present, and every stronger claim still needs an independent corpus or the priced native-4096 semantic run.", s["body"]),
        Spacer(1, 2 * mm),
        p("architecture rule", s["h2"]),
        p("a component earns its place only by improving the measured objective at a defensible resource point. novelty has no positive weight.", s["body"]),
        PageBreak(),
        p("the research tree", s["h1"]),
        p("retrieval mechanisms and memory representations are independent factors. they meet only at candidate fusion; selection cannot repair missing evidence, and generation cannot be called grounded without source IDs.", s["body"]),
        ResearchTree(172 * mm, 102 * mm),
        Spacer(1, 2 * mm),
        p("the design space", s["h2"]),
    ]
    design_rows = [
        ["layer", "mechanisms investigated", "causal question"],
        ["retrieval / dense", "FAISS, ScaNN, USearch, DiskANN, Turbovec, Matryoshka", "quality and feasible memory/latency at native 4096-d"],
        ["retrieval / sparse", "BM25S, Lucene/Tantivy, SPLADE++, BGE-M3 sparse", "source representation or learned expansion?"],
        ["retrieval / graph", "typed traversal, PPR, HippoRAG, Graphiti/Zep", "real association beyond degree, anchors, and lexical overlap?"],
        ["memory model", "episodic raw records, facts/summaries, intervals/supersession", "does derived memory improve access without losing provenance?"],
        ["fusion / selection", "RRF, MiniLM, BGE rerankers, MaxSim, RankLLM, FlashRank", "candidate gain or ordering gain, at what cost?"],
        ["grounding / reader", "exact evidence, oracle context, answer generation", "does retrieval become supported answer accuracy?"],
    ]
    story += [result_table(design_rows, s, [29 * mm, 75 * mm, 68 * mm]), PageBreak()]

    story += [p("what survived", s["h1"])]
    survived = [
        ["mechanism", "held-out or mechanics evidence", "decision"],
        ["speaker-prefix BM25S", "+0.02966 mean Recall@10 over raw across five reused splits", "keep as conventional candidate baseline"],
        ["MiniLM fixed-pool", "+0.04262 Recall@10; CI [0.01299, 0.06643]", "keep gated; multi-hop regressed"],
        ["BGE-M3 MaxSim", "+0.02922 Recall@10 on BM25 pool; CI [0.01419, 0.04085]", "retain experimental; storage/encoding block default"],
        ["PPR vs degree sham", "+0.56447 Recall@10; CI [0.55794, 0.57030]", "association is real, general graph win is not"],
        ["FAISS HNSW", "synthetic Recall@10 0.66523 at efSearch 64; 1.0 at 1024", "expose and attest search effort"],
        ["compact SQLite vectors", "48.50% database reduction; bit-exact logical equivalence", "keep"],
    ]
    story += [result_table(survived, s, [42 * mm, 81 * mm, 49 * mm]), Spacer(1, 7 * mm), p("what was eliminated", s["h1"])]
    rejected = [
        ["mechanism", "falsifying evidence", "decision"],
        ["BGE learned sparse", "-0.09070 pre-rerank Recall@10 vs BM25S; CI below zero", "reject this configuration on LoCoMo"],
        ["general PPR default", "+0.00859 vs BM25S; CI crosses zero; multi-hop delta 0", "reject; temporal follow-up open"],
        ["unconditional MiniLM", "multi-hop -0.09127; CI [-0.13158, -0.03101]", "reject; gate or retrain"],
        ["speaker router", "same recall as prefix while retaining both indexes", "reject"],
        ["two-field sparse RRF", "0.56056 vs 0.57136 for one prefix index", "reject on measured split"],
        ["Turbovec 4-bit", "0.85344 synthetic Recall@10 at about 5.2x compression", "reject default"],
        ["local LongMemEval score", "948/948 haystack sessions gold; zero distractors", "invalidate and fail admission"],
    ]
    story += [result_table(rejected, s, [42 * mm, 81 * mm, 49 * mm]), PageBreak()]

    story += [
        p("the ceiling we can defend", s["h1"]),
        p("the two selector experiments used different conversation splits. each panel is therefore a within-split candidate ceiling, not a cross-model leaderboard.", s["body"]),
        CeilingBars(172 * mm, 63 * mm),
        p("MiniLM closes 0.04262 absolute Recall@10 inside its pool and leaves a 0.04266 selector-oracle gap. MaxSim closes 0.02922 inside its BM25 pool and leaves 0.07316. the main unsolved problem is still evidence that never enters the pool.", s["body"]),
        Spacer(1, 3 * mm),
        p("resource truth", s["h1"]),
    ]
    resource_rows = [
        ["component", "measured resource", "interpretation"],
        ["BGE-M3 token vectors", "837,558,272 bytes for 5,881 turns; 13.38 min CPU encoding", "positive MaxSim lift does not erase storage/build cost"],
        ["BGE learned sparse", "163,804 postings; equal top-25 comparison", "more machinery, lower candidate ceiling than BM25S"],
        ["MiniLM", "287.06 ms mean per 25-document pool on CPU", "aggregate lift, material latency, multi-hop failure"],
        ["native 4096-d HNSW", "Recall@10 0.66523 at ef64; 0.98125 near ef256; 1.0 at ef1024", "default search effort was not harmless in mechanics"],
        ["SQLite compaction", "16,384 bytes saved per node; 48.50% measured DB reduction", "conventional storage fix with logical equivalence"],
        ["experiment cost", "$0 external cost; 0 provider calls in all retained claims", "all quality runs were local/offline"],
    ]
    story += [result_table(resource_rows, s, [43 * mm, 69 * mm, 60 * mm]), Spacer(1, 7 * mm)]
    note = Table([[p("a useful asymmetry: BM25S won candidate generation; neural selectors helped ordering. the measured architecture should reflect that split instead of forcing one model to do both jobs.", s["callout"])]], colWidths=[172 * mm])
    note.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F7F3")), ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#A8DCCF")), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [note, PageBreak()]

    story += [
        p("what changed in the system", s["h1"]),
        p("the research forced measurement repairs before feature work. search responses now expose corpus generation, resolved configuration, executed stages, channel counts, graph anchors, and reranker evidence. `as_of` reaches dense, sparse, graph, cache, and final filtering. deduplication preserves evidence identity. enabled optional stages fail closed.", s["body"]),
        p("SQLite remains authoritative and now avoids duplicating the same native 4096-d vector when there is no distinct raw representation. HNSW controls are explicit. source-derived sparse text can include speaker metadata without changing evidence IDs. the LongMemEval runner rejects oracle-context subsets instead of publishing attractive nonsense.", s["body"]),
        p("verification", s["h2"]),
    ]
    verification = [
        ["check", "result"],
        ["full Python suite", "387 passed, 3 skipped in 48.00 s"],
        ["legacy verification suite", "16 passed"],
        ["compile + dependency integrity", "passed; no broken Python requirements"],
        ["TypeScript provider tests", "4 passed"],
        ["Prettier", "all matched files use the expected style"],
        ["provider/model calls", "zero in every retained experiment artifact"],
    ]
    story += [result_table(verification, s, [62 * mm, 110 * mm]), Spacer(1, 4 * mm)]
    story += [
        p("the frontend dependency repair fetched 113 registry tarballs after pnpm rejected the existing link layout. it changed no lockfile or dependency declaration and is not counted as an experiment/provider call.", s["small"]),
        Spacer(1, 5 * mm),
        p("what is still not true", s["h1"]),
        p("there is no defensible SOTA claim, no validated 10M-100M semantic run, no grounded-answer result, no transformer KV-cache replacement result, no independent local LongMemEval retrieval corpus, and no external-backend Pareto winner.", s["body"]),
        p("the next gates", s["h2"]),
        p("1. acquire an independent exact-source memory corpus with distractors.<br/>2. run the priced, preflight-bound native 4096-d semantic Flat-versus-HNSW evaluation.<br/>3. test a compact category-aware selector against the conventional stack.<br/>4. run the episodic x semantic x temporal memory-model factorial, then the grounded reader oracle-gap study.", s["body"]),
        p("until those gates pass, the simpler architecture wins.", s["callout"]),
        PageBreak(),
        p("evidence map", s["h1"]),
        p("the frozen claim ledger contains 19 rows: 9 measured, 5 rejected, 1 invalidated, and 4 open. every non-open row resolves to an exact artifact SHA-256. ledger SHA-256: <font name='Segoe-Bold'>6a12c941f10a538fdb5fd1d35a76385e9ea8a3a177d370d6abda94efc634cca3</font>.", s["body"]),
    ]
    evidence_rows = [
        ["artifact", "SHA-256 prefix", "role"],
        ["LoCoMo BM25S baseline", "ad0bd4699065", "conventional exact-evidence baseline"],
        ["sparse multiseed v2", "5465725141c3", "speaker-prefix robustness"],
        ["field routing", "557e1160ef45", "router and sparse RRF rejection"],
        ["MiniLM rerank", "4f21a5fa87e1", "fixed-pool selection and multi-hop failure"],
        ["associative graph v2", "2d096c59fbf8", "PPR, degree sham, RRF"],
        ["BGE-M3 mechanisms", "b87aa4cdc4a8", "learned sparse and MaxSim"],
        ["FAISS 4096 frontier", "9aea9c49b4aa", "HNSW mechanics"],
        ["Turbovec multiseed", "6cca904276fd", "quantized ANN frontier"],
        ["LongMemEval failed receipt", "51a94c39b910", "dataset-admission invalidation"],
    ]
    story += [result_table(evidence_rows, s, [72 * mm, 43 * mm, 57 * mm]), Spacer(1, 7 * mm), p("selected primary sources", s["h1"])]
    refs = [
        "1. TurboQuant - https://arxiv.org/abs/2402.18096",
        "2. Vespa ranking and phased retrieval - https://docs.vespa.ai/en/ranking.html",
        "3. ColBERTv2 - https://arxiv.org/abs/2112.01488",
        "4. SPLADE++ - https://doi.org/10.1145/3477495.3531857",
        "5. DiskANN / FreshDiskANN implementation lineage - https://github.com/microsoft/DiskANN",
        "6. HippoRAG - https://arxiv.org/abs/2405.14831",
        "7. Graphiti - https://github.com/getzep/graphiti",
        "8. Matryoshka Representation Learning - https://arxiv.org/abs/2205.13147",
        "9. BGE-M3 - https://arxiv.org/abs/2402.03216",
        "10. LongMemEval - https://arxiv.org/abs/2410.10813",
    ]
    story.extend(p(item, s["reference"]) for item in refs)
    story += [Spacer(1, 5 * mm), p("full prior-art coverage: `docs/research/prior-art-mechanism-ledger.md` - all 12 Tier-S and 27 Tier-A systems in the requested order, with mechanisms, licensing boundaries, causal hypotheses, and smallest falsifying experiments.", s["small"])]

    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(json.dumps({"output": str(OUTPUT), "pages_expected": "inspect with pdfinfo", "ledger_sha256": EXPECTED_LEDGER_SHA256}, sort_keys=True))


if __name__ == "__main__":
    build()
