"""Render an eval run as Markdown and as a self-contained HTML page."""

from __future__ import annotations

import html
from typing import Any

PCT_KEYS = (
    "correctness",
    "citation_faithfulness",
    "answers_fully_grounded",
    "abstention_accuracy",
    "false_abstention_rate",
)
HEADLINES = [
    ("Answer correctness (LLM judge)", "correctness"),
    ("Citation faithfulness (code-checked)", "citation_faithfulness"),
    ("Answers with every citation verified", "answers_fully_grounded"),
    ("Abstention on unanswerable questions", "abstention_accuracy"),
    ("False abstention on answerable questions", "false_abstention_rate"),
]


def pct(value: float) -> str:
    """Format a fraction as a percentage with one decimal."""
    return f"{value * 100:.1f}%"


def retrieval_rows(retrieval: dict[str, dict[str, float]]) -> list[list[str]]:
    """Table rows (variant, hit@1, hit@3, hit@5, MRR) for the retrieval ablation."""
    return [
        [name, pct(m["hit@1"]), pct(m["hit@3"]), pct(m["hit@5"]), f"{m['mrr']:.3f}"] for name, m in retrieval.items()
    ]


def question_rows(results: list[dict[str, Any]]) -> list[list[str]]:
    """Per-question table rows: id, type, retrieval rank, outcome, verified citations, verdict."""
    rows = []
    for r in results:
        answer, item = r["answer"], r["item"]
        cites = answer["citations"]
        verified = f"{sum(c['verified'] for c in cites)}/{len(cites)}" if cites else "-"
        rank = str(r["hit_rank"]) if r["hit_rank"] else ("-" if not item["answerable"] else "miss")
        outcome = "abstained" if answer["abstained"] else "answered"
        expected = "answerable" if item["answerable"] else "unanswerable"
        verdict = r["verdict"] or ("correct abstention" if answer["abstained"] else "should have abstained")
        rows.append([item["id"], expected, rank, outcome, verified, verdict])
    return rows


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _md_headline(summary: dict[str, Any]) -> str:
    """Markdown table of the headline metrics plus latency and token totals."""
    rows = [[label, pct(summary[key])] for label, key in HEADLINES]
    rows.append(["Mean latency per question", f"{summary['mean_latency_s']:.1f} s"])
    rows.append(["Tokens (input / output)", f"{summary['input_tokens']:,} / {summary['output_tokens']:,}"])
    return md_table(["Metric", "Value"], rows)


def render_markdown(run: dict[str, Any], retrieval: dict[str, dict[str, float]]) -> str:
    """Render the full Markdown report.

    Args:
        run: Output of evaluation.results_to_json.
        retrieval: Retrieval ablation metrics per variant.

    Returns:
        Markdown text.
    """
    meta, summary = run["meta"], run["summary"]
    counts = (
        f"{summary['questions']} questions ({summary['answerable']} answerable, {summary['unanswerable']} "
        f"unanswerable). {summary['citations_verified']} of {summary['citations']} citations verified."
    )
    blocks = [f"# Eval report: {meta['run_name']}", _meta_line(meta), counts, "## Headline metrics"]
    blocks += [_md_headline(summary), "## Retrieval (answerable questions, gold evidence in top k)"]
    blocks.append(md_table(["Retriever", "hit@1", "hit@3", "hit@5", "MRR"], retrieval_rows(retrieval)))
    header = ["id", "expected", "hit rank", "outcome", "verified cites", "verdict"]
    blocks += ["## Per question", md_table(header, question_rows(run["results"])), "## Answers"]
    blocks += [_md_answer(r) for r in run["results"]]
    return "\n\n".join(blocks) + "\n"


def _meta_line(meta: dict[str, Any]) -> str:
    """One-line description of models and settings."""
    return (
        f"Run at {meta['timestamp']} - answer model `{meta['answer_model']}`, judge `{meta['judge_model']}`, "
        f"embedder `{meta['embed_model']}`, reranker `{meta['rerank_model']}`, top-k {meta['final_k']}, "
        f"LLM backend `{meta.get('llm_backend', 'api')}`."
    )


def _md_answer(result: dict[str, Any]) -> str:
    """Markdown block with one question, its answer and its citations."""
    answer = result["answer"]
    lines = [f"### {result['item']['id']}: {result['item']['question']}", "", answer["text"], ""]
    for c in answer["citations"]:
        mark = "verified" if c["verified"] else "NOT FOUND"
        lines.append(f'- [{c["number"]}] {c["title"]}, {c["location"]} ({mark}): "{c["quote"]}"')
    if result["judge_reasoning"]:
        lines += ["", f"Judge: **{result['verdict']}** - {result['judge_reasoning']}"]
    return "\n".join(lines)


CSS = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f6f7f9;color:#1d2430}
main{max-width:1080px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
.meta{color:#5b6472;font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:20px}
.card{background:#fff;border:1px solid #e3e6eb;border-radius:10px;padding:14px 16px}
.card b{display:block;font-size:28px;color:#1f4fd1}.card span{font-size:13px;color:#5b6472}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e3e6eb;border-radius:10px;
overflow:hidden;font-size:14px}th,td{padding:8px 10px;border-bottom:1px solid #eef0f3;text-align:left;
vertical-align:top}th{background:#f0f2f5;font-weight:600}
.ok{color:#137333;font-weight:600}.bad{color:#b3261e;font-weight:600}.mid{color:#9a6700;font-weight:600}
details{background:#fff;border:1px solid #e3e6eb;border-radius:10px;padding:10px 14px;margin:8px 0}
summary{cursor:pointer;font-weight:600}blockquote{margin:6px 0;padding:4px 10px;border-left:3px solid #c9d3f5;
color:#36404f;background:#f8f9fc}
"""


def _verdict_class(text: str) -> str:
    """CSS class for a verdict cell."""
    if text in ("correct", "correct abstention"):
        return "ok"
    return "mid" if text == "partially_correct" else "bad"


def html_table(header: list[str], rows: list[list[str]], colour_last: bool = False) -> str:
    """Render an escaped HTML table (optionally colouring the last column by verdict)."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = []
    for row in rows:
        cells = [f"<td>{html.escape(c)}</td>" for c in row]
        if colour_last:
            cells[-1] = f'<td class="{_verdict_class(row[-1])}">{html.escape(row[-1])}</td>'
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _html_answer(result: dict[str, Any]) -> str:
    """Collapsible HTML block with one answer and its quoted evidence."""
    answer, item = result["answer"], result["item"]
    quotes = "".join(
        f"<blockquote><b>[{c['number']}]</b> {html.escape(c['title'])}, {html.escape(c['location'])} "
        f'<span class="{"ok" if c["verified"] else "bad"}">{"verified" if c["verified"] else "not found"}</span>'
        f"<br>&ldquo;{html.escape(c['quote'])}&rdquo;</blockquote>"
        for c in answer["citations"]
    )
    judge = (
        (
            f'<p><b>Judge:</b> <span class="{_verdict_class(result["verdict"])}">{html.escape(result["verdict"])}'
            f"</span> - {html.escape(result['judge_reasoning'])}</p>"
        )
        if result["judge_reasoning"]
        else ""
    )
    return (
        f"<details><summary>{html.escape(item['id'])}: {html.escape(item['question'])}</summary>"
        f"<p>{html.escape(answer['text'])}</p>{quotes}{judge}</details>"
    )


def render_html(run: dict[str, Any], retrieval: dict[str, dict[str, float]]) -> str:
    """Render the full self-contained HTML report.

    Args:
        run: Output of evaluation.results_to_json.
        retrieval: Retrieval ablation metrics per variant.

    Returns:
        HTML document text.
    """
    meta, summary = run["meta"], run["summary"]
    cards = "".join(
        f'<div class="card"><b>{pct(summary[key])}</b><span>{html.escape(label)}</span></div>'
        for label, key in HEADLINES
    )
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><title>Eval report: "
        f"{html.escape(meta['run_name'])}</title><style>{CSS}</style></head><body><main>"
        f"<h1>Eval report: {html.escape(meta['run_name'])}</h1>"
        f"<p class=meta>{html.escape(_meta_line(meta).replace('`', ''))}<br>{summary['questions']} questions "
        f"({summary['answerable']} answerable, {summary['unanswerable']} unanswerable) - "
        f"{summary['citations_verified']} of {summary['citations']} citations verified - mean latency "
        f"{summary['mean_latency_s']:.1f} s</p><div class=cards>{cards}</div>"
        f"<h2>Retrieval (answerable questions, gold evidence in top k)</h2>"
        f"{html_table(['Retriever', 'hit@1', 'hit@3', 'hit@5', 'MRR'], retrieval_rows(retrieval))}"
        f"<h2>Per question</h2>"
        f"{html_table(['id', 'expected', 'hit rank', 'outcome', 'verified cites', 'verdict'], question_rows(run['results']), True)}"  # noqa: E501
        f"<h2>Answers</h2>{''.join(_html_answer(r) for r in run['results'])}</main></body></html>"
    )
