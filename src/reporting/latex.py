from __future__ import annotations

import csv
from pathlib import Path

from src.utils.paths import ensure_dir


def escape_latex(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def csv_to_latex(csv_path: str | Path, tex_path: str | Path, caption: str, label: str) -> None:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ensure_dir(Path(tex_path).parent)
    if not rows:
        Path(tex_path).write_text("% No rows available.\n", encoding="utf-8")
        return
    columns = rows[0].keys()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label}}}",  # labels are keys, not typeset text -- must NOT be escaped
        r"\begin{tabular}{" + "l" * len(list(columns)) + "}",
        r"\toprule",
        " & ".join(escape_latex(col) for col in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(escape_latex(row[col]) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    Path(tex_path).write_text("\n".join(lines), encoding="utf-8")
