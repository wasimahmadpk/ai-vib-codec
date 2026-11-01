"""Terminal tables. Color is added only when stdout is a terminal."""

from __future__ import annotations

import os
import sys

GREEN = "\033[1;32m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def color_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    highlight: set[str] | None = None,
    note: str = "",
    numeric: bool = True,
) -> None:
    """Print one aligned table. Rows whose first cell is in ``highlight`` are green."""
    chosen = highlight or set()
    use_color = color_enabled()
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def paint(line: str, code: str) -> str:
        if not use_color:
            return line
        return f"{code}{line}{RESET}"

    def join(cells: list[str], header: bool = False) -> str:
        parts = []
        for index, cell in enumerate(cells):
            text = cell.rjust(widths[index]) if numeric and index and not header else cell.ljust(widths[index])
            parts.append(text)
        return "  ".join(parts)

    rule = "  ".join("─" * width for width in widths)
    print()
    print(paint(title, BOLD))
    if any(headers):
        print(paint(join(headers, header=True), DIM))
        print(paint(rule, DIM))
    for row in rows:
        line = join(row)
        marked = row[0] in chosen
        if marked and use_color:
            print(paint(f"▸ {line}", GREEN))
        elif marked:
            print(f"▸ {line}")
        else:
            print(f"  {line}")
    if note:
        print(paint(note, DIM))
