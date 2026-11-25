"""Export Markdown documentation files to professional, publication-quality PDFs.

Usage:
    uv run --with reportlab --with xhtml2pdf --with markdown python scripts/export_md_to_pdf.py [input.md] [output.pdf | -o output.pdf] [--title "Custom Title"]

Examples:
    uv run --with reportlab --with xhtml2pdf --with markdown python scripts/export_md_to_pdf.py docs/docker-beginner-guide.md -o docs/user-guide.pdf
    uv run --with reportlab --with xhtml2pdf --with markdown python scripts/export_md_to_pdf.py docs/docker-beginner-guide.md docs/user-guide.pdf
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


# Clean replacements for unicode emojis and symbols to ensure crisp rendering
EMOJI_REPLACEMENTS = {
    "⚡": "[Auto]",
    "🚀": "[GPU Vision]",
    "📄": "[Fast CPU]",
    "⚙️": "[Settings]",
    "⚙": "[Settings]",
    "🛡️": "[Security]",
    "🛡": "[Security]",
    "👉": "->",
    "➔": "->",
    "✓": "[OK]",
    "✨": "*",
    "▾": "v",
    "🧬": "",
    "🔬": "",
    "📊": "",
    "⚖": "",
    "🌐": "",
    "🏛": "",
    "⚠": "[WARNING]",
    "ℹ️": "[NOTE]",
    "💡": "[TIP]",
    "️": "",  # variation selector
}


def preprocess_markdown(md_text: str) -> str:
    """Preprocess markdown for clean PDF rendering: callouts, emojis, table cleanup."""
    import markdown

    # 1. Replace emojis with clean textual / badge representations
    for em, repl in EMOJI_REPLACEMENTS.items():
        md_text = md_text.replace(em, repl)

    # 2. Convert GitHub Admonitions:
    # > [!NOTE]
    # > text
    def replace_admonition(match):
        kind = match.group(1).upper()
        content = match.group(2)
        cleaned_lines = []
        for line in content.strip().splitlines():
            cleaned_line = re.sub(r"^>\s?", "", line)
            cleaned_lines.append(cleaned_line)
        inner_html = markdown.markdown("\n".join(cleaned_lines))
        return (
            f'<div class="admonition admonition-{kind.lower()}">'
            f'<div class="admonition-title">{kind}</div>\n'
            f'{inner_html}\n'
            f'</div>\n'
        )

    admonition_regex = re.compile(
        r">\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n((?:>.*(?:\n|$))*)",
        re.MULTILINE,
    )
    md_text = admonition_regex.sub(replace_admonition, md_text)

    return md_text


def get_css_template(font_paths: dict, doc_title: str) -> str:
    """Generate scoped CSS with embedded fonts and print margins."""
    return f"""
@font-face {{
    font-family: 'AppFont';
    src: url('{font_paths["regular"]}');
}}
@font-face {{
    font-family: 'AppFont';
    src: url('{font_paths["bold"]}');
    font-weight: bold;
}}
@font-face {{
    font-family: 'AppFont';
    src: url('{font_paths["italic"]}');
    font-style: italic;
}}
@font-face {{
    font-family: 'AppFont';
    src: url('{font_paths["bold_italic"]}');
    font-weight: bold;
    font-style: italic;
}}
@font-face {{
    font-family: 'AppMono';
    src: url('{font_paths["mono"]}');
}}

@page {{
    size: a4 portrait;
    margin: 18mm 16mm 18mm 16mm;
    @top-right {{
        content: "{doc_title}";
        font-family: 'AppFont';
        font-size: 8pt;
        color: #64748b;
    }}
    @bottom-center {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'AppFont';
        font-size: 8pt;
        color: #64748b;
    }}
}}

body {{
    font-family: 'AppFont', Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.45;
    color: #1e293b;
}}

h1 {{
    font-size: 19pt;
    color: #1e40af;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 5px;
    margin-top: 14px;
    margin-bottom: 10px;
    page-break-after: avoid;
}}

h2 {{
    font-size: 13.5pt;
    color: #1e3a8a;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 3px;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
}}

h3 {{
    font-size: 11pt;
    color: #2563eb;
    margin-top: 12px;
    margin-bottom: 5px;
    page-break-after: avoid;
}}

h4 {{
    font-size: 9.5pt;
    color: #0f172a;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
}}

p {{
    margin-top: 4px;
    margin-bottom: 6px;
}}

ul, ol {{
    margin-top: 3px;
    margin-bottom: 6px;
    padding-left: 18px;
}}

li {{
    margin-bottom: 2.5px;
}}

table {{
    width: 100%;
    margin: 8px 0;
}}

th, td {{
    border: 0.5pt solid #cbd5e1;
    padding: 4.5px 7px;
    font-size: 8.5pt;
    text-align: left;
    vertical-align: top;
}}

th {{
    background-color: #f1f5f9;
    font-weight: bold;
    color: #0f172a;
}}

pre, code {{
    font-family: 'AppMono', Courier, monospace;
    font-size: 8pt;
    background-color: #f8fafc;
}}

code {{
    padding: 1px 3px;
    color: #0369a1;
    border: 0.5pt solid #e2e8f0;
}}

pre {{
    padding: 7px;
    margin: 6px 0;
    border: 0.5pt solid #cbd5e1;
    background-color: #f8fafc;
}}

pre code {{
    border: none;
    padding: 0;
    color: #0f172a;
}}

blockquote {{
    border-left: 3px solid #94a3b8;
    background-color: #f8fafc;
    padding: 5px 10px;
    margin: 6px 0;
    font-style: italic;
    color: #475569;
}}

.admonition {{
    border-left: 4px solid #3b82f6;
    background-color: #eff6ff;
    padding: 6px 10px;
    margin: 8px 0;
}}

.admonition-tip {{
    border-left-color: #10b981;
    background-color: #ecfdf5;
}}

.admonition-note {{
    border-left-color: #3b82f6;
    background-color: #eff6ff;
}}

.admonition-important {{
    border-left-color: #8b5cf6;
    background-color: #f5f3ff;
}}

.admonition-warning {{
    border-left-color: #f59e0b;
    background-color: #fffbeb;
}}

.admonition-caution {{
    border-left-color: #ef4444;
    background-color: #fef2f2;
}}

.admonition-title {{
    font-weight: bold;
    font-size: 8.5pt;
    margin-bottom: 3px;
    color: #0f172a;
}}

hr {{
    border: 0.5pt solid #e2e8f0;
    margin: 10px 0;
}}
"""


def convert_md_to_pdf(input_file: Path, output_file: Path, title: str | None = None) -> bool:
    """Read markdown, convert to HTML, render to styled PDF."""
    import markdown
    from xhtml2pdf import pisa

    if not input_file.exists():
        print(f"Error: Input file does not exist: {input_file}", file=sys.stderr)
        return False

    with open(input_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Extract or fallback document title
    if not title:
        first_line = md_text.strip().splitlines()[0] if md_text.strip() else ""
        if first_line.startswith("#"):
            title = first_line.lstrip("#").strip()
        else:
            title = input_file.stem.replace("-", " ").title()

    processed_md = preprocess_markdown(md_text)

    # Convert to HTML
    html_body = markdown.markdown(
        processed_md,
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
    )

    # Prepare fonts in an OS temp folder so xhtml2pdf resource policy permits access
    temp_font_dir = Path(tempfile.gettempdir()) / "second-brain-pdf-fonts"
    temp_font_dir.mkdir(parents=True, exist_ok=True)

    windir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    font_candidates = {
        "regular": windir / "arial.ttf",
        "bold": windir / "arialbd.ttf",
        "italic": windir / "ariali.ttf",
        "bold_italic": windir / "arialbi.ttf",
        "mono": windir / "consola.ttf",
    }

    local_fonts = {}
    for key, path in font_candidates.items():
        dest = temp_font_dir / path.name
        if path.exists() and (not dest.exists() or dest.stat().st_size != path.stat().st_size):
            shutil.copy2(path, dest)
        local_fonts[key] = dest.resolve().as_posix()

    css = get_css_template(local_fonts, title)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f_out:
        pisa_status = pisa.CreatePDF(full_html, dest=f_out)

    if pisa_status.err:
        print(f"PDF creation completed with warnings/errors: {pisa_status.err}", file=sys.stderr)
        return False

    file_size_kb = output_file.stat().st_size / 1024
    print(f"Successfully generated PDF: {output_file} ({file_size_kb:.1f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export markdown files to styled PDFs.")
    parser.add_argument(
        "input",
        nargs="?",
        default="docs/docker-beginner-guide.md",
        help="Path to input .md file (default: docs/docker-beginner-guide.md)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to output .pdf file (default: <input>.pdf)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_flag",
        default=None,
        help="Path to output .pdf file (overrides positional output)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title for running headers",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    resolved_output = args.output_flag or args.output
    if resolved_output:
        output_path = Path(resolved_output)
    else:
        output_path = input_path.with_suffix(".pdf")

    success = convert_md_to_pdf(input_path, output_path, title=args.title)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
