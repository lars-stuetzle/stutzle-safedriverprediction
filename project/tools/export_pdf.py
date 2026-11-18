"""Export executed notebooks to landscape PDF files (A4), next to the notebooks.

The notebook is converted to HTML with nbconvert and printed to PDF with a headless Chromium-based browser
(Microsoft Edge or Google Chrome), so no LaTeX installation is needed. The notebooks are not executed again:
run them first (e.g. with "Restart & Run All").

Usage (from the project folder):
    python tools/export_pdf.py notebooks/01_eda.ipynb notebooks/02_preparation.ipynb ...
    python tools/export_pdf.py            # all notebooks in notebooks/
"""

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

PROJECT_DIR = Path(__file__).resolve().parent.parent

# landscape A4 pages; smaller font for code and outputs so that wide tables fit on one line
PRINT_CSS = """
<style>
  @page { size: A4 landscape; margin: 10mm 12mm; }
  body { margin: 0 !important; padding: 0 !important; }
  .jp-Notebook { padding: 0 !important; }
  .jp-InputArea-editor, .jp-OutputArea-output pre { font-size: 10.5px !important; line-height: 1.3 !important; }
  .jp-RenderedHTMLCommon { font-size: 13px !important; }
  .jp-RenderedImage img { max-width: 100% !important; height: auto !important; }
  .jp-RenderedImage, .jp-InputArea { break-inside: avoid; }
  .jp-MarkdownCell h2 { break-after: avoid; }
</style>
<script>
  // shrink code, text output or tables that are wider than the page, so that nothing is cut off
  window.addEventListener('load', () => {
    document.querySelectorAll('.jp-OutputArea-output pre, .jp-OutputArea-output table, .jp-InputArea-editor pre')
      .forEach(element => {
        const available = element.parentElement.clientWidth;
        if (element.scrollWidth > available) {
          element.style.zoom = (available / element.scrollWidth).toFixed(3);
        }
      });
  });
</script>
"""

BROWSER_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]


def find_browser():
    """Path of a Chromium-based browser that can print to PDF."""
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in BROWSER_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    sys.exit("No Edge/Chrome/Chromium found: install one or export from Jupyter (File > Save and Export As).")


def export(notebook_path, browser):
    """Convert one notebook to HTML, then print it to <notebook>.pdf in landscape."""
    notebook = nbformat.read(notebook_path, as_version=4)
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
    html = html.replace("</head>", PRINT_CSS + "</head>", 1)

    pdf_path = notebook_path.with_suffix(".pdf")
    pdf_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        html_path = Path(temp_dir) / "notebook.html"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(temp_dir) / 'profile'}",
                        f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                       check=True, capture_output=True)
        # on Windows the browser may return before the PDF is written: wait until its size is stable
        wait_for_file(pdf_path)
    print(f"{notebook_path.name} -> {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)")


def wait_for_file(path, timeout_s=180):
    """Wait until the file exists and its size has stopped changing."""
    last_size = -1
    for _ in range(timeout_s):
        if path.exists() and path.stat().st_size == last_size and last_size > 0:
            return
        last_size = path.stat().st_size if path.exists() else -1
        time.sleep(1)
    sys.exit(f"The PDF {path} was not created within {timeout_s} s.")


if __name__ == "__main__":
    paths = [Path(arg).resolve() for arg in sys.argv[1:]] or sorted((PROJECT_DIR / "notebooks").glob("*.ipynb"))
    browser = find_browser()
    for path in paths:
        export(path, browser)
