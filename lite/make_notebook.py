"""Generate the WASM/JupyterLite notebook (document_similarity.ipynb).

Run:  python3 lite/make_notebook.py
Writes lite/content/document_similarity.ipynb
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "content", "document_similarity.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(lines)}


cells = [
    md(
        "# Document Similarity — in-browser (WASM) edition\n",
        "\n",
        "This is the **JupyterLite / Pyodide** build of the Document Similarity tool. ",
        "Everything runs locally in your web browser — there is no server, no login, ",
        "and your documents never leave your computer.\n",
        "\n",
        "It uses [MinHash](https://ekzhu.com/datasketch/minhash.html) to estimate the ",
        "Jaccard similarity between documents so you can find and remove duplicate or ",
        "near-duplicate texts in your corpus.\n",
        "\n",
        "> **This in-browser version is intended for small corpora and for testing.** ",
        "It runs entirely in your browser tab, which is single-threaded and limited by your ",
        "browser's memory, so large corpora are slow and can **crash the tab**. For larger ",
        "datasets, use the Cloud (Binder) version linked in the ",
        "[README](https://github.com/Australian-Text-Analytics-Platform/document-similarity#readme).\n",
        "\n",
        "> **Your data stays private:** because everything runs locally, your documents are ",
        "**never uploaded to any server** — a good fit for quick trials or sensitive data, as ",
        "long as the corpus is small.\n",
        "\n",
        "> **First run:** the cell below downloads the Python packages into your browser. ",
        "This takes ~30–60 seconds the first time and is cached afterwards. Run the cells ",
        "in order from top to bottom.",
    ),
    md("## 1. Setup\n", "Run this cell first to install the required packages into the in-browser kernel."),
    code(
        "# Install packages into the Pyodide (in-browser) kernel.\n",
        "# numpy / pandas / scipy / matplotlib / ipywidgets ship with Pyodide; the rest\n",
        "# are fetched here. 'datasketch' is the only pure-PyPI package (MinHash/LSH).\n",
        "import piplite\n",
        "await piplite.install([\n",
        "    'ipywidgets',\n",
        "    'datasketch',\n",
        "    'nltk',\n",
        "    'tqdm',\n",
        "    'openpyxl',\n",
        "    'seaborn',\n",
        "    'bokeh',\n",
        "])\n",
        "print('Packages installed.')",
    ),
    code(
        "# Load the Document Similarity tool and the lightweight in-browser file uploader.\n",
        "print('Loading DocumentSimilarity...')\n",
        "from document_similarity import DocumentSimilarity, FileUploaderWidget\n",
        "\n",
        "ds = DocumentSimilarity()\n",
        "print('Finished loading.')",
    ),
    md(
        "## 2. Load your data\n",
        "\n",
        "Upload one or more **.txt**, **.csv**, **.xlsx** files, or a **.zip** archive of them.\n",
        "\n",
        "* **.txt** — each file is treated as one document.\n",
        "* **.csv / .xlsx** — each row is one document. The tool uses a column named `text` "
        "(falls back to the first column) and an optional `text_name` column for the document name.\n",
        "\n",
        "**Want to try it right away?** A few sample documents are included in the "
        "`sample_documents/` folder (two climate texts and two coffee texts are deliberate "
        "near-duplicates). Use the file picker to select them.",
    ),
    code(
        "# Display the uploader. Choose your files, then click 'Build corpus'.\n",
        "loader = FileUploaderWidget()\n",
        "loader",
    ),
    md(
        "<b>Automatic deduplication of identical documents</b><br>\n",
        "When you load the corpus below, the tool first finds any 100% identical documents ",
        "and keeps only the first of each (alphabetically by name). The similarity analysis ",
        "then runs on the remaining, non-identical texts.",
    ),
    code(
        "# Load the corpus you just built into the tool.\n",
        "ds.set_text_df(loader.corpus_df)",
    ),
    code(
        "# Preview the loaded documents\n",
        "ds.text_df.head()",
    ),
    md(
        "## 3. Calculate document similarity\n",
        "\n",
        "Set the parameters below, then run the calculation.\n",
        "\n",
        "* **ngram_value** — number of consecutive words compared at a time (1 = word level).\n",
        "* **actual_jaccard** — `False` uses fast MinHash estimation (recommended); `True` computes exact Jaccard.\n",
        "* **num_perm** — number of MinHash permutations; higher = more accurate, slower.\n",
        "* **similarity_cutoff** — documents at or above this score (0–1) are flagged as similar.",
    ),
    code(
        "ngram_value = 1\n",
        "actual_jaccard = False  # True or False\n",
        "ds.exclude_punc = True  # exclude punctuation when comparing\n",
        "num_perm = 256\n",
        "similarity_cutoff = 0.6  # between 0 and 1",
    ),
    code(
        "# Run the similarity calculation\n",
        "ds.calculate_similarity(ngram_value, num_perm, similarity_cutoff, actual_jaccard)",
    ),
    md(
        "## 4. Analyse similar documents\n",
        "\n",
        "The histogram shows how many similar documents are found at each Jaccard similarity level.",
    ),
    code("ds.plot_hash_similarity_by_source(ds.deduplication_df)"),
    md(
        "<b>Heatmap of similar documents</b><br>\n",
        "The heatmap shows Jaccard similarity scores between pairs of similar documents ",
        "(hover over a cell to see the document names). Only pairs above the cut-off are shown.\n",
        "\n",
        "`plot_range` controls which pairs are drawn: `'y'` for all, `'n'` for none, a single ",
        "number like `30`, or a range like `10-25`.",
    ),
    code(
        "plot_width = 900\n",
        "plot_height = 800\n",
        "font_size = '14px'\n",
        "text_color = 'white'\n",
        "plot_range = 'y'  # 'y' (all), 'n' (none), '30', or '10-25'\n",
        "\n",
        "print('There are {} document pair(s) in the current process.'.format(ds.deduplication_df.shape[0]))\n",
        "ds.plot_heatmap_similarity(similarity_cutoff, plot_width, plot_height, font_size, text_color, plot_range)",
    ),
    md(
        "<b>Review and decide</b><br>\n",
        "The interactive table below lists each pair of similar documents with a recommended ",
        "'keep' or 'remove' status (by default the shorter document of a pair is removed). ",
        "Use the controls to view each pair side-by-side and change the decision.",
    ),
    code("ds.display_deduplication_text()"),
    md(
        "## 5. Save your results\n",
        "\n",
        "Save the documents you chose to keep (or the ones marked for removal) and download them. ",
        "Click the download link that appears after you click **'Save non-duplicated texts'** or ",
        "**'Save duplicated texts'**.\n",
        "\n",
        "**This in-browser (Lite) version always saves a single `.csv`** (with `text_name` and `text` ",
        "columns), regardless of how you uploaded your data. This keeps saving fast and within the ",
        "browser's memory limits. For very large corpora, use the ",
        "[Cloud (Binder) version](https://github.com/Australian-Text-Analytics-Platform/document-similarity#cloud-version-binder), ",
        "which can also export a zip of `.txt` files to match a file-based upload and streams the "
        "download from a server.",
    ),
    code(
        "rows_to_display = 5\n",
        "ds.finalise_and_save(rows_to_display)",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python (Pyodide)", "language": "python", "name": "python"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT)
