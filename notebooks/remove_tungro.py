"""Remove tungro from all simulation notebooks."""
import json
import re
from pathlib import Path

NOTEBOOKS = [
    "rice_disease_simulation.ipynb",
    "rice_healthy_leaf_simulation.ipynb",
    "rice_blast_simulation.ipynb",
]

TUNGRO_SCORE_LINE = re.compile(
    r'\n    "tungro":[^\n]+\n(?:                        \+[^\n]+\n)?',
)
TUNGRO_REC_BLOCK = re.compile(
    r'\n    "tungro": \([^)]+\(?:[^)]*\)[^)]*\)*\),',
    re.DOTALL,
)


def strip_tungro(text: str) -> str:
    text = text.replace(', "tungro"', "")
    text = text.replace('"tungro", ', "")
    text = text.replace('"tungro": "#e07b00",\n', "")
    text = text.replace("; *Tungro* = broad yellow-orange discoloration", "")
    text = text.replace(" *Leaf Smut* = many tiny dark specks; *Tungro* = broad yellow-orange discoloration;",
                        " *Leaf Smut* = many tiny dark specks;")
    text = text.replace("5 rice diseases + *healthy*", "4 rice diseases + *healthy*")
    text = text.replace("the 5 rice diseases + *healthy*", "the 4 rice diseases + *healthy*")
    text = text.replace("Score the 5 rice diseases + *healthy*", "Score the 4 rice diseases + *healthy*")
    text = text.replace("six labels", "five labels")
    text = text.replace("The six labels", "The five labels")
    text = text.replace("`bacterial blight`, `blast`, `brown spot`, `leaf smut`, `tungro`, plus `healthy`.",
                        "`bacterial blight`, `blast`, `brown spot`, `leaf smut`, plus `healthy`.")
    text = text.replace("early BLB, blast, and tungro signs.", "early BLB and blast signs.")
    text = TUNGRO_SCORE_LINE.sub("\n", text)
    text = re.sub(
        r'         "Avoid excess nitrogen; remove heavily infected leaves."\]\),\n\n'
        r'        "Rice Tungro \(virus complex\).*?'
        r'         "Plant tungro-resistant varieties and synchronise planting."\]\),\n\n',
        '         "Avoid excess nitrogen; remove heavily infected leaves."]),\n\n',
        text,
        flags=re.DOTALL,
    )
    return text


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] in ("markdown", "code"):
            src = "".join(cell["source"])
            new_src = strip_tungro(src)
            if new_src != src:
                cell["source"] = [line + "\n" for line in new_src.splitlines(True)]
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {path.name}")


if __name__ == "__main__":
    root = Path(__file__).parent
    for name in NOTEBOOKS:
        patch_notebook(root / name)
