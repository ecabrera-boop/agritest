"""Clone disease simulation notebook for Rice Blast leaf sample."""
import json
from pathlib import Path

src = Path(__file__).parent / "rice_disease_simulation.ipynb"
dst = Path(__file__).parent / "rice_blast_simulation.ipynb"

nb = json.loads(src.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nb["cells"][0]["source"] = [
    "# Rice Blast Disease Identification — Interactive Simulation\n",
    "\n",
    "This notebook runs the disease identification pipeline on a **rice leaf with classic Blast symptoms**:\n",
    "spindle-shaped lesions with gray-white centers, dark brown borders, and yellow halos.\n",
    "\n",
    "It mirrors the real pipeline in `ml/` (`preprocessor.py`, `lesion_features.py`, `layer2_disease.py`) "
    "and is **fully self-contained** (numpy, scipy, Pillow, matplotlib only).\n",
    "\n",
    "## What to expect on a Blast-infected leaf\n",
    "\n",
    "| Step | Blast signal | Graph you'll see |\n",
    "|------|--------------|------------------|\n",
    "| 1. Load | Green leaf with spindle-shaped spots | Original photo |\n",
    "| 2. Preprocess | Letterbox resize to 128x128 | Before vs after |\n",
    "| 3. Color analysis | Gray centers + brown margins shift hue/sat histograms | Channel maps + histograms |\n",
    "| 4. Segment lesion | Off-green patches detected along the leaf | Red lesion overlay |\n",
    "| 5. Extract features | High aspect ratio + eccentricity (spindle shape) | Feature bar charts |\n",
    "| 6. Classify | **Blast** should score highest | Confidence bar chart |\n",
    "| 7. Flow diagram | Photo -> diagnosis | Pipeline flowchart |\n",
    "| 8. Dashboard | Blast management tips (PhilRice-aligned) | Summary dashboard |\n",
    "\n",
    "**Sample image:** `assets/blast_leaf.png` — rice leaf with spindle Blast lesions.\n",
    "\n",
    "**Compare with:**\n",
    "- `rice_healthy_leaf_simulation.ipynb` (`assets/healthy_leaf.jpg`)\n",
    "- `rice_disease_simulation.ipynb` (`assets/sample_leaf.png` — brown spot sample)\n",
    "\n",
    "> **Note:** Step 6 uses a transparent rule-based scorer for teaching, not the production `.pkl` model.\n",
]

src1 = "".join(nb["cells"][1]["source"])
src1 = src1.replace(
    'IMAGE_PATH = Path("assets/sample_leaf.png")',
    'IMAGE_PATH = Path("assets/blast_leaf.png")',
)
src1 = src1.replace(
    '    IMAGE_PATH = Path("notebooks/assets/sample_leaf.png")  # if run from repo root',
    '    IMAGE_PATH = Path("notebooks/assets/blast_leaf.png")  # if run from repo root',
)
nb["cells"][1]["source"] = [line + "\n" for line in src1.splitlines(True)]

nb["cells"][4]["source"] = [
    "## Step 1 — Load the Blast-infected leaf photo\n",
    "\n",
    "Classic **Rice Blast** (*Pyricularia oryzae*) shows **spindle- or diamond-shaped lesions**: "
    "a gray-white necrotic center, a dark brown margin, and sometimes a yellow chlorotic halo on green leaf tissue.\n",
]

nb["cells"][8]["source"] = [
    "## Step 3 — Look at the colors (RGB and HSV)\n",
    "\n",
    "Blast lesions add **gray desaturated centers** (low S, mid V) and **brown margins** (hue toward red/orange). "
    "Watch the **Saturation** and **Hue** histograms — lesion pixels pull away from the healthy green peak.\n",
]

nb["cells"][10]["source"] = [
    "## Step 4 — Segment the Blast lesions\n",
    "\n",
    "The mask marks off-green pixels inside the leaf: brown/yellow hues, dark necrotic tissue, and pale gray centers. "
    "On Blast leaves you should see **multiple elongated red patches** aligned with the leaf veins.\n",
]

nb["cells"][12]["source"] = [
    "## Step 5 — Extract lesion features\n",
    "\n",
    "Blast lesions are **spindle-shaped** — expect **high aspect ratio and eccentricity**, moderate grayness in HSV moments, "
    "and texture (GLCM) distinct from uniform healthy green.\n",
]

nb["cells"][14]["source"] = [
    "## Step 6 — Score the diseases (expect BLAST to win)\n",
    "\n",
    "The scorer boosts **blast** when it sees gray lesion centers plus elongated shape (high aspect ratio / eccentricity). "
    "**Brown spot** needs rounder lesions; **bacterial blight** needs long pale streaks.\n",
]

nb["cells"][18]["source"] = [
    "## Step 8 — Final diagnosis dashboard\n",
    "\n",
    "Expected result: **BLAST** with PhilRice-aligned management steps (reduce nitrogen, scout at tillering/booting, fungicide at heading).\n",
]

nb["cells"].extend([
    {
        "cell_type": "markdown",
        "id": "compare-blast-md",
        "metadata": {},
        "source": [
            "## Side-by-side comparison (Blast vs healthy vs brown spot)\n",
            "\n",
            "Compare lesion coverage across the three sample notebooks.\n",
        ],
    },
    {
        "cell_type": "code",
        "id": "compare-blast-code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# === Bonus: compare Blast vs healthy vs brown-spot samples =================\n",
            "samples = [\n",
            "    (\"Blast (this notebook)\", Path(\"assets/blast_leaf.png\"), CLASS_COLORS[\"blast\"]),\n",
            "    (\"Healthy leaf\", Path(\"assets/healthy_leaf.jpg\"), CLASS_COLORS[\"healthy\"]),\n",
            "    (\"Brown spot sample\", Path(\"assets/sample_leaf.png\"), CLASS_COLORS[\"brown spot\"]),\n",
            "]\n",
            "rows = []\n",
            "for label, path, color in samples:\n",
            "    if not path.exists():\n",
            "        path = Path(\"notebooks\") / path\n",
            "    if not path.exists():\n",
            "        continue\n",
            "    pil = Image.open(path).convert(\"RGB\")\n",
            "    arr = np.array(letterbox_resize(pil, IMG_SIZE, BG_COLOR))\n",
            "    les, lf = segment_lesion(arr)\n",
            "    cov = les.sum() / max(lf.sum(), 1)\n",
            "    rows.append((label, arr, cov, color))\n",
            "\n",
            "if rows:\n",
            "    fig, axes = plt.subplots(1, len(rows) + 1, figsize=(4 * (len(rows) + 1), 4))\n",
            "    for ax, (label, arr, cov, color) in zip(axes[:-1], rows):\n",
            "        ax.imshow(arr); ax.set_title(f\"{label}\\n{cov*100:.1f}% lesion\"); ax.axis(\"off\")\n",
            "    axes[-1].bar([r[0] for r in rows], [r[2]*100 for r in rows],\n",
            "                   color=[r[3] for r in rows], edgecolor=\"black\")\n",
            "    axes[-1].set_ylabel(\"Lesion coverage (%)\"); axes[-1].set_title(\"Coverage comparison\")\n",
            "    axes[-1].tick_params(axis=\"x\", rotation=15, labelsize=8)\n",
            "    fig.suptitle(\"Blast vs healthy vs brown spot — lesion coverage\", y=1.03)\n",
            "    plt.tight_layout(); plt.show()\n",
            "    print(f\"This notebook diagnosis: {diagnosis.upper()} ({confidence*100:.1f}%)\")\n",
            "else:\n",
            "    print(\"Comparison images not found in assets/.\")\n",
        ],
    },
])

dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {dst} ({len(nb['cells'])} cells)")

# Patch segment_lesion + blast scorer after write (clone uses base disease notebook).
nb = json.loads(dst.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "def segment_lesion(arr):" in src and "gray_center" not in src:
        cell["source"] = [line + "\n" for line in src.replace(
            "    pale_spot = (s_ < 0.15) & (v_ >= 0.65)\n"
            "    lesion = _clean_mask(leaf & (brown_yellow | dark_necrotic | pale_spot))\n",
            "    pale_spot = (s_ < 0.15) & (v_ >= 0.65)\n"
            "    gray_center = (s_ < 0.28) & (v_ >= 0.22) & (v_ <= 0.78)\n"
            "    hsv_lesion = leaf & (brown_yellow | dark_necrotic | pale_spot | gray_center)\n"
            "    lesion = _clean_mask(hsv_lesion)\n",
        ).replace(
            "        if otsu_mask.sum() >= MIN_LESION_PIXELS:\n"
            "            return otsu_mask, leaf\n"
            "    return np.zeros_like(leaf), leaf\n",
            "        if otsu_mask.sum() >= MIN_LESION_PIXELS:\n"
            "            green_frac = float(((h_ >= 35) & (h_ <= 95) & (s_ >= 0.18) & leaf).sum() / max(leaf.sum(), 1))\n"
            "            if otsu_mask.sum() / max(leaf.sum(), 1) > 0.80 and green_frac > 0.90:\n"
            "                return _clean_mask(hsv_lesion), leaf\n"
            "            return otsu_mask, leaf\n"
            "    return _clean_mask(hsv_lesion), leaf\n",
        ).splitlines(True)]
    if '"blast":' in src and "yellowness" not in src.split('"blast":')[1][:200]:
        cell["source"] = [line + "\n" for line in src.replace(
            'grayness = float(((ls < 0.22) & (lv >= 0.2) & (lv <= 0.72)).mean())',
            'grayness = float(((ls < 0.28) & (lv >= 0.22) & (lv <= 0.78)).mean())',
        ).replace(
            '    "blast":            2.6 * grayness + 1.6 * clamp((aspect - 2.2) / 1.8) + 1.0 * ecc\n'
            '                        + 0.5 * brownness,',
            '    "blast":            2.4 * grayness + 1.4 * clamp((aspect - 1.5) / 1.5) + 1.2 * ecc\n'
            '                        + 0.9 * brownness + 0.8 * yellowness\n'
            '                        + 1.0 * clamp(coverage / 0.12),',
        ).splitlines(True)]
dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
