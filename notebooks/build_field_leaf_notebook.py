"""Clone disease simulation notebook for the field streak leaf photo."""
import json
from pathlib import Path

src = Path(__file__).parent / "rice_disease_simulation.ipynb"
dst = Path(__file__).parent / "rice_field_leaf_analysis.ipynb"

nb = json.loads(src.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nb["cells"][0]["source"] = [
    "# Field Rice Leaf Disease Analysis — Interactive Notebook\n",
    "\n",
    "This notebook analyzes a **real field photo** of a rice leaf held in a paddy: "
    "green blade with **narrow yellowish-orange to reddish-brown elongated streaks** "
    "running parallel to the veins.\n",
    "\n",
    "It mirrors the real pipeline in `ml/` (`preprocessor.py`, `lesion_features.py`, "
    "`layer2_disease.py`) and is **fully self-contained** for Steps 1–8 "
    "(numpy, scipy, Pillow, matplotlib only). Step 9 optionally calls the **trained "
    "production model** (`ml.pipeline.run_pipeline`) when you run from the project root.\n",
    "\n",
    "## What the photo shows\n",
    "\n",
    "| Clue | Observation |\n",
    "|------|-------------|\n",
    "| Lesion color | Yellow–orange to reddish-brown |\n",
    "| Lesion shape | Narrow elongated streaks (not round spots) |\n",
    "| Orientation | Parallel to leaf veins |\n",
    "| Setting | Flooded rice paddy |\n",
    "\n",
    "## Pipeline steps\n",
    "\n",
    "| Step | What happens | Graph you'll see |\n",
    "|------|--------------|------------------|\n",
    "| 1. Load | Read the field leaf photo | Original image |\n",
    "| 2. Preprocess | Letterbox-resize to 128x128 | Before vs after |\n",
    "| 3. Color analysis | RGB / HSV — orange–brown pulls hue away from green | Channel maps + histograms |\n",
    "| 4. Segment lesion | Mark off-green streak pixels | Red lesion overlay |\n",
    "| 5. Extract features | High aspect ratio streaks + brown/yellow color | Feature bar charts |\n",
    "| 6. Classify | Rule-based scores for 4 diseases + healthy | Confidence bar chart |\n",
    "| 7. Flow diagram | Photo → diagnosis | Pipeline flowchart |\n",
    "| 8. Dashboard | Final answer + management tips | Summary dashboard |\n",
    "| 9. Production model | Optional: run trained CNN/RF pipeline | Real model probs |\n",
    "\n",
    "**Sample image:** `assets/field_streak_leaf.png`\n",
    "\n",
    "**Disease classes:** `bacterial blight`, `blast`, `brown spot`, `leaf smut`, plus `healthy`.\n",
    "\n",
    "**Compare with:**\n",
    "- `rice_blast_simulation.ipynb` (`assets/blast_leaf.png`)\n",
    "- `rice_disease_simulation.ipynb` (`assets/sample_leaf.png` — brown spot)\n",
    "- `rice_healthy_leaf_simulation.ipynb` (`assets/healthy_leaf.jpg`)\n",
    "\n",
    "> **Note:** Steps 1–8 use a transparent rule-based scorer for teaching. "
    "Step 9 uses the same trained model as the Django upload page when available.\n",
]

src1 = "".join(nb["cells"][1]["source"])
src1 = src1.replace(
    'IMAGE_PATH = Path("assets/sample_leaf.png")',
    'IMAGE_PATH = Path("assets/field_streak_leaf.png")',
)
src1 = src1.replace(
    '    IMAGE_PATH = Path("notebooks/assets/sample_leaf.png")  # if run from repo root',
    '    IMAGE_PATH = Path("notebooks/assets/field_streak_leaf.png")  # if run from repo root',
)
nb["cells"][1]["source"] = [line + "\n" for line in src1.splitlines(True)]

nb["cells"][4]["source"] = [
    "## Step 1 — Load the field leaf photo\n",
    "\n",
    "A hand holds a rice leaf above a flooded paddy. Look for **narrow orange–brown streaks** "
    "along the midsection of the blade — these are the disease signals the pipeline will segment.\n",
]

nb["cells"][8]["source"] = [
    "## Step 3 — Look at the colors (RGB and HSV)\n",
    "\n",
    "Healthy tissue stays in the **green hue band**. Streak lesions pull pixels toward "
    "**yellow / orange / brown** (lower saturation in necrotic parts). Compare Hue and "
    "Saturation histograms against a uniform healthy leaf.\n",
]

nb["cells"][10]["source"] = [
    "## Step 4 — Segment the streak lesions\n",
    "\n",
    "The mask marks off-green pixels inside the leaf: brown/yellow hues and darker necrotic tissue. "
    "On this field leaf you should see **elongated red patches** aligned with the veins "
    "(not round brown-spot disks).\n",
]

nb["cells"][12]["source"] = [
    "## Step 5 — Extract lesion features\n",
    "\n",
    "Elongated streak diseases raise **aspect ratio** and **eccentricity**. Color moments "
    "capture brownness / yellowness. Texture (GLCM, LBP) differs from uniform healthy green.\n",
]

nb["cells"][14]["source"] = [
    "## Step 6 — Score the diseases\n",
    "\n",
    "The teaching scorer boosts **blast** when it sees elongated brown/yellow lesions "
    "(early blast can look like short streaks). **Bacterial blight** scores high on very "
    "long pale streaks; **brown spot** prefers rounder lesions.\n",
]

nb["cells"][18]["source"] = [
    "## Step 8 — Final diagnosis dashboard\n",
    "\n",
    "The teaching scorer gives a transparent answer. Confirm against Step 9 (production model) "
    "when you have the trained weights — the Django app uses that same path.\n",
]

# Soften blast-oriented gray_center patches from blast builder — reuse milder blast scoring
# that favors elongated brown streaks (matches this field photo).
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src_txt = "".join(cell["source"])
    if "def segment_lesion(arr):" in src_txt and "gray_center" not in src_txt:
        cell["source"] = [line + "\n" for line in src_txt.replace(
            "    pale_spot = (s_ < 0.15) & (v_ >= 0.65)\n"
            "    lesion = _clean_mask(leaf & (brown_yellow | dark_necrotic | pale_spot))\n",
            "    pale_spot = (s_ < 0.15) & (v_ >= 0.65)\n"
            "    # Field streaks: catch orange-brown elongated lesions + slight desaturation\n"
            "    orange_streak = ((h_ <= 45) | (h_ >= 150)) & (s_ >= 0.12) & (v_ >= 0.25) & (v_ <= 0.85)\n"
            "    hsv_lesion = leaf & (brown_yellow | dark_necrotic | pale_spot | orange_streak)\n"
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
    if '"blast":' in src_txt and "clamp(coverage / 0.12)" not in src_txt:
        cell["source"] = [line + "\n" for line in src_txt.replace(
            '    "blast":            2.6 * grayness + 1.6 * clamp((aspect - 2.2) / 1.8) + 1.0 * ecc\n'
            '                        + 0.5 * brownness,',
            '    "blast":            1.4 * grayness + 1.8 * clamp((aspect - 1.4) / 1.6) + 1.3 * ecc\n'
            '                        + 1.4 * brownness + 1.0 * yellowness\n'
            '                        + 1.0 * clamp(coverage / 0.10),',
        ).splitlines(True)]

nb["cells"].extend([
    {
        "cell_type": "markdown",
        "id": "prod-pipeline-md",
        "metadata": {},
        "source": [
            "## Step 9 — Production model (same as the web app)\n",
            "\n",
            "Run the trained CNN / RF stack from `ml.pipeline.run_pipeline`. "
            "Start Jupyter from the **project root** (`rice_disease_project/`) so imports resolve.\n",
            "\n",
            "On this field photo the production model typically reports **blast** "
            "(early lesions can look like short orange–brown streaks), with "
            "**bacterial blight** as the runner-up.\n",
        ],
    },
    {
        "cell_type": "code",
        "id": "prod-pipeline-code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# === Step 9: production pipeline (CNN if available, else RF) ================\n",
            "import sys\n",
            "\n",
            "ROOT = Path.cwd()\n",
            "if (ROOT / \"ml\" / \"pipeline.py\").exists():\n",
            "    PROJECT_ROOT = ROOT\n",
            "elif (ROOT.parent / \"ml\" / \"pipeline.py\").exists():\n",
            "    PROJECT_ROOT = ROOT.parent\n",
            "else:\n",
            "    PROJECT_ROOT = ROOT\n",
            "\n",
            "if str(PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            "img_for_model = IMAGE_PATH if IMAGE_PATH.exists() else PROJECT_ROOT / \"notebooks\" / \"assets\" / \"field_streak_leaf.png\"\n",
            "\n",
            "try:\n",
            "    from ml.pipeline import run_pipeline\n",
            "\n",
            "    result = run_pipeline(str(img_for_model.resolve()))\n",
            "    print(\"=\" * 60)\n",
            "    print(\"PRODUCTION MODEL RESULT\")\n",
            "    print(\"=\" * 60)\n",
            "    print(f\"  Engine / health : {result.get('engine', 'rf')} / {result.get('health')}\")\n",
            "    print(f\"  Disease         : {result.get('disease_name') or '(none)'}\")\n",
            "    print(f\"  Confidence      : {result.get('confidence', 0) * 100:.1f}%\")\n",
            "    probs = result.get(\"all_disease_probs\") or {}\n",
            "    if probs:\n",
            "        print(\"  Class probabilities:\")\n",
            "        for name, pct in sorted(probs.items(), key=lambda x: -x[1]):\n",
            "            print(f\"    {name:20s} {pct:5.1f}%\")\n",
            "    if result.get(\"rejection_reason\"):\n",
            "        print(f\"  Rejection       : {result['rejection_reason']}\")\n",
            "    print()\n",
            "    print(\"Recommendation:\")\n",
            "    print(result.get(\"recommendation\", \"\")[:500])\n",
            "\n",
            "    if probs:\n",
            "        order = list(probs.keys())\n",
            "        vals = [probs[k] for k in order]\n",
            "        colors = [CLASS_COLORS.get(k, \"#888888\") for k in order]\n",
            "        fig, ax = plt.subplots(figsize=(8, 3.5))\n",
            "        ax.barh(order[::-1], vals[::-1], color=colors[::-1], edgecolor=\"black\")\n",
            "        ax.set_xlabel(\"Probability (%)\")\n",
            "        ax.set_title(\"Production model — disease probabilities\")\n",
            "        ax.set_xlim(0, 100)\n",
            "        for y, v in enumerate(vals[::-1]):\n",
            "            ax.text(v + 1, y, f\"{v:.1f}%\", va=\"center\", fontsize=9)\n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "except Exception as exc:\n",
            "    print(\"Production pipeline not available in this environment.\")\n",
            "    print(f\"  ({type(exc).__name__}: {exc})\")\n",
            "    print(\"Teaching-scorer diagnosis from Step 6 still applies above.\")\n",
            "    print(f\"  Teaching result: {diagnosis.upper()} ({confidence * 100:.1f}%)\")\n",
        ],
    },
    {
        "cell_type": "markdown",
        "id": "compare-field-md",
        "metadata": {},
        "source": [
            "## Side-by-side comparison (this leaf vs other samples)\n",
            "\n",
            "Compare lesion coverage against the blast, brown-spot, and healthy sample notebooks.\n",
        ],
    },
    {
        "cell_type": "code",
        "id": "compare-field-code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# === Bonus: compare field leaf vs other notebook samples ====================\n",
            "samples = [\n",
            "    (\"This field leaf\", Path(\"assets/field_streak_leaf.png\"), \"#c45c26\"),\n",
            "    (\"Blast sample\", Path(\"assets/blast_leaf.png\"), CLASS_COLORS[\"blast\"]),\n",
            "    (\"Brown spot sample\", Path(\"assets/sample_leaf.png\"), CLASS_COLORS[\"brown spot\"]),\n",
            "    (\"Healthy leaf\", Path(\"assets/healthy_leaf.jpg\"), CLASS_COLORS[\"healthy\"]),\n",
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
            "    fig, axes = plt.subplots(1, len(rows) + 1, figsize=(3.6 * (len(rows) + 1), 4))\n",
            "    for ax, (label, arr, cov, color) in zip(axes[:-1], rows):\n",
            "        ax.imshow(arr)\n",
            "        ax.set_title(f\"{label}\\n{cov * 100:.1f}% lesion\")\n",
            "        ax.axis(\"off\")\n",
            "    axes[-1].bar([r[0] for r in rows], [r[2] * 100 for r in rows],\n",
            "                   color=[r[3] for r in rows], edgecolor=\"black\")\n",
            "    axes[-1].set_ylabel(\"Lesion coverage (%)\")\n",
            "    axes[-1].set_title(\"Coverage comparison\")\n",
            "    axes[-1].tick_params(axis=\"x\", rotation=20, labelsize=8)\n",
            "    fig.suptitle(\"Field leaf vs blast / brown spot / healthy\", y=1.03)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    print(f\"Teaching-scorer diagnosis: {diagnosis.upper()} ({confidence * 100:.1f}%)\")\n",
            "else:\n",
            "    print(\"Comparison images not found in assets/.\")\n",
        ],
    },
])

dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {dst} ({len(nb['cells'])} cells)")
