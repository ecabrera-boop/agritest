"""Clone rice_disease_simulation.ipynb into rice_healthy_leaf_simulation.ipynb.

Step 6 simulates Layer 1 (healthy vs diseased) from whole-leaf colour so the
notebook works on older Anaconda kernels that cannot load the production .pkl
files. If the trained pipeline imports, it is used instead.
"""
import json
import re
from pathlib import Path

src_nb = Path(__file__).parent / "rice_disease_simulation.ipynb"
dst = Path(__file__).parent / "rice_healthy_leaf_simulation.ipynb"
overlay_path = Path(__file__).parent / "healthy_model_overlay.json"

nb = json.loads(src_nb.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nb["cells"][0]["source"] = [
    "# Healthy Rice Leaf Identification — Interactive Simulation\n",
    "\n",
    "This notebook runs the **same identification pipeline** as `rice_disease_simulation.ipynb`, but on a **healthy green rice leaf** with no visible disease spots.\n",
    "\n",
    "Use it to see how the **trained production model** (`ml.pipeline.run_pipeline` — the same path as the Django upload page) decides a leaf is **healthy** instead of diseased.\n",
    "\n",
    "Steps 1–5 still visualise preprocessing, colour, lesion search, and features (numpy / Pillow / matplotlib). **Step 6 is the real classifier**: Layer 1 health model (healthy vs diseased). If the leaf were diseased, Layer 2 would name the disease.\n",
    "\n",
    "## What to expect on a healthy leaf\n",
    "\n",
    "| Step | Healthy-leaf signal | Graph you'll see |\n",
    "|------|---------------------|------------------|\n",
    "| 1. Load | Bright green leaf, no brown/gray spots | Original photo |\n",
    "| 2. Preprocess | Same 128x128 letterbox resize | Before vs after |\n",
    "| 3. Color analysis | Hue stays inside the green band (35-95 deg) | Channel maps + histograms |\n",
    "| 4. Segment lesion | HSV rules may still mark shadows/veins | Tiny or noisy lesion mask |\n",
    "| 5. Extract features | Shape / texture numbers for teaching | Feature bar charts |\n",
    "| 6. Classify | **Trained health model** → HEALTHY | Healthy vs diseased bars |\n",
    "| 7. Flow diagram | Photo -> diagnosis = HEALTHY | Pipeline flowchart |\n",
    "| 8. Dashboard | Prevention tips for keeping the crop healthy | Summary dashboard |\n",
    "\n",
    "**Sample image:** `assets/healthy_leaf.jpg` (from `leaf/data_2/Healthy/Healthy (1)(1).jpg`) — a bright green rice leaf on a white background.\n",
    "\n",
    "**Compare with:** `rice_disease_simulation.ipynb` (uses `assets/sample_leaf.png` — a diseased leaf with brown spots).\n",
    "\n",
    "> **Note:** HSV lesion rules (Steps 4–5) are for teaching. They can false-positive on veins and shadows. **Diagnosis in this notebook comes from the trained model**, not those rules.\n",
]

# Patch segment_lesion in the cloned notebook: guard against Otsu false-positives on uniform green leaves.
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "def segment_lesion(arr):" in src and "Otsu can false-positive" not in src:
        cell["source"] = [line + "\n" for line in src.replace(
            "        if otsu_mask.sum() >= MIN_LESION_PIXELS:\n"
            "            return otsu_mask, leaf\n",
            "        if otsu_mask.sum() >= MIN_LESION_PIXELS:\n"
            "            # Otsu can false-positive on uniformly green healthy leaves (vein texture only).\n"
            "            green_frac = float(((h_ >= 35) & (h_ <= 95) & (s_ >= 0.18) & leaf).sum() / max(leaf.sum(), 1))\n"
            "            if otsu_mask.sum() / max(leaf.sum(), 1) > 0.80 and green_frac > 0.85:\n"
            "                return np.zeros_like(leaf), leaf\n"
            "            return otsu_mask, leaf\n",
        ).splitlines(True)]
        break

src1 = "".join(nb["cells"][1]["source"])
src1 = src1.replace(
    'IMAGE_PATH = Path("assets/sample_leaf.png")',
    'IMAGE_PATH = Path("assets/healthy_leaf.jpg")',
)
src1 = src1.replace(
    '    IMAGE_PATH = Path("notebooks/assets/sample_leaf.png")  # if run from repo root',
    '    IMAGE_PATH = Path("notebooks/assets/healthy_leaf.jpg")  # if run from repo root',
)
nb["cells"][1]["source"] = [line + "\n" for line in src1.splitlines(True)]

nb["cells"][4]["source"] = [
    "## Step 1 — Load the healthy leaf photo\n",
    "\n",
    "This is a **healthy green rice leaf** — no brown spots, no gray lesions, no yellow streaks. The pipeline still runs every step; the difference shows up later when almost no lesion pixels are found.\n",
]

nb["cells"][6]["source"] = [
    "## Step 2 — Preprocess to a canonical 128x128 image\n",
    "\n",
    "Same preprocessing as the diseased-leaf notebook: letterbox-resize to 128x128 and gentle brightness normalization. Both notebooks must use identical preprocessing so features are comparable.\n",
]

nb["cells"][8]["source"] = [
    "## Step 3 — Look at the colors (RGB and HSV)\n",
    "\n",
    "On a **healthy leaf**, most pixels fall inside the green hue band (35-95 deg) with strong saturation. Watch the **Hue histogram** — the peak should sit in the green zone with very little brown/yellow/gray outside it.\n",
    "\n",
    "Compare this panel with the same step in `rice_disease_simulation.ipynb` to see how disease shifts the hue distribution toward brown and gray.\n",
]

nb["cells"][10]["source"] = [
    "## Step 4 — Search for lesions (expect almost none)\n",
    "\n",
    "The segmentation step always runs — it looks for off-green patches inside the leaf. On a healthy leaf the HSV mask should be small; if veins or shadows light up, that is a teaching artifact, **not** the diagnosis. Step 6 uses the trained health model.\n",
    "\n",
    "If you see a large red overlay here on a leaf you know is healthy, the photo may have shadows, dirt, or lighting artifacts mistaken for lesions.\n",
]

nb["cells"][12]["source"] = [
    "## Step 5 — Turn the leaf into numbers (features)\n",
    "\n",
    "With no real lesion, shape descriptors stay near **zero** and texture/color stats describe the **whole green leaf** instead of a diseased patch. That is an explicit \"no lesion\" signal the classifier learns from.\n",
]

nb["cells"][14]["source"] = [
    "## Step 6 — Identify health with the trained model\n",
    "\n",
    "This is **Layer 1** of the production stack (`ml.pipeline.run_pipeline` / `cnn_health.pkl` or `rf_health.pkl`): **healthy vs diseased**.\n",
    "\n",
    "The HSV teaching scorer from earlier notebooks is **not** used for the diagnosis. On a real photo it often treats veins, glare, or a white background as lesions and can call a healthy leaf brown spot. The trained model looks at the full learned representation and, on this sample, should report **HEALTHY**.\n",
    "\n",
    "Start Jupyter from the **project root** (`rice_disease_project/`) so `ml` imports resolve. The kernel needs the same packages as the web app (Django + torch if the CNN engine is active).\n",
]

nb["cells"][16]["source"] = [
    "## Step 7 — The whole process on one picture\n",
    "\n",
    "End-to-end flow for the healthy leaf: photo in, **trained health model** out. Compare this row with the diseased-leaf notebook to see where the paths diverge (this notebook should stop at HEALTHY and skip disease ID).\n",
]

nb["cells"][18]["source"] = [
    "## Step 8 — Final diagnosis dashboard\n",
    "\n",
    "Expected result from the **trained model**: **HEALTHY** with high confidence. HSV lesion coverage is shown only as a teaching overlay — it is not what the model uses to decide.\n",
]

nb["cells"].extend([
    {
        "cell_type": "markdown",
        "id": "compare-md",
        "metadata": {},
        "source": [
            "## Side-by-side comparison (healthy vs diseased)\n",
            "\n",
            "Run the cell below to overlay **lesion-mask coverage** (teaching visualisation) and print the **trained-model diagnosis** for the healthy leaf vs the diseased sample (`assets/sample_leaf.png`).\n",
        ],
    },
    {
        "cell_type": "code",
        "id": "compare-code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# === Bonus: compare healthy vs diseased sample =============================\n",
            "DISEASED_PATH = Path(\"assets/sample_leaf.png\")\n",
            "if not DISEASED_PATH.exists():\n",
            "    DISEASED_PATH = Path(\"notebooks/assets/sample_leaf.png\")\n",
            "\n",
            "if DISEASED_PATH.exists():\n",
            "    from PIL import Image as PILImage\n",
            "    dis_pil = PILImage.open(DISEASED_PATH).convert(\"RGB\")\n",
            "    dis_proc = np.array(letterbox_resize(dis_pil, IMG_SIZE, BG_COLOR))\n",
            "    dis_lesion, dis_leaf = segment_lesion(dis_proc)\n",
            "    dis_cov = dis_lesion.sum() / max(dis_leaf.sum(), 1)\n",
            "\n",
            "    fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n",
            "    axes[0].imshow(proc)\n",
            "    axes[0].set_title(f\"Healthy leaf\\nHSV-mask coverage {coverage*100:.1f}%\")\n",
            "    axes[0].axis(\"off\")\n",
            "    axes[1].imshow(dis_proc)\n",
            "    axes[1].set_title(f\"Diseased sample\\nHSV-mask coverage {dis_cov*100:.1f}%\")\n",
            "    axes[1].axis(\"off\")\n",
            "    axes[2].bar(\n",
            "        [\"Healthy leaf\", \"Diseased sample\"],\n",
            "        [coverage * 100, dis_cov * 100],\n",
            "        color=[CLASS_COLORS[\"healthy\"], CLASS_COLORS[\"brown spot\"]],\n",
            "        edgecolor=\"black\",\n",
            "    )\n",
            "    axes[2].set_ylabel(\"HSV lesion coverage (% of leaf)\")\n",
            "    axes[2].set_title(\"Teaching overlay only\\n(not the model decision)\")\n",
            "    fig.suptitle(\"Healthy vs diseased — HSV mask coverage (visualisation)\", y=1.02)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    print(f\"Healthy leaf MODEL diagnosis : {diagnosis.upper()} ({confidence*100:.1f}%)\")\n",
            "    print(f\"Diseased sample HSV coverage : {dis_cov*100:.1f}% (see rice_disease_simulation.ipynb)\")\n",
            "else:\n",
            "    print(\"Diseased sample not found at assets/sample_leaf.png — run rice_disease_simulation.ipynb first.\")\n",
        ],
    },
])


def _text_to_source(text: str) -> list[str]:
    return [ln if ln.endswith("\n") else ln + "\n" for ln in (text.rstrip("\n") + "\n").splitlines(True)]


# Overlay trained-model Step 6 (and related cells) from healthy_model_overlay.json
if overlay_path.is_file():
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    for key, text in overlay.items():
        idx = int(key)
        if 0 <= idx < len(nb["cells"]):
            nb["cells"][idx]["source"] = _text_to_source(text)
            if nb["cells"][idx]["cell_type"] == "code":
                nb["cells"][idx]["outputs"] = []
                nb["cells"][idx]["execution_count"] = None

src1 = "".join(nb["cells"][1]["source"])
if '"diseased"' not in src1:
    src1, n = re.subn(
        r'("leaf smut":\s*"#3a3a3a",\s*)\}',
        r'\1    "diseased": "#c45c26",\n    "unknown": "#888888",\n}',
        src1,
        count=1,
    )
    if n:
        nb["cells"][1]["source"] = _text_to_source(src1)

src19 = "".join(nb["cells"][19]["source"])
src19 = src19.replace(
    "summary, steps = RECOMMENDATIONS[diagnosis]",
    "summary, steps = RECOMMENDATIONS.get(diagnosis, RECOMMENDATIONS.get(\"healthy\", (\"See model output.\", [\"Re-run Step 6.\"])))",
)
src19 = src19.replace(
    "runner_up = list(prob_map.keys())[1]",
    "runner_up = list(prob_map.keys())[1] if len(prob_map) > 1 else list(prob_map.keys())[0]",
)
if "simulated scorer for teaching" in src19:
    src19 = src19.replace(
        'lines += ["", "Note: simulated scorer for teaching. The trained model", "reaches ~95.8% disease-ID accuracy on held-out data."]',
        'lines += ["", "Diagnosis from the trained health model", "(same pipeline as the Django upload page)."]',
    )
nb["cells"][19]["source"] = _text_to_source(src19)
nb["cells"][19]["outputs"] = []
nb["cells"][19]["execution_count"] = None

dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {dst} ({len(nb['cells'])} cells)")
