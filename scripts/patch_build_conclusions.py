#!/usr/bin/env python
"""One-off: replace outcome() cells with md(_conc[...]) in build script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "scripts" / "build_avo_sygnal_types_2.py"
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
out: list[str] = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip().startswith("outcome("):
        while i < len(lines):
            if '"""),' in lines[i]:
                i += 1
                break
            i += 1
        continue
    out.append(line)
    i += 1
text = "".join(out)

replacements = [
    (
        'print(f"пропуски: {np.isnan(X).sum()}, inf: {np.isinf(X).sum()}")"""),',
        'print(f"пропуски: {np.isnan(X).sum()}, inf: {np.isinf(X).sum()}")"""),\n    md(_conc["AFTER_LOAD"]),',
    ),
    (
        'print(pd.Series(row_max).describe().round(2))"""),',
        'print(pd.Series(row_max).describe().round(2))"""),\n    md(_conc["AFTER_EDA"]),',
    ),
    (
        'print(feat_df.describe().round(3))"""),',
        'print(feat_df.describe().round(3))"""),\n    md(_conc["AFTER_FEATURES"]),',
    ),
    (
        'plt.title("Корреляции признаков")\nplt.tight_layout()\nplt.show()"""),',
        'plt.title("Корреляции признаков")\nplt.tight_layout()\nplt.show()"""),\n    md(_conc["AFTER_PSD"]),',
    ),
    (
        'plt.ylabel("PC2")\nplt.show()"""),',
        'plt.ylabel("PC2")\nplt.title("PC1 vs PC2 (цвет = KMeans k=3)")\nplt.tight_layout()\nplt.show()"""),\n    md(_conc["AFTER_PCA"]),',
    ),
    (
        'plt.title("ICA: компоненты 1 и 2")\nplt.tight_layout()\nplt.show()"""),',
        'plt.title("ICA: компоненты 1 и 2")\nplt.tight_layout()\nplt.show()"""),\n    md(_conc["AFTER_ICA"]),',
    ),
    (
        'print(f"лучший silhouette: {best_sil}")"""),',
        'print(f"лучший silhouette: {best_sil}")"""),\n    md(_conc["AFTER_MODELS"]),',
    ),
    (
        'display(tune_df.round(4))"""),',
        'display(tune_df.round(4))"""),\n    md(_conc["AFTER_TUNE"]),',
    ),
    (
        'submission.head()"""),',
        'submission.head()"""),\n    md(_conc["FINAL"]),',
    ),
    (
        '    md("""## Kaggle\n\n1. Выполните §9 — появится `submission.csv`.\n2. Загрузите его на https://www.kaggle.com/competitions/signal-types-classification\n3. **Вставьте скриншот лидерборда в ячейку ниже** после отправки."""),\n    md("""_Скриншот Kaggle: вставьте изображение сюда после отправки submission._"""),',
        '    md(_conc["KAGGLE"]),\n    md(_conc["KAGGLE_PLACEHOLDER"]),',
    ),
]

for old, new in replacements:
    if old not in text:
        print("MISSING:", old[:80].replace("\n", "\\n"))
    else:
        text = text.replace(old, new, 1)

# Fix PCA viz if title was missing - check if we need alternate pattern
if 'md(_conc["AFTER_PCA"])' not in text:
    alt_old = 'plt.tight_layout()\nplt.show()"""),\n    code("""# @title §6. FastICA'
    if alt_old in text:
        text = text.replace(
            alt_old,
            'plt.title("PC1 vs PC2 (цвет = psd)")\nplt.tight_layout()\nplt.show()"""),\n    md(_conc["AFTER_PCA"]),\n    code("""# @title §6. FastICA',
            1,
        )
    else:
        print("AFTER_PCA insert failed")

p.write_text(text, encoding="utf-8")
print("outcome left:", text.count("outcome("))
print("md inserts:", sum(text.count(f'md(_conc["{k}"])') for k in [
    "AFTER_LOAD", "AFTER_EDA", "AFTER_FEATURES", "AFTER_PSD", "AFTER_PCA",
    "AFTER_ICA", "AFTER_MODELS", "AFTER_TUNE", "FINAL", "KAGGLE", "KAGGLE_PLACEHOLDER",
]))
