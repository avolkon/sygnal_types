# -*- coding: utf-8 -*-
import json
from pathlib import Path

nb_path = Path("notebooks/avo_sygnal_types_8.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

cell = r'''# @title §2. Загрузка данных
from pathlib import Path
import io

DATA_NAME = "Run200_Wave_0_1.txt"

# локально: data/ ; Colab: /content/ или /content/data/
candidates = [
    Path("data") / DATA_NAME,
    Path("..") / "data" / DATA_NAME,
    Path("/content") / DATA_NAME,
    Path("/content/data") / DATA_NAME,
]

raw_df = None
for path in candidates:
    if path.exists():
        raw_df = pd.read_csv(path, sep=" ", header=None, skipinitialspace=True)
        print("загружено из:", path.resolve())
        break
    print("не найден:", path)

# fallback: интерактивный upload (Colab), только если файл нигде не найден
if raw_df is None:
    try:
        from google.colab import files
        print(f"Файл не найден в путях — выберите {DATA_NAME} вручную")
        uploaded = files.upload()
        if not uploaded:
            raise FileNotFoundError("Upload отменён, файл не выбран")
        # берём DATA_NAME, иначе первый загруженный файл
        key = DATA_NAME if DATA_NAME in uploaded else next(iter(uploaded))
        raw_df = pd.read_csv(io.BytesIO(uploaded[key]), sep=" ", header=None, skipinitialspace=True)
        print("загружено из upload:", key)
    except ImportError:
        raise FileNotFoundError(
            f"Положите {DATA_NAME} в data/ (локально) или в /content/ (Colab)"
        )

print("исходный размер:", raw_df.shape)
display(raw_df.head())
X = raw_df.drop(columns=DROP_COLS, errors="ignore").to_numpy(dtype=np.float64)
assert X.shape == (N_SAMPLES, 500), X.shape
print(f"матрица волн: {X.shape}, min={X.min():.0f}, max={X.max():.0f}")
print(f"пропуски: {np.isnan(X).sum()}, inf: {np.isinf(X).sum()}")
'''

nb["cells"][5]["source"] = cell.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# verify exact match to expected (normalize newlines)
got = "".join(nb["cells"][5]["source"]).replace("\r\n", "\n").strip()
exp = cell.replace("\r\n", "\n").strip()
print("MATCH" if got == exp else "DIFF")
if got != exp:
    import difflib
    for line in difflib.unified_diff(exp.splitlines(), got.splitlines(), lineterm=""):
        print(line)
