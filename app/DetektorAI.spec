# PyInstaller: DetektorAI.exe (folder dist\DetektorAI). Budowanie: powershell -File app\build_exe.ps1
# Modele nie sa w archiwum exe - build_exe.ps1 kopiuje je obok (dist\DetektorAI\models, statistical\runs),
# wiec folder jest przenosny, a desktop.py szuka models\ najpierw obok exe, potem wyzej (repo).
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
APP = ROOT / "app"

hidden = (["server", "helpers", "train", "train.predict_system_one", "train.system_one_model",
           "statistical", "statistical.features", "statistical.model",  # model.joblib odwoluje sie do statistical.model
           "sklearn.linear_model", "sklearn.pipeline", "sklearn.preprocessing", "sklearn.tree","webview.platforms.edgechromium", "webview.platforms.winforms"]
          + collect_submodules("transformers.models.distilbert")
          + collect_submodules("transformers.models.deberta_v2"))

a = Analysis(
    [str(APP / "desktop.py")],
    pathex=[str(APP), str(ROOT)],
    datas=[(str(APP / "index.html"), "app")] + collect_data_files("webview") + collect_data_files("transformers", include_py_files=False),
    hiddenimports=hidden,
    excludes=["lightning", "pytorch_lightning", "tensorboard", "matplotlib", "IPython", "jupyter", "notebook",
              "torchvision", "torchaudio", "triton", "pytest", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DetektorAI", console=False,
          icon=str(APP / "icon.ico") if (APP / "icon.ico").exists() else None, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="DetektorAI", upx=False)
