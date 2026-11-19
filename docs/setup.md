# Setup and reproduction notes

## Reproducing our exact files

The README's commands need internet access: they pull the data fresh from the city's API and download a 90 MB text encoder from Hugging Face. The city keeps editing the table, so a fresh pull can differ from our snapshot by a few rows: on 2026-11-13 it had 34 more rows, and each hit rate moved by less than 0.01 point. Notebook 01 prints whether your data matches ours.

To regenerate the committed files exactly, use the pinned environment and our snapshot:

```bash
pip install -r requirements-lock.txt      # every package pinned, not only the direct ones
mkdir -p data/raw                         # place inspections_snapshot.parquet here
```

The snapshot is not in the repo, because raw inspector comments contain names and emails. Keep `data/` unpublished. To get the file, open an issue on the GitHub repository. With it in place, notebooks 01 to 04 regenerate every file in `results/` and `preds/` byte for byte, except the M3 files, which come from the optional notebook 05, and the comparison files, which come from the optional notebook 06.

After a run, `git status` lists the four notebooks as modified, because executing them in place updates their saved outputs and timestamps. Run `git status results preds` to check the outputs: it should print nothing.

Two warnings are harmless: "Kernel is running over TCP without encryption" from nbconvert, and "Redirects are currently not supported" from torch.

## Run order

Run every notebook from `notebooks/` with `jupyter nbconvert --to notebook --execute --inplace <name>.ipynb`.

1. `01_data`, `02_baselines`, `03_models`, `04_results`, in that order. This is the README's command.
2. `05_finetune`, optional. It reads the predictions written by 02 to 04 and writes `results/finetune_test.csv`.
3. `04_results` once more, only if you ran 05. Notebook 04 draws `text_value_bar.png` from that file and skips the chart when the file is missing.
4. `06_model_comparison`, optional. It reads the saved predictions of every earlier notebook, and includes M3 when notebook 05 has been run.

## Hardware

- Verified on Apple Silicon (macOS) only. Linux and Windows are untested, and the README's commands assume a POSIX shell.
- M1, the final model, is deterministic. M2 uses text embeddings, which differ in the sixth decimal between devices. That can move M2's hit rate by about 0.3 points.
- Notebook 06 (the wider model comparison) runs on the CPU in about 12 minutes. A second run on our machine gave identical files. It needs our snapshot: with freshly pulled data, early stopping can select a different number of trees, and the notebook then stops instead of scoring a model that was not frozen.
- Notebook 05 fine-tunes on an Apple GPU (MPS) when one is available, which takes about 40 minutes, and on the CPU otherwise, which is much slower. CUDA is not wired in. A GPU run is not guaranteed to give identical numbers twice.

## macOS: LightGBM and OpenMP

On macOS, `import lightgbm` fails with `Library not loaded: @rpath/libomp.dylib` until the OpenMP library is available. Do one of the following after `pip install -r requirements.txt` and before running the notebooks.

Option 1, system-wide (needs Homebrew):

```bash
brew install libomp
```

Option 2, inside the virtual environment only. It points LightGBM at the copy of OpenMP that ships with scikit-learn. The virtual environment must be active, and Apple's command line tools must be installed (`xcode-select --install`).

```bash
LIB=$(python -c "import importlib.util as u, pathlib as p; print(p.Path(u.find_spec('lightgbm').origin).parent/'lib/lib_lightgbm.dylib')")
install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB" && codesign --force --sign - "$LIB"
```

Check with `python -c "import lightgbm"`. Reinstalling LightGBM undoes option 2.

XGBoost, which only notebook 06 imports, fails the same way. Option 1 fixes both libraries. With option 2, repeat the step for XGBoost:

```bash
LIB=$(python -c "import importlib.util as u, pathlib as p; print(p.Path(u.find_spec('xgboost').origin).parent/'lib/libxgboost.dylib')")
install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB" && codesign --force --sign - "$LIB"
```

Check with `python -c "import xgboost"`.
