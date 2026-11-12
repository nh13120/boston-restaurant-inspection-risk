# macOS: LightGBM and OpenMP

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
