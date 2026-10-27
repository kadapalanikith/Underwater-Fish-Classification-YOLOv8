# Training notebooks

These notebooks are Google Colab workflows for downloading the Kaggle fish-detection dataset and training the project's YOLOv8 detector.

| Notebook | What it contains |
|---|---|
| `training/fish_detection_training.ipynb` | Training workflow with no execution counters; some saved cell outputs remain. |
| `training/fish_detection_training_executed.ipynb` | Same workflow with recorded execution metadata and outputs. |

## Kaggle setup

The credential cell reads `KAGGLE_USERNAME` and `KAGGLE_KEY` from the active Python environment and writes `~/.kaggle/kaggle.json`. It does not contain credentials. In a Colab runtime, set the variables interactively before running that cell:

```python
import getpass
import os

os.environ["KAGGLE_USERNAME"] = input("Kaggle username: ")
os.environ["KAGGLE_KEY"] = getpass.getpass("Kaggle API key: ")
```

For a local run, export the variables in the shell or use a secret manager. Do not type literal API credentials into notebook source, outputs, or Git-tracked files. A Kaggle credential found in the earlier notebook history was removed from the current notebook files; revoke and replace that key because historical Git revisions may still contain it.
