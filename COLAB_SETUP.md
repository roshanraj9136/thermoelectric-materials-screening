# Colab Setup

Use this at the top of the Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/estm.xlsx" /content/
!pip install matminer pymatgen shap -q
```

The main code expects:

```python
ESTM_PATH = os.path.join(OUTPUT_DIR, "estm.xlsx")
```

In Colab, `OUTPUT_DIR` is usually `/content`, so the copied file should be available as:

```text
/content/estm.xlsx
```
