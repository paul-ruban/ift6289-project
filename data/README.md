## Data related files and explanations

There are two datasets **SQUAD** and **SQUAD2.0**.
* `data/raw` - contains datasets with raw JSON files, the splits are already done
* `data/from_hf` - contains configurations to load datasets using `datasets` library from Huggingface

### Download dataset configuraions from Huggingface
```
cd data/from_hf
git-lfs clone https://huggingface.co/datasets/data/squad
git-lfs clone https://huggingface.co/datasets/data/squad_v2
```

### Usage

```python
from datasets import load_dataset
squad = load_dataset("data/from_hf/squad")
```