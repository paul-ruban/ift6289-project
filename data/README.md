## Data related files and explanations

There are two datasets **SQUAD** and **SQUAD2.0**.
* `data/raw` - contains datasets with raw JSON files, the splits are already done
* `data/from_hf` - contains configurations to load datasets using `datasets` library

### Usage

```python
from datasets import load_dataset
squad = load_dataset("data/from_hf/squad")
```