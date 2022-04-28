# ift6289-project (Model Compression for Extractive QnA)

## Repo Structure

* `config` - configuration files for training
* `data` - contains SQUAD datasets
* `scripts` - scrips to run training experiments (examples are included at the top of the scripts)
* `src` - source code of the project

To install the project, run:
```
cd ift6289-project
source installation.sh
```

To run a training experiment, run:
```
cd scripts
python run.py -c ../config/config.json
```

To quantize the model, run:
```
cd scripts
python quantize.py --c ../config/quant.json
```