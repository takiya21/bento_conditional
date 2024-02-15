

## Visual impression estimation system considering attribute information

The data that support the findings of this article are not publicly available due to privacy. They can be requested from the author at taki@cv.info.gifu-u.ac.jp.

## Install
```
conda create -n [environment_name] python -y
conda activate [environment_name]
pip install -r requirements.txt
```

## Environment
Ordinal Implementation setting is below

|  Device |  Detail  |
|  --  |  --  |
|  GPU  |  RTX3090  |
|  CPU  |  Intel (R) Core(TM) i7-10750H  |
|  CUDA  |  11.6  |
|  Python  |  3.8  |

### How to use
```python closs_valid_train.py --batch_size $batch_size --in_w $w --lr $lr --weight_decay $weight_decay --optim $optim --seed 0 --conditional_flg 0 --bottle $bottle_size```

### About the File


| File | about |
| - | - |
| closs_valid_train.py | Files to train and test |
| log.py | File to write log |
| read_dataset.py | File to read dataset |