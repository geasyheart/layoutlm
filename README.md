### 说明
使用layoutLM在FUNSD数据集上进行微调，分成两个实现方式：
1. main分支，采用layoutLM实现
2. layoutXLM分支，采用layoutXLM实现

更多请参考[here](https://geasyheart.github.io/2022/12/12/layoutLM%E5%BE%AE%E8%B0%83FUNSD%E6%95%B0%E6%8D%AE%E9%9B%86/)。

### 运行
1. mkdir data && wget https://guillaumejaume.github.io/FUNSD/dataset.zip && unzip dataset.zip
2. python preprocess.py
3. python train_funsd.py