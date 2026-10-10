# NLP作业一：文本分类
> 作业目标：分别使用词袋模型、词嵌入、预训练BERT完成新闻文本分类
> 数据集：NYT新闻数据集（主分类数据集），AG News数据集（仅用于训练Word2Vec）
> 实验设置：NYT数据集分层随机划分，Training/Validation/Test = 80% /10% /10%，`random_state=42`保证可复现
> 评价指标：Accuracy、Macro‑F1，所有模型最终指标均在Test测试集上获取

## 项目结构

```
nlp_hw1/ 
├── data_preprocess.py     # 数据预处理脚本，完成数据集分层划分、分词 
├── run_task1.py           # Task1：Binary BoW + Word‑Frequency BoW 
├── run_task2.py           # Task2：GloVe预训练、Word2Vec‑AG、      Word2Vec‑NYT‑train 
├── run_task3.py           # Task3：BERT‑base‑uncased微调文本分类 
├── data/ 
│   ├── nyt.csv            # 原始 NYT 数据集 
│   └── ag.csv             # 原始 AG News 数据集
├── results/ 
│   ├── task1_result.csv   # Task1 输出实验结果 
│   ├── task2_result.csv   # Task2 输出实验结果 
│   └── task3_result.csv   # Task3 输出实验结果 
└── requirements.txt       # 项目依赖
```

## 环境依赖安装
Python版本建议：3.10 ~ 3.12
```bash
pip install -r requirements.txt
```

## 数据集准备
Task2 GloVe实验需要手动下载 `glove.6B.zip`，解压后取出其中 `glove.6B.100d.txt`，放到项目路径 `./data/glove/`。
> GloVe官方下载地址：http://nlp.stanford.edu/data/glove.6B.zip


## 运行步骤

### 1、数据预处理

完成数据读取、文本清洗、NLTK分词、分层划分80‑10‑10数据集，生成处理后的csv文件。

```
python data_preprocess.py
```

### 2、运行 Task1 Bag of Words

两个实验：Binary Bag of Words、Word Frequency，分类器统一使用LogisticRegression；超参C在validation集调参，测试集输出指标。

输出结果：`./results/task1_result.csv`

```
python run_task1.py
```

### 3、运行 Task2 Word2Vec

三组实验：

1. 预训练 GloVe‑6B‑100d；
2. Word2Vec 在 AG News 数据集训练；
3. Word2Vec 仅使用 NYT 训练集训练。

> 文档向量构造规则：文档内全部有效词向量算术平均，得到 100 维文档向量，送入 LogisticRegression 分类。

输出结果：`./results/task2_result.csv`

```
python run_task2.py
```

### 4、运行 Task3 **Pre-trained Neural Models**

> 参数：`google‑bert/bert‑base‑uncased`，`max_length=64`，训练 epoch=3。 

输出结果：`./results/task3_result.csv`

```
python run_task3.py
```

## 实验全部结果汇总（Test 测试集）

表格

| 方法               | Accuracy | Macro‑F1 |
| ------------------ | -------- | -------- |
| Binary_BoW         | 0.9905   | 0.9764   |
| Word_Frequency_BoW | 0.9896   | 0.9742   |
| GloVe_6B_100d      | 0.9844   | 0.9623   |
| Word2Vec_AGNews    | 0.9783   | 0.9482   |
| Word2Vec_NYTtrain  | 0.9818   | 0.9533   |
| BERT_base_uncased  | 0.9835   | 0.9647   |
