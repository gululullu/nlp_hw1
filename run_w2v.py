import os
import csv
import ast
import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_data(data_dir="./data"):
    """
    读取预处理数据集
    return: train_tokens,y_train, val_tokens,y_val, test_tokens,y_test, ag_tokens
    """
    df_train = pd.read_csv(f"{data_dir}/nyt_train.csv")
    df_val = pd.read_csv(f"{data_dir}/nyt_val.csv")
    df_test = pd.read_csv(f"{data_dir}/nyt_test.csv")
    df_ag = pd.read_csv(f"{data_dir}/ag_processed.csv")

    # 还原tokens字段：csv保存列表后变成字符串，需要解析回list
    df_train["tokens"] = df_train["tokens"].apply(ast.literal_eval)
    df_val["tokens"] = df_val["tokens"].apply(ast.literal_eval)
    df_test["tokens"] = df_test["tokens"].apply(ast.literal_eval)
    df_ag["tokens"] = df_ag["tokens"].apply(ast.literal_eval)

    train_tokens = df_train["tokens"].tolist()
    y_train = df_train["label_id"].tolist()

    val_tokens = df_val["tokens"].tolist()
    y_val = df_val["label_id"].tolist()

    test_tokens = df_test["tokens"].tolist()
    y_test = df_test["label_id"].tolist()

    ag_tokens = df_ag["tokens"].tolist()

    return train_tokens, y_train, val_tokens, y_val, test_tokens, y_test, ag_tokens


def evaluate(y_true, y_pred, method_name):
    """评估函数，计算Accuracy与Macro‑F1"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    res = {
        "method": method_name,
        "accuracy": round(acc,4),
        "macro_f1": round(macro_f1,4)
    }
    print(f"[{method_name}] Accuracy:{acc:.4f}, Macro‑F1:{macro_f1:.4f}")
    return res


def get_doc_embedding(tokens_list, wv, emb_dim=100):
    """
    文档向量 = 文档内全部有效词向量算术平均
    :param tokens_list: list[list[str]]，每篇文档的token序列
    :param wv: 词向量字典 {word: np.ndarray(100,)}
    :param emb_dim: 词向量维度，固定100
    :return: np.ndarray shape=(样本数,100)，全部文档向量矩阵
    处理边界：文档所有词都不在词表中，则输出全0向量
    """
    doc_embs = []
    for tokens in tokens_list:
        vecs = []
        for w in tokens:
            if w in wv:
                vecs.append(wv[w])
        if len(vecs) > 0:
            # e(d) = 1/n * sum(e(w_i))
            doc_vec = np.mean(np.array(vecs), axis=0)
        else:
            # OOV全部，填充零向量
            doc_vec = np.zeros(emb_dim)
        doc_embs.append(doc_vec)
    return np.array(doc_embs)


def run_glove(train_tokens, y_train, val_tokens, y_val, test_tokens, y_test):
    """
    实验1：加载本地预训练GloVe glove.6B.100d.txt
    本地文件路径：./data/glove/glove.6B.100d.txt
    """
    glove_path = "./data/glove/glove.6B.100d.txt"
    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            glove_dict[word] = vec
    print(f"GloVe加载完成，词表大小:{len(glove_dict)}")

    X_train = get_doc_embedding(train_tokens, glove_dict)
    X_val = get_doc_embedding(val_tokens, glove_dict)
    X_test = get_doc_embedding(test_tokens, glove_dict)

    best_c, best_f1 =1.0,0.0
    for c in [0.01,0.1,1.0,10.0]:
        clf = LogisticRegression(C=c, max_iter=3000)
        clf.fit(X_train, y_train)
        yp = clf.predict(X_val)
        f1_val = evaluate(y_val, yp, f"glove_c={c}")["macro_f1"]
        if f1_val>best_f1:
            best_f1 = f1_val
            best_c = c
    print(f"GloVe best C={best_c}")

    clf_best = LogisticRegression(C=best_c, max_iter=3000)
    clf_best.fit(X_train, y_train)
    y_test_pred = clf_best.predict(X_test)
    return evaluate(y_test, y_test_pred, "GloVe_6B_100d")


def train_w2v_ag(train_tokens, y_train, val_tokens, y_val, test_tokens, y_test, ag_tokens):
    """
    实验2：在AG News文本训练Word2Vec，vector_size=100
    使用gensim.Word2Vec；窗口window=5，min_count=2过滤低频词
    """
    model = Word2Vec(
        sentences=ag_tokens,
        vector_size=100,    
        window=5,
        min_count=2,        # 出现少于2次的词不构建词向量
        workers=4
    )
    wv = model.wv  # 获取词向量对象

    # 生成训练、验证、测试集文档向量
    X_train = get_doc_embedding(train_tokens, wv)
    X_val = get_doc_embedding(val_tokens, wv)
    X_test = get_doc_embedding(test_tokens, wv)

    # val调参正则系数C
    best_c, best_f1 = 1.0, 0.0
    for c in [0.01,0.1,1.0,10.0]:
        clf = LogisticRegression(C=c, max_iter=3000)
        clf.fit(X_train, y_train)
        yp = clf.predict(X_val)
        f1_val = evaluate(y_val, yp, f"w2v_ag_c={c}")["macro_f1"]
        if f1_val>best_f1:
            best_f1 = f1_val
            best_c = c
    print(f"W2V‑AG best C={best_c}")

    # test集评估最终指标
    clf_best = LogisticRegression(C=best_c, max_iter=3000)
    clf_best.fit(X_train, y_train)
    y_test_pred = clf_best.predict(X_test)
    return evaluate(y_test, y_test_pred, "Word2Vec_AGNews")


def train_w2v_nyt(train_tokens, y_train, val_tokens, y_val, test_tokens, y_test):
    """
    实验3：Word2Vec训练于 NYT训练集
    """
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )
    wv = model.wv

    X_train = get_doc_embedding(train_tokens, wv)
    X_val = get_doc_embedding(val_tokens, wv)
    X_test = get_doc_embedding(test_tokens, wv)

    best_c, best_f1 =1.0,0.0
    for c in [0.01,0.1,1.0,10.0]:
        clf = LogisticRegression(C=c, max_iter=3000)
        clf.fit(X_train, y_train)
        yp = clf.predict(X_val)
        f1_val = evaluate(y_val, yp, f"w2v_nyt_c={c}")["macro_f1"]
        if f1_val>best_f1:
            best_f1 = f1_val
            best_c = c
    print(f"W2V‑NYT‑train best C={best_c}")

    clf_best = LogisticRegression(C=best_c, max_iter=3000)
    clf_best.fit(X_train, y_train)
    y_test_pred = clf_best.predict(X_test)
    return evaluate(y_test, y_test_pred, "Word2Vec_NYTtrain")


def main():
    out_path = "./results/task2_result.csv"
    # 删除旧结果，避免重复行
    if os.path.exists(out_path):
        os.remove(out_path)

    train_tokens, y_train, val_tokens, y_val, test_tokens, y_test, ag_tokens = load_data()

    # 依次执行三组实验
    res_glove = run_glove(train_tokens, y_train, val_tokens, y_val, test_tokens, y_test)
    res_ag = train_w2v_ag(train_tokens, y_train, val_tokens, y_val, test_tokens, y_test, ag_tokens)
    res_nyt = train_w2v_nyt(train_tokens, y_train, val_tokens, y_val, test_tokens, y_test)

    # 写入csv结果
    header = ["method", "accuracy", "macro_f1"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(res_glove)
        writer.writerow(res_ag)
        writer.writerow(res_nyt)
    print(f"\nTask2全部实验完成，结果保存在 {out_path}")


if __name__ == "__main__":
    main()
