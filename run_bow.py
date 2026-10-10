import os
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_nyt_splits(data_dir="./data"):
    """
    读取预处理划分好的NYT数据集
    返回：训练/验证/测试集的清洗后文本 和 数字标签label_id
    """
    df_train = pd.read_csv(f"{data_dir}/nyt_train.csv")
    df_val = pd.read_csv(f"{data_dir}/nyt_val.csv")
    df_test = pd.read_csv(f"{data_dir}/nyt_test.csv")

    # 使用预处理完成的text_clean字段，不需要再做清洗
    X_train_text = df_train["text_clean"].tolist()
    y_train = df_train["label_id"].tolist()

    X_val_text = df_val["text_clean"].tolist()
    y_val = df_val["label_id"].tolist()

    X_test_text = df_test["text_clean"].tolist()
    y_test = df_test["label_id"].tolist()

    return X_train_text, y_train, X_val_text, y_val, X_test_text, y_test


def evaluate(y_true, y_pred, method_name):
    """
    模型评估函数
    y_true: 真实标签
    y_pred: 预测标签
    return: dict{"method", "accuracy", "macro_f1"}
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    res = {
        "method": method_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4)
    }
    print(f"[{method_name}] Accuracy: {acc:.4f}, Macro‑F1: {macro_f1:.4f}")
    return res


def run_binary_bow(X_train_text, y_train, X_val_text, y_val, X_test_text, y_test):
    """
    实验1：Binary Bag‑of‑Words
    binary=True：词出现为1，不出现为0，忽略出现频次
    """
    vectorizer = CountVectorizer(binary=True)
    X_train = vectorizer.fit_transform(X_train_text)  # 仅训练集构建词表
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    # 在验证集搜索超参C（正则化系数）
    best_c = 1.0
    best_f1 = 0.0
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, max_iter=2000)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        val_f1 = evaluate(y_val, y_val_pred, f"binary_c={c}")["macro_f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_c = c
    print(f"\nBinary BoW best C on val = {best_c}")

    # 使用验证集选出最优C，在test集合得到最终指标
    clf_best = LogisticRegression(C=best_c, max_iter=2000)
    clf_best.fit(X_train, y_train)
    y_test_pred = clf_best.predict(X_test)
    return evaluate(y_test, y_test_pred, "Binary_BoW")


def run_word_freq_bow(X_train_text, y_train, X_val_text, y_val, X_test_text, y_test):
    """
    实验2：Word Frequency Bag‑of‑Words
    binary=False：向量值为单词在文档中的出现次数
    """
    vectorizer = CountVectorizer(binary=False)
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    # 验证集调参C
    best_c = 1.0
    best_f1 = 0.0
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, max_iter=2000)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        val_f1 = evaluate(y_val, y_val_pred, f"freq_c={c}")["macro_f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_c = c
    print(f"\nWord‑Frequency BoW best C on val = {best_c}")

    clf_best = LogisticRegression(C=best_c, max_iter=2000)
    clf_best.fit(X_train, y_train)
    y_test_pred = clf_best.predict(X_test)
    return evaluate(y_test, y_test_pred, "Word_Frequency_BoW")


def main():
    out_path = "./results/task1_result.csv"
    # 如果旧结果文件存在，直接删除，避免重复行
    if os.path.exists(out_path):
        os.remove(out_path)

    # 读取划分好的数据集
    X_train_text, y_train, X_val_text, y_val, X_test_text, y_test = load_nyt_splits()

    # 运行两个实验
    res_binary = run_binary_bow(X_train_text, y_train, X_val_text, y_val, X_test_text, y_test)
    res_freq = run_word_freq_bow(X_train_text, y_train, X_val_text, y_val, X_test_text, y_test)

    # 将test集指标写入csv
    header = ["method", "accuracy", "macro_f1"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(res_binary)
        writer.writerow(res_freq)
    print(f"\nTask1全部实验完成，结果保存在 {out_path}")


if __name__ == "__main__":
    main()
