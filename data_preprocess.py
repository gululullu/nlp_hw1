import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import word_tokenize


def clean_text(raw_text: str) -> str:
    """基础文本清洗：去掉多余符号、换行、转小写"""
    if pd.isna(raw_text):
        return ""
    text = str(raw_text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()


def tokenize_text(text: str):
    """严格使用作业要求 nltk word_tokenize"""
    return word_tokenize(text)


def main():
    nyt_path = "./data/nyt.csv"
    ag_path = "./data/ag.csv"

    df_nyt = pd.read_csv(nyt_path)
    print(f"原始nyt数据集总条数：{len(df_nyt)}")
    print("nyt标签分布：")
    print(df_nyt["label"].value_counts())

    df_ag = pd.read_csv(ag_path)
    print(f"\n原始ag数据集总条数：{len(df_ag)}")

    # 分层划分 80/10/10
    df_train, df_temp = train_test_split(
        df_nyt,
        test_size=0.2,
        stratify=df_nyt["label"],
        random_state=42,
        shuffle=True
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp["label"],
        random_state=42,
        shuffle=True
    )

    print(f"\ntrain:{len(df_train)} | val:{len(df_val)} | test:{len(df_test)}")
    print("train集标签分布：")
    print(df_train["label"].value_counts())

    label_mapping = {
        "sports": 0,
        "business": 1,
        "politics": 2
    }

    def process_df(df):
        df = df.copy()
        df["text_clean"] = df["text"].apply(clean_text)
        df["tokens"] = df["text_clean"].apply(tokenize_text)
        df["label_id"] = df["label"].map(label_mapping)
        return df

    df_train = process_df(df_train)
    df_val = process_df(df_val)
    df_test = process_df(df_test)

    df_ag = df_ag.copy()
    df_ag["text_clean"] = df_ag["text"].apply(clean_text)
    df_ag["tokens"] = df_ag["text_clean"].apply(tokenize_text)

    df_train.to_csv("./data/nyt_train.csv", index=False)
    df_val.to_csv("./data/nyt_val.csv", index=False)
    df_test.to_csv("./data/nyt_test.csv", index=False)
    df_ag.to_csv("./data/ag_processed.csv", index=False)

    print("\n数据预处理完成，文件输出到 ./data/")
    print("输出文件列表：")
    print("  nyt_train.csv  训练集")
    print("  nyt_val.csv    验证集")
    print("  nyt_test.csv   测试集")
    print("  ag_processed.csv 处理好的AG新闻")


if __name__ == "__main__":
    main()
