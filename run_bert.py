import os
import csv
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# -------------------------- 全局超参 --------------------------
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
NUM_EPOCHS = 3
NUM_LABELS = 3
OUTPUT_DIR = "./bert_output"  # 模型中间输出、checkpoint保存位置
RESULT_PATH = "./results/task3_result.csv"

# 自动判断设备：有cuda用GPU，否则CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备：{device}")


def load_nyt_data(data_dir="./data"):
    """
    读取nyt划分好的数据集
    返回文本列表、数字标签label_id
    BERT不使用之前分词好的tokens，直接传入原始清洗后的文本，由BERT自带tokenizer处理
    """
    df_train = pd.read_csv(f"{data_dir}/nyt_train.csv")
    df_val = pd.read_csv(f"{data_dir}/nyt_val.csv")
    df_test = pd.read_csv(f"{data_dir}/nyt_test.csv")

    train_texts = df_train["text_clean"].tolist()
    train_labels = df_train["label_id"].tolist()

    val_texts = df_val["text_clean"].tolist()
    val_labels = df_val["label_id"].tolist()

    test_texts = df_test["text_clean"].tolist()
    test_labels = df_test["label_id"].tolist()

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


class NytDataset(torch.utils.data.Dataset):
    """
    封装HuggingFace Dataset；把tokenizer编码后的结果包装成pytorch数据集
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # 取单条样本，转tensor
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    """
    Trainer回调计算指标：返回accuracy、macro‑f1
    eval_pred: (predictions, label_ids)
    """
    predictions, true_labels = eval_pred
    pred_labels = predictions.argmax(axis=-1)
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    # 删除旧结果，避免重复
    if os.path.exists(RESULT_PATH):
        os.remove(RESULT_PATH)

    # 1.加载原始数据
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_nyt_data()

    # 2.加载bert‑base‑uncased分词器
    print("\n加载BERT tokenizer：", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3.对文本编码，作业要求max_length=64，truncation截断，padding补全
    print("对文本进行tokenizer编码(max_length=64)...")
    train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)

    # 4.构建Dataset对象
    train_dataset = NytDataset(train_encodings, train_labels)
    val_dataset = NytDataset(val_encodings, val_labels)
    test_dataset = NytDataset(test_encodings, test_labels)

    # 5.加载分类模型，3个输出标签
    print("加载BERT分类模型，num_labels=", NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    # 6.TrainingArguments训练参数设置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,  # 固定3 epoch
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",    # 每训练完一个epoch做一次验证集评估
        save_strategy="epoch",
        load_best_model_at_end=True,    # 训练结束自动加载val集效果最好的checkpoint
        metric_for_best_model="macro_f1",
        logging_dir="./bert_logs",
        logging_steps=20,
        disable_tqdm=False,
    )

    # 7.初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 8.开始训练
    print("\n开始BERT训练，epoch =", NUM_EPOCHS)
    trainer.train()

    # 9.在【test测试集】上评估（作业要求最终指标必须来自test）
    print("\n在NYT Test集上进行最终评估")
    test_metrics = trainer.evaluate(test_dataset)
    test_acc = round(test_metrics["eval_accuracy"], 4)
    test_macrof1 = round(test_metrics["eval_macro_f1"], 4)

    print(f"\nTask3 BERT 最终测试集结果：")
    print(f"Accuracy: {test_acc}, Macro‑F1: {test_macrof1}")

    # 写入csv结果
    res = {
        "method": "BERT_base_uncased",
        "accuracy": test_acc,
        "macro_f1": test_macrof1
    }
    header = ["method", "accuracy", "macro_f1"]
    with open(RESULT_PATH, "w", newline="", encoding="utf‑8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(res)
    print(f"结果保存到 {RESULT_PATH}")


if __name__ == "__main__":
    main()
