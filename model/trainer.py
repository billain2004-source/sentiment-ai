import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset


LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


def train(data_path: str, output_dir: str = 'model/fine_tuned', epochs: int = 3):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].str.strip().str.lower()
    df = df[df['label'].isin(LABEL2ID)]
    df['label_id'] = df['label'].map(LABEL2ID)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize(texts):
        return tokenizer(list(texts), truncation=True, padding=True, max_length=128)

    train_enc = tokenize(train_df['text'])
    test_enc = tokenize(test_df['text'])

    train_dataset = SentimentDataset(train_enc, list(train_df['label_id']))
    test_dataset = SentimentDataset(test_enc, list(test_df['label_id']))

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_steps=10,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Training on {len(train_dataset)} samples, evaluating on {len(test_dataset)} samples...")
    trainer.train()

    results = trainer.evaluate()
    acc = results.get('eval_accuracy', 0) * 100
    print(f"\nTest Accuracy: {acc:.2f}%")
    print("\nDetailed Report:")
    preds = np.argmax(trainer.predict(test_dataset).predictions, axis=-1)
    print(classification_report(test_df['label_id'], preds, target_names=list(LABEL2ID.keys())))

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    return acc
