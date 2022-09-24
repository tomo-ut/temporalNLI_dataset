from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import time
import pandas as pd
import pickle
import os
import evaluate


class DataRoberta(Dataset):
    def __init__(self, df):
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.labelid_dict = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}

        self.df = df

        self.base_path = ''
        # Using a pre-trained BERT tokenizer to encode sentences
        self.tokenizer = AutoTokenizer.from_pretrained('nlp-waseda/roberta-large-japanese')
        self.data = None
        self.init_data()

    def init_data(self):
        self.data = self.load_data(self.df)

    def load_data(self, df):
        MAX_LEN = 512
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        premise_list = df['premise'].to_list()
        hypothesis_list = df['hypothesis'].to_list()
        label_list = df['gold_label'].to_list()

        for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
            premise_id = self.tokenizer.encode(premise, add_special_tokens=False)
            hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens=False)
            pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + \
                [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
            premise_len = len(premise_id)
            hypothesis_len = len(hypothesis_id)

            segment_ids = torch.tensor([0] * (premise_len + 2) + [1] *
                                       (hypothesis_len + 1))  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

            token_ids.append(torch.tensor(pair_token_ids))
            seg_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)
            y.append(self.label_dict[label])

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        print(len(dataset))
        return dataset

    def get_data_loaders(self, batch_size=32, shuffle=True):
        data_loader = DataLoader(
            self.data,
            shuffle=shuffle,
            batch_size=batch_size
        )
        return data_loader


def multi_acc(y_pred, y_test):
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc


def train(model, train_loader, val_loader, optimizer):
    total_step = len(train_loader)

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)

            loss, prediction = model(pair_token_ids,
                                     token_type_ids=seg_ids,
                                     attention_mask=mask_ids,
                                     labels=labels).values()

            acc = multi_acc(prediction, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc.item()

        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_acc = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
                optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                labels = y.to(device)

                loss, prediction = model(pair_token_ids,
                                         token_type_ids=seg_ids,
                                         attention_mask=mask_ids,
                                         labels=labels).values()

                acc = multi_acc(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def evaluation(model, test_df):
    test_dataset = DataRoberta(test_df)
    test_loader = test_dataset.get_data_loaders(shuffle=False)
    model.eval()
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    gold_labels = test_df["gold_label"]
    metric.compute()


torch.manual_seed(42)
device = torch.device("mps")

df = pd.read_csv("dataset/train_wakati.tsv", sep="\t")
val_df = df.sample(int(len(df) * 0.1))
train_df = df.drop(val_df.index)
train_dataset = DataRoberta(train_df)
train_loader = train_dataset.get_data_loaders()
val_dataset = DataRoberta(val_df)
val_loader = val_dataset.get_data_loaders()

model = AutoModelForSequenceClassification.from_pretrained('nlp-waseda/roberta-large-japanese', num_labels=3)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)


EPOCHS = 5
train(model, train_loader, val_loader, optimizer)

torch.save(model, 'models/pt_model_roberta.pt')

# 評価
test_df = pd.read_csv("dataset/test_wakati.tsv", sep="\t")

model = torch.load('models/pt_model_roberta.pt')
evaluation(model, test_df)
