import evaluate
from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from ast import literal_eval
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from transformers import RobertaConfig, TFRobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from pyknp import Juman
import torch
import tensorflow as tf
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm


class Vocab:
    # 正解ラベルの設定（今回はcontradiction, entailment, neutralの３値を設定）
    def __init__(self):
        self.token_index = {label: i for i, label in enumerate(["contradiction", "entailment", "neutral"])}
        self.index_token = {v: k for k, v in self.token_index.items()}

    def encode(self, labels):
        label_ids = [self.token_index.get(label) for label in labels]
        return label_ids

    def decode(self, label_ids):
        labels = [self.index_token.get(label_id) for label_id in label_ids]
        return labels

    @property
    def size(self):
        return len(self.token_index)

    def save(self, file_path):
        with open(file_path, 'w') as f:
            config = {
                'token_index': self.token_index,
                'index_token': self.index_token
            }
            f.write(json.dumps(config))

    @classmethod
    def load(cls, file_path):
        with open(file_path) as f:
            config = json.load(f)
            vocab = cls()
            vocab.token_index = config.token_index
            vocab.index_token = config.index_token
        return vocab


def convert_examples_to_features(x, y, vocab, max_seq_length, tokenizer):
    features = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'label_ids': np.asarray(vocab.encode(y))
    }
    for pairs in x:
        tokens = [tokenizer.cls_token]
        token_type_ids = []
        for i, sent in enumerate(pairs):
            word_tokens = tokenizer.tokenize(sent)
            tokens.extend(word_tokens)
            tokens += [tokenizer.sep_token]
            len_sent = len(word_tokens) + 1
            token_type_ids += [i] * len_sent

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        features['input_ids'].append(input_ids)
        features['attention_mask'].append(attention_mask)
        features['token_type_ids'].append(token_type_ids)

    for name in ['input_ids', 'attention_mask', 'token_type_ids']:
        features[name] = pad_sequences(features[name], padding='post', maxlen=max_seq_length)

    x = [features['input_ids'], features['attention_mask'], features['token_type_ids']]
    y = features['label_ids']
    return x, y


def build_model(pretrained_model_name_or_path, num_labels):
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=num_labels
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        config=config
    )
    return model


def evaluate(model):
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()
    with open(pred_detail, 'w') as outfile:
        outfile.write(f'{hypr_dict}\n')
        outfile.write('premise\thypothesis\ttrue\tpred\tc/w\n')
        for i in range(len(x_test)):
            c_w = (y_true[i] == y_pred[i])
            outfile.write(f'{x_test[i][0]}\t{x_test[i][1]}\t{y_true[i]}\t{y_pred[i]}\t{c_w}\n')
            if not c_w:
                false_list.append(f'{x_test[i][0]}\t{x_test[i][1]}\t{y_true[i]}\t{y_pred[i]}\t{c_w}\n')
    with open(pred_detail.replace('.txt', '') + '_false.txt', 'w') as outfile:
        outfile.write(f'{hypr_dict}\n')
        outfile.write('premise\thypothesis\ttrue\tpred\tc/w\n')
        for false in false_list:
            outfile.write(false)
    return classification_report(y_true, y_pred, digits=4, output_dict=True)


with open('./hyper_parameter.txt', 'r') as infile:
    hypr_dict = literal_eval(infile.read())


# ハイパーパラメータの設定
batch_size = 10
epochs = 50
maxlen = 250
model_path = 'models/'
train_data_name = hypr_dict['train_data_name']
test_data_name = hypr_dict['test_data_name']
test_size = hypr_dict['test_size']
seed = hypr_dict['seed']
model_name = hypr_dict['model_name']
config_name = hypr_dict['config_name']
pred_detail = hypr_dict['pred_detail']

# トークナイザ
target_vocab = Vocab()
pretrained_model_name_or_path = 'nlp-waseda/roberta-large-japanese'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

# モデルの構築
model = build_model(pretrained_model_name_or_path, target_vocab.size)

# 訓練データ作成
df = pd.read_csv("dataset/" + train_data_name, sep="\t")
df_test = pd.read_csv("dataset/" + test_data_name, sep="\t")
premises = list(df['premise'])
hypotheses = list(df['hypothesis'])
x = [(premise, hypothesis) for (premise, hypothesis) in zip(premises, hypotheses)]
y = list(df['gold_label'])

# 全データをファインチューニングに使う場合
if hypr_dict['train_test_split']:
    x_train, x_test_t, y_train, y_test_t = train_test_split(x, y, test_size=test_size, random_state=seed)
else:
    x_train = x
    y_train = y

# テストデータ作成
if test_data_name == train_data_name:
    x_test = x_test_t
    y_test = y_test_t
else:
    premises_test = list(df_test['premise'])
    hypotheses_test = list(df_test['hypothesis'])
    x_test = [(premise, hypothesis) for (premise, hypothesis) in zip(premises_test, hypotheses_test)]
    y_test = list(df_test['gold_label'])

juman = Juman(jumanpp=True)
# for i, (premise, hypothesis) in tqdm(enumerate(x_train)):
#     x_train[i] = (' '.join([mrph.midasi for mrph in juman.analysis(premise)]),
#                   ' '.join([mrph.midasi for mrph in juman.analysis(hypothesis)]))
x_train = [(' '.join([mrph.midasi for mrph in juman.analysis(premise)]),
            ' '.join([mrph.midasi for mrph in juman.analysis(hypothesis)]))
           for (premise, hypothesis) in x_train]
x_test = [(' '.join([mrph.midasi for mrph in juman.analysis(premise)]),
           ' '.join([mrph.midasi for mrph in juman.analysis(hypothesis)]))
          for (premise, hypothesis) in x_test]


train_dataloader = DataLoader(x_train, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(x_test, batch_size=8)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("mps")
model.to(device)


progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

features_train, labels_train = convert_examples_to_features(
    x_train,
    y_train,
    target_vocab,
    max_seq_length=maxlen,
    tokenizer=tokenizer
)

# モデルのファインチューニング
model.fit(x=features_train,
          y=labels_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          callbacks=callbacks)
model.save_pretrained(model_path)

os.rename('models/pytorch_model.bin', 'models/' + model_name)
os.rename('models/pytorch_model.json', 'models/' + config_name)

# モデルの読み込み（モデルがある場合に使用）
config = AutoConfig.from_json_file('models/' + config_name)
model = AutoModelForSequenceClassification.from_pretrained('models/' + model_name, config=config)

# 任意のデータでテスト
features_test, labels_test = convert_examples_to_features(
    x_test, y_test, target_vocab, max_seq_length=maxlen, tokenizer=tokenizer)

# 混同行列の作成, ラベルの予測
classify_result = evaluate(model, target_vocab, features_test, labels_test)

with open('results/result.txt', 'a') as outfile:
    ind = ['precision', 'recall', 'f1-score', 'support']
    result1 = ['', model_name, config_name, train_data_name, 1 - test_size, test_data_name, seed]
    result2 = [classify_result['accuracy'], classify_result['macro avg']['support']]
    result3 = [classify_result['macro avg'][name]for name in ind[:3]]
    result4 = [classify_result['weighted avg'][name]for name in ind[:3]]
    result5 = [classify_result['entailment'][name]for name in ind]
    result6 = [classify_result['contradiction'][name]for name in ind]
    result7 = [classify_result['neutral'][name]for name in ind]
    result = result1 + result2 + result3 + result4 + result5 + result6 + result7
    result = [str(r) for r in result]
    r_str = "\t".join(result)
    outfile.write(f'{r_str}\n')
