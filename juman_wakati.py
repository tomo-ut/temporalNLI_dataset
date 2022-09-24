from pyknp import Juman
import pandas as pd
from tqdm import tqdm

# 訓練データ作成
df = pd.read_csv("dataset/train.tsv", sep="\t")
df_test = pd.read_csv("dataset/test.tsv", sep="\t")
premises = list(df['premise'])
hypotheses = list(df['hypothesis'])
x_train = [(premise, hypothesis) for (premise, hypothesis) in zip(premises, hypotheses)]

premises_test = list(df_test['premise'])
hypotheses_test = list(df_test['hypothesis'])
x_test = [(premise, hypothesis) for (premise, hypothesis) in zip(premises_test, hypotheses_test)]

juman = Juman(jumanpp=True)
x_train = [(' '.join([mrph.midasi for mrph in juman.analysis(premise)]),
            ' '.join([mrph.midasi for mrph in juman.analysis(hypothesis)]))
           for (premise, hypothesis) in x_train]
x_test = [(' '.join([mrph.midasi for mrph in juman.analysis(premise)]),
           ' '.join([mrph.midasi for mrph in juman.analysis(hypothesis)]))
          for (premise, hypothesis) in x_test]

for t in ['train', 'test']:
    with open(f"dataset/{t}.tsv", 'r') as infile:
        lines = infile.read().splitlines()
        with open(f"dataset/{t}_wakati.tsv", 'w') as outfile:
            outfile.write(lines[0] + '\n')
            if t == 'train':
                for i in range(len(x_train)):
                    line = lines[i+1].split('\t')
                    line[1] = x_train[i][0]
                    line[2] = x_train[i][1]
                    outfile.write('\t'.join(line) + '\n')
            elif t == 'test':
                for i in range(len(x_test)):
                        line = lines[i+1].split('\t')
                        line[1] = x_test[i][0]
                        line[2] = x_test[i][1]
                        outfile.write('\t'.join(line) + '\n')