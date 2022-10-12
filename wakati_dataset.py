from pyknp import Juman
from tqdm import tqdm


def wakati(ver):
    with open(f'./dataset/{ver}/train.tsv', 'r') as infile:
        lines_train = infile.read().splitlines()
        lines_train = [line.split('\t') for line in lines_train]

    with open(f'./dataset/{ver}/test.tsv', 'r') as infile:
        lines_test = infile.read().splitlines()
        lines_test = [line.split('\t') for line in lines_test]

    jumanpp = Juman(jumanpp=True)

    for i, line in tqdm(enumerate(lines_train)):
        premise_result = jumanpp.analysis(line[1])
        lines_train[i][1] = ' '.join([mrph.midasi for mrph in premise_result.mrph_list()])
        hyp_result = jumanpp.analysis(line[2])
        lines_train[i][2] = ' '.join([mrph.midasi for mrph in hyp_result.mrph_list()])

    for i, line in tqdm(enumerate(lines_test)):
        premise_result = jumanpp.analysis(line[1])
        lines_test[i][1] = ' '.join([mrph.midasi for mrph in premise_result.mrph_list()])
        hyp_result = jumanpp.analysis(line[2])
        lines_test[i][2] = ' '.join([mrph.midasi for mrph in hyp_result.mrph_list()])

    with open(f'dataset/{ver}/train_wakati.tsv', 'w') as out:
        for line in lines_train:
            out.write('\t'.join(line) + '\n')

    with open(f'dataset/{ver}/test_wakati.tsv', 'w') as out:
        for line in lines_test:
            out.write('\t'.join(line) + '\n')


if __name__ == '__main__':
    ver = 'ver_1_1'
    wakati(ver)
