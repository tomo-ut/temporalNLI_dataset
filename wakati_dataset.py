from pyknp import Juman
from tqdm import tqdm


def wakati(name):
    with open(name + '.tsv', 'r') as infile:
        lines = infile.read().splitlines()
        lines = [line.split('\t') for line in lines]
    jumanpp = Juman(jumanpp=True)

    for i, line in tqdm(enumerate(lines)):
        premise_result = jumanpp.analysis(line[1])
        lines[i][1] = ' '.join([mrph.midasi for mrph in premise_result.mrph_list()])
        hyp_result = jumanpp.analysis(line[2])
        lines[i][2] = ' '.join([mrph.midasi for mrph in hyp_result.mrph_list()])

    with open(name + '_wakati' + '.tsv', 'w') as out:
        for line in lines:
            out.write('\t'.join(line) + '\n')


if __name__ == '__main__':
    ver = 'ver_1_1'
    split = '82'
    name = f'./dataset/{ver}/train_{split}'
    wakati(name)
