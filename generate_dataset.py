import random
from copy import deepcopy
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm


def assign_word(element):
    if 'agent' in element:
        return(random.choice(agent_list))
    if 'np' in element:
        return(random.choice(np_list))
    elif 'tp' in element:
        # return(random.choice([str(random.randint(0, 23)) + '時', str(random.randint(1, 12)) + '月']))
        return(random.choice([str(random.randint(1, 12)) + '月']))
    elif 'interval' in element:
        return(str(random.randint(1, 5)) + random.choice(['時間', '日間', '年間']))
    elif 'place' in element:
        return(random.choice(place_list))
    else:
        return(element)


def assign_verb(elements, ind):
    c_elements = deepcopy(elements)
    for i in range(len(elements)):
        if i != ind and 'vp' in elements[i]:
            if 'imp' in elements[i] or 'conj' in elements[i]:
                c_elements[i] = 'し'
            else:
                c_elements[i] = 'する'
    mask_text = ''.join([tokenizer.mask_token if 'vp' in e else e for e in c_elements])
    return fill_mask(mask_text)[0]['token_str'].replace(' ', '')


random.seed(0)

with open('./template.tsv', 'r') as infile:
    templates = infile.read().splitlines()[1:]
    templates = [template.split('\t') for template in templates]

with open('./place_list.txt', 'r') as infile:
    place_list = infile.read().splitlines()

with open('./np_list.txt', 'r') as infile:
    np_list = infile.read().splitlines()

with open('./agent_list.txt', 'r') as infile:
    agent_list = infile.read().splitlines()

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

texts = []

for template in tqdm(templates):
    for _ in range(1):
        memo = {}
        (prem, hyp) = template
        prem_elements = prem.split(' ')
        hyp_elements = hyp.split(' ')
        for elements in [prem_elements, hyp_elements]:
            for i, element in enumerate(elements):
                if element in memo:
                    elements[i] = memo[element]
                elif 'vp' not in element:
                    elements[i] = assign_word(element)
                    memo[element] = elements[i]
            for i, element in enumerate(elements):
                if element not in (list(memo.keys()) + list(memo.values())):
                    elements[i] = assign_verb(elements, i)
        texts.append([''.join(prem_elements), ''.join(hyp_elements)])

with open('dataset.tsv', 'w') as outfile:
    outfile.write('premise\thypothesis\n')
    for text in texts:
        outfile.write(f'{text[0]}\t{text[1]}\n')
