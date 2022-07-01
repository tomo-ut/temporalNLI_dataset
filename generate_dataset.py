import random
from copy import deepcopy
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from janome.tokenizer import Tokenizer
from kotodama import kotodama
from pyknp import Juman
import torch
import numpy as np
import re


def assign_word_wo_vp(element):
    if 'agent' in element:
        return(random.choice(proper_agent_list))
    if 'np' in element:
        return(random.choice(np_list))
    elif 'tp' in element:
        # return(random.choice([str(random.randint(0, 23)) + '時', str(random.randint(1, 12)) + '月']))
        return(random.choice([str(random.randint(1, 12)) + '月']))
    elif 'interval' in element:
        return(str(random.randint(1, 5)) + random.choice(['時間', '日間', '年間']))
    elif 'place' in element:
        return(random.choice(place_list))
    elif 'time_unit' in element:
        return(random.choice(['月', '年', '日']))
    else:
        return(element)


def assign_word_wo_np(element):
    if 'agent' in element:
        return(random.choice(proper_agent_list))
    elif 'vp_zi' in element:
        if 'conj' in element:
            return(random.choice(vp_zi_list)[1])
        elif 'imp' in element:
            return(random.choice(vp_zi_list)[2])
        elif 'past' in element:
            return kotodama.transformVerb(random.choice(vp_zi_list)[0], {"過去"})
        elif 'prog' in element:
            return kotodama.transformVerb(random.choice(vp_zi_list)[0], {"て"})
        elif 'coni' in element:
            return kotodama.transformVerb(random.choice(vp_zi_list)[0], {"です・ます"})[:-2]
        else:
            return(random.choice(vp_zi_list)[0])
    elif 'vp_ta' in element:
        if 'conj' in element:
            return(random.choice(vp_ta_list)[1])
        elif 'imp' in element:
            return(random.choice(vp_ta_list)[2])
        elif 'past' in element:
            return kotodama.transformVerb(random.choice(vp_ta_list)[0], {"過去"})
        elif 'prog' in element:
            return kotodama.transformVerb(random.choice(vp_ta_list)[0], {"て"})
        elif 'coni' in element:
            return kotodama.transformVerb(random.choice(vp_ta_list)[0], {"です・ます"})[:-2]
        else:
            return(random.choice(vp_ta_list)[0])
    elif 'tp' in element:
        # return(random.choice([str(random.randint(0, 23)) + '時', str(random.randint(1, 12)) + '月']))
        return(random.choice([str(random.randint(1, 12)) + '月']))
    elif 'interval' in element:
        return(str(random.randint(1, 5)) + random.choice(['時間', '日間', '年間']))
    elif 'place' in element:
        return(random.choice(place_list))
    elif 'time_unit' in element:
        return(random.choice(['月', '年', '日']))
    else:
        return(element)


def assign_word(element):
    if 'agent' in element:
        return(random.choice(proper_agent_list))
    if 'np' in element:
        return(random.choice(np_list))
    elif 'vp_zi' in element:
        if 'conj' in element:
            return(random.choice(vp_zi_list)[1])
        elif 'imp' in element:
            return(random.choice(vp_zi_list)[2])
        elif 'past' in element:
            return kotodama.transformVerb(random.choice(vp_zi_list)[0], {"過去"})
        elif 'prog' in element:
            return kotodama.transformVerb(random.choice(vp_zi_list)[0], {"て"})
        elif 'coni' in element:
            return kotodama.transformVerb(random.choice(vp_zi_list)[0], {"です・ます"})[:-2]
        else:
            return(random.choice(vp_zi_list)[0])
    elif 'vp_ta' in element:
        if 'conj' in element:
            return(random.choice(vp_ta_list)[1])
        elif 'imp' in element:
            return(random.choice(vp_ta_list)[2])
        elif 'past' in element:
            return kotodama.transformVerb(random.choice(vp_ta_list)[0], {"過去"})
        elif 'prog' in element:
            return kotodama.transformVerb(random.choice(vp_ta_list)[0], {"て"})
        elif 'coni' in element:
            return kotodama.transformVerb(random.choice(vp_ta_list)[0], {"です・ます"})[:-2]
        else:
            return(random.choice(vp_ta_list)[0])
    elif 'tp' in element:
        # return(random.choice([str(random.randint(0, 23)) + '時', str(random.randint(1, 12)) + '月']))
        return(random.choice([str(random.randint(1, 12)) + '月']))
    elif 'interval' in element:
        return(str(random.randint(1, 5)) + random.choice(['時間', '日間', '年間']))
    elif 'place' in element:
        return(random.choice(place_list))
    elif 'time_unit' in element:
        return(random.choice(['月', '年', '日']))
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
    if model_name == bert:
        mask_text = ''.join([tokenizer.mask_token if 'vp' in e else e for e in c_elements])
    elif model_name == roberta:
        sentence = ''.join([tokenizer.mask_token if 'vp' in e else e for e in c_elements])
        mask_text = ' '.join([token.surface for token in janome_tokenizer.tokenize(sentence)]).replace('[ MASK ]', '[MASK]')
    result = fill_mask(mask_text)[0]
    return result['token_str'].replace(' ', ''), result['sequence'].replace(' ', '')


def assign_np(elements, ind):
    c_elements = deepcopy(elements)
    np_count = sum([1 if 'np' in e else 0 for e in elements])
    if np_count > 1:
        for i in range(len(elements)):
            if i != ind and 'np' in elements[i]:
                c_elements[i] = '太郎'
    if model_name == bert:
        mask_text = ''.join([tokenizer.mask_token if 'np' in e else e for e in c_elements])
    elif model_name == roberta:
        sentence = ''.join([tokenizer.mask_token if 'np' in e else e for e in c_elements])
        mask_text = ' '.join([token.surface for token in janome_tokenizer.tokenize(sentence)]).replace('[ MASK ]', '[MASK]')
    result = fill_mask(mask_text)[0]

    return result['token_str'].replace(' ', ''), result['sequence'].replace(' ', '')


def transform_end(verb, text):
    for token in janome_tokenizer.tokenize(text):
        if token.surface == verb:
            if token.part_of_speech.split(',')[0] != '動詞':
                return token.base_form + 'する'
            return token.base_form
    return 'error'


def inflect_memo(memo, element, verb):
    element = element.replace('_conj', '').replace('_imp', '')
    memo[element] = verb
    try:
        memo[element + '_conj'] = kotodama.transformVerb(verb, {'過去'})[:-1]
        memo[element + '_imp'] = kotodama.transformVerb(verb, {'否定'})[:-2]
    except(ValueError):
        pass
    return memo


def compute_perplexity(sentence):
    sentence = ' '.join([token.surface for token in janome_tokenizer.tokenize(sentence)])
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())


def generate_dataset_with_verb_predict(out_file):
    texts = []
    for template in tqdm(templates):
        for _ in range(1):
            memo = {}
            (prem, hyp) = template
            prems = re.split("(?<=。)", prem)[:-1]
            prems_elements = [prem.split(' ') for prem in prems]
            hyp_elements = hyp.split(' ')
            for elements in (prems_elements + [hyp_elements]):
                for i, element in enumerate(elements):
                    if element in memo:
                        elements[i] = memo[element]
                    elif 'vp' not in element:
                        elements[i] = assign_word_wo_vp(element)
                        memo[element] = elements[i]
                for i, element in enumerate(elements):
                    if element not in (list(memo.keys()) + list(memo.values())):
                        elements[i], text = assign_verb(elements, i)
                        end_form = transform_end(elements[i], text)
                        memo = inflect_memo(memo, element, end_form)
                    elif element in memo:
                        elements[i] = memo[element]

            texts.append([''.join(sum(prems_elements, [])), ''.join(hyp_elements)])

    with open(out_file, 'w') as outfile:
        outfile.write('premise\thypothesis\n')
        for text in texts:
            outfile.write(f'{text[0]}\t{text[1]}\n')


def generate_dataset_with_np_predict(out_file):
    texts = []
    for template in tqdm(templates):
        for _ in range(1):
            memo = {}
            (prem, hyp) = template
            prems = re.split("(?<=。)", prem)[:-1]
            prems_elements = [prem.split(' ') for prem in prems]
            hyp_elements = hyp.split(' ')
            for elements in (prems_elements + [hyp_elements]):
                for i, element in enumerate(elements):
                    if element in memo:
                        elements[i] = memo[element]
                    elif 'np' not in element:
                        elements[i] = assign_word_wo_np(element)
                        memo[element] = elements[i]
                for i, element in enumerate(elements):
                    if element not in (list(memo.keys()) + list(memo.values())):
                        elements[i], text = assign_np(elements, i)
                        memo[element] = elements[i]
                    elif element in memo:
                        elements[i] = memo[element]

            texts.append([''.join(sum(prems_elements, [])), ''.join(hyp_elements)])

    with open(out_file, 'w') as outfile:
        outfile.write('premise\thypothesis\n')
        for text in texts:
            outfile.write(f'{text[0]}\t{text[1]}\n')


def generate_dataset_with_perplexity(out_file):
    texts = []
    for template in tqdm(templates):
        for _ in range(1):
            memo = {}
            (prem, hyp) = template
            prems = re.split("(?<=。)", prem)[:-1]
            prems_elements = [prem.split(' ') for prem in prems]
            hyp_elements = hyp.split(' ')
            for elements in (prems_elements + [hyp_elements]):
                perplexity = 100
                while (perplexity > 50):
                    temp_memo = deepcopy(memo)
                    new_elements = deepcopy(elements)
                    for i, element in enumerate(new_elements):
                        if element in temp_memo:
                            new_elements[i] = temp_memo[element]
                        else:
                            new_elements[i] = assign_word(element)
                            temp_memo[element] = new_elements[i]
                    perplexity = compute_perplexity(''.join(new_elements))
                memo = temp_memo
                elements = new_elements
                print(''.join(elements))

            texts.append([''.join(sum(prems_elements, [])), ''.join(hyp_elements)])

    with open(out_file, 'w') as outfile:
        outfile.write('premise\thypothesis\n')
        for text in texts:
            outfile.write(f'{text[0]}\t{text[1]}\n')



if __name__ == '__main__':
    random.seed(0)
    janome_tokenizer = Tokenizer()
    kotodama.setSegmentationEngine(kotodama.SegmentationEngine.JANOME, janome_tokenizer)
    kotodama.disableError(Juman())

    with open('./dataset/template/template_test.tsv', 'r') as infile:
        templates = infile.read().splitlines()[1:]
        templates = [template.split('\t') for template in templates]

    with open('./vocab_list/place_list.txt', 'r') as infile:
        place_list = infile.read().splitlines()

    with open('./vocab_list/np_list.txt', 'r') as infile:
        np_list = infile.read().splitlines()

    with open('./vocab_list/intransive_verb_list.txt', 'r') as infile:
        vp_zi_list = infile.read().splitlines()
        vp_zi_list = [vp_zi.split(',') for vp_zi in vp_zi_list]

    with open('./vocab_list/transive_verb_list.txt', 'r') as infile:
        vp_ta_list = infile.read().splitlines()
        vp_ta_list = [vp_ta.split(',') for vp_ta in vp_ta_list]

    with open('./vocab_list/common_agent_list.txt', 'r') as infile:
        common_agent_list = infile.read().splitlines()

    with open('./vocab_list/proper_agent_list.txt', 'r') as infile:
        proper_agent_list = infile.read().splitlines()

    bert = "cl-tohoku/bert-base-japanese-whole-word-masking"
    roberta = "nlp-waseda/roberta-large-japanese"
    model_name = roberta

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    out_file = 'dataset/' + 'roberta/dataset_with_perplexity.tsv'
    # generate_dataset_with_verb_predict(out_file)
    # generate_dataset_with_np_predict(out_file)
    generate_dataset_with_perplexity(out_file)
