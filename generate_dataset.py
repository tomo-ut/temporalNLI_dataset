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


# vp 以外に語を割り当てる
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

# np 以外に語を割り当てる


def assign_word_wo_np(element):
    if 'agent' in element:
        return(random.choice(proper_agent_list))
    # 動詞を活用させる
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

# 全てに語を割り当てる


def assign_word(element):
    if 'agent' in element:
        return(random.choice(proper_agent_list))
    if 'np' in element:
        return(random.choice(np_list))
    elif 'vp' in element:
        if 'conj' in element:
            return(random.choice(vp_list)[1])
        elif 'imp' in element:
            return(random.choice(vp_list)[2])
        elif 'past' in element:
            return kotodama.transformVerb(random.choice(vp_list)[0], {"過去"})
        elif 'prog' in element:
            return kotodama.transformVerb(random.choice(vp_list)[0], {"て"})
        elif 'coni' in element:
            return kotodama.transformVerb(random.choice(vp_list)[0], {"です・ます"})[:-2]
        else:
            return(random.choice(vp_list)[0])
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


# MLMによる動詞の割り当て
def assign_verb(elements, ind, memo):
    c_elements = deepcopy(elements)
    for i in range(len(elements)):
        if i != ind and 'vp' in elements[i]:
            if 'imp' in elements[i] or 'conj' in elements[i]:
                c_elements[i] = 'し'
            else:
                c_elements[i] = 'する'
    if model_name == bert:
        for i, e in enumerate(c_elements):
            if 'past' in e or 'prog' in e:
                if 'past' in e:
                    p = random.choices(['た', 'だ'], weights=[7, 3])
                else:
                    p = random.choices(['て', 'で'], weights=[7, 3])
                c_elements[i + 1:i + 1] = p
                elements[i + 1:i + 1] = p
                memo[p[0]] = p[0]
        mask_text = ''.join([tokenizer.mask_token if 'vp' in e else e for e in c_elements])
    elif model_name == roberta:
        sentence = ''.join([tokenizer.mask_token if 'vp' in e else e for e in c_elements])
        mask_text = ' '.join([token.surface for token in janome_tokenizer.tokenize(sentence)]
                             ).replace('[ MASK ]', '[MASK]')
    result = fill_mask(mask_text)[0]
    return result['token_str'].replace(' ', ''), result['sequence'].replace(' ', '')


# MLMによる名詞の割り当て
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
        mask_text = ' '.join([token.surface for token in janome_tokenizer.tokenize(sentence)]
                             ).replace('[ MASK ]', '[MASK]')
    result = fill_mask(mask_text)[0]

    return result['token_str'].replace(' ', ''), result['sequence'].replace(' ', '')


# 終止形に戻す
def transform_end(verb, text):
    for token in janome_tokenizer.tokenize(text):
        if token.surface == verb:
            if token.part_of_speech.split(',')[0] != '動詞':
                return token.base_form + 'する'
            return token.base_form
    return 'error'


# 活用形をメモ
def inflect_memo(memo, element, verb):
    element = re.sub('_conj|_imp|_prog|_past|_coni', '', element)
    memo[element] = verb
    try:
        memo[element + '_conj'] = kotodama.transformVerb(verb, {'過去'})[:-1]
        memo[element + '_imp'] = kotodama.transformVerb(verb, {'否定'})[:-2]
        memo[element + '_prog'] = kotodama.transformVerb(verb, {'て'})
        memo[element + '_past'] = kotodama.transformVerb(verb, {'過去'})
        memo[element + '_coni'] = kotodama.transformVerb(verb, {'です・ます'})[:-2]
    except(ValueError):
        pass
    return memo


# perplexity の計算
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


# vpをMLMで予測してデータを生成
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
                        elements[i], text = assign_verb(elements, i, memo)
                        end_form = transform_end(elements[i], text)
                        memo = inflect_memo(memo, element, end_form)
                    elif element in memo:
                        elements[i] = memo[element]

            texts.append([''.join(sum(prems_elements, [])), ''.join(hyp_elements)])

    with open(out_file, 'w') as outfile:
        outfile.write('premise\thypothesis\n')
        for text in texts:
            outfile.write(f'{text[0]}\t{text[1]}\n')


# npをMLMで予測してデータを生成
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


# MLMによる予測はなしで、perplexityベースで生成
def generate_dataset_with_perplexity(out_file):
    texts = []
    for template in tqdm(templates):
        # 1つのテンプレートあたりいくつのインスタンスを生成するか
        for _ in range(1):
            memo = {}
            (prem, hyp) = template
            prems = re.split("(?<=。)", prem)[:-1]
            prems_elements = [prem.split(' ') for prem in prems]
            hyp_elements = hyp.split(' ')
            e_index = 0
            sentences = prems_elements + [hyp_elements]
            while e_index < len(sentences):
                perplexity = 100
                counter = 0
                while (perplexity > 50):
                    temp_memo = deepcopy(memo)
                    new_elements = deepcopy(sentences[e_index])
                    for i, element in enumerate(new_elements):
                        if element in temp_memo:
                            new_elements[i] = temp_memo[element]
                        else:
                            new_elements[i] = assign_word(element)
                            temp_memo[element] = new_elements[i]
                    perplexity = compute_perplexity(''.join(new_elements))
                    # print(f"{int(perplexity)}" + "\t" + ''.join(new_elements))
                    counter += 1
                    # いつまで経ってもperplexityが小さくならない場合、最初の文ではまっている可能性があるのでやり直し
                    if counter >= 10:
                        break
                if counter >= 10:
                    memo = {}
                    sentences = prems_elements + [hyp_elements]
                    e_index = 0
                    continue
                memo = temp_memo
                sentences[e_index] = new_elements
                e_index += 1

            texts.append([''.join(sum(sentences[:-1], [])), ''.join(sentences[-1])])

    with open(out_file, 'w') as outfile:
        outfile.write('premise\thypothesis\n')
        for text in texts:
            outfile.write(f'{text[0]}\t{text[1]}\n')


if __name__ == '__main__':
    random.seed(0)
    janome_tokenizer = Tokenizer()
    kotodama.setSegmentationEngine(kotodama.SegmentationEngine.JANOME, janome_tokenizer)
    kotodama.disableError(Juman())

    with open('./dataset/template/template.tsv', 'r') as infile:
        templates = infile.read().splitlines()[1:]
        templates = [template.split('\t') for template in templates]

    with open('./vocab_list/place_list.txt', 'r') as infile:
        place_list = infile.read().splitlines()

    # with open('./vocab_list/np_list.txt', 'r') as infile:
    with open('./vocab_list/noun_list.txt', 'r') as infile:
        np_list = infile.read().splitlines()

    with open('./vocab_list/intransive_verb_list.txt', 'r') as infile:
        vp_zi_list = infile.read().splitlines()
        vp_zi_list = [vp_zi.split(',') for vp_zi in vp_zi_list]

    with open('./vocab_list/transive_verb_list.txt', 'r') as infile:
        vp_ta_list = infile.read().splitlines()
        vp_ta_list = [vp_ta.split(',') for vp_ta in vp_ta_list]

    with open('./vocab_list/basic_verb_list.txt', 'r') as infile:
        vp_list = infile.read().splitlines()
        vp_list = [vp.split(',') for vp in vp_list]

    with open('./vocab_list/common_agent_list.txt', 'r') as infile:
        common_agent_list = infile.read().splitlines()

    with open('./vocab_list/proper_agent_list.txt', 'r') as infile:
        proper_agent_list = infile.read().splitlines()

    bert = "cl-tohoku/bert-base-japanese-whole-word-masking"
    roberta = "nlp-waseda/roberta-large-japanese"

    for model_name_str in ['bert', 'roberta']:
        if model_name_str == "roberta":
            model_name = roberta
        else:
            model_name = bert
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

        out_file = 'dataset/' + model_name_str
        generate_dataset_with_verb_predict(out_file + '/dataset_with_vp_predict')
        generate_dataset_with_np_predict(out_file + '/dataset_with_np_predict')
        generate_dataset_with_perplexity(out_file + '/dataset_with_perplexity')
