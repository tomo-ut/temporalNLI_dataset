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
import pickle
import MeCab


# 単語の品詞を取得する
def get_pos(word):
    p_token = f"{word}\n"
    parsed_lines = tagger.parse(p_token).splitlines()[:-1]
    features = [parsed_line.split("\t")[1] for parsed_line in parsed_lines]
    pos = [feature.split(",")[0] for feature in features]
    return ''.join(pos)


# 動詞の活用
def verb_transform(element, verb_list):
    if 'conj' in element:
        return(random.choice(verb_list)[1])
    elif 'imp' in element:
        return(random.choice(verb_list)[2])
    elif 'past' in element:
        return kotodama.transformVerb(random.choice(verb_list)[0], {"過去"})
    elif 'prog' in element:
        return kotodama.transformVerb(random.choice(verb_list)[0], {"て"})
    elif 'coni' in element:
        return kotodama.transformVerb(random.choice(verb_list)[0], {"です・ます"})[:-2]
    else:
        return(random.choice(verb_list)[0])


# 全てに語を割り当てる woに指定されたものはパスされる
def assign_word(element, without=[]):
    if 'agent' in element and "agent" not in without:
        return(random.choice(proper_agent_list))
    if 'np' in element and "np" not in without:
        return(random.choice(np_list))
    if "vp" not in without:
        if 'vp' in element:
            return verb_transform(element, vp_list)
        if 'vp_zi' in element:
            return verb_transform(element, vp_zi_list)
        if 'vp_ta' in element:
            return verb_transform(element, vp_ta_list)
    if 'tp' in element and 'tp' not in without:
        # return(random.choice([str(random.randint(0, 23)) + '時', str(random.randint(1, 12)) + '月']))
        return(random.choice([str(random.randint(1, 12)) + '月']))
    if 'interval' in element and 'interval' not in without:
        return(str(random.randint(1, 5)) + random.choice(['時間', '日間', '年間']))
    if 'place' in element and 'place' not in without:
        return(random.choice(place_list))
    if 'time_unit' in element and 'timu_unit' not in without:
        return(random.choice(['月', '年', '日']))

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
                        elements[i] = assign_word(element, ["vp"])
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
                        elements[i] = assign_word(element, ["np"])
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


# テンプレートにあった格フレームの選択
def choice_cf(verb):
    cases = re.match(".+?\\[(.+?):\\d+\\]", verb).groups()[0].split(',')
    origin_cases = [case for case in cases]
    cases = set([re.match("(.+?)\\d*$", case).groups()[0] for case in origin_cases])
    case_cands = [cf_key for cf_key in cf_keys if cases <= cf_key]
    # 全ての条件を満たした有効な核フレームが得られるまでループ
    while True:
        is_validcf = True
        # 選んだ格フレーム、動詞がブラックリストに入ってる動詞や変な動詞に該当しなくなるまでループ
        while True:
            is_in_nokaku = False
            selected_case = ','.join(sorted(list(random.choice(case_cands))))
            selected_entry = random.choice(list(cf_dict[selected_case].keys()))
            split_entries = selected_entry.split('+')
            origin_verb = ''.join([split_entry[:split_entry.find('/')] for split_entry in split_entries])
            if origin_verb in black_verb_set:
                continue
            if 'ノ格' not in ''.join(list(cases)):
                for case in cases:
                    if 'ノ格~' + case in selected_case:
                        is_in_nokaku = True
            if is_in_nokaku:
                continue
            if '外の関係' in selected_case:
                continue
            if '+' not in selected_entry.replace('+する/する', '').replace('+れる/れる', ''):
                break
        selected_dict = cf_dict[selected_case][selected_entry]
        new_dict = {}
        word_list = []
        # 選んだ格フレームからテンプレート中の格に単語割り当て
        for case in origin_cases:
            case_wo_ind = re.match("(.+?)\\d*$", case).groups()[0]
            inds = list(range(len(selected_dict[case_wo_ind])))
            random.shuffle(inds)
            for ind in inds:
                words = selected_dict[case_wo_ind][ind].split('+')
                word = ''.join([w[:w.find('/')] for w in words])
                pos = get_pos(word)
                if word in word_list or word in black_vocab_set or '名詞' not in pos or '動詞' in pos:
                    continue
                elif case_wo_ind == 'ガ格' and word not in agent_list:
                    continue
                else:
                    word_list.append(word)
                new_dict[case] = word
                # if 'ノ格~' + case_wo_ind in selected_dict:
                #     nokakus = random.choice(selected_dict['ノ格~' + case_wo_ind]).split('+')
                #     nokaku = ''.join([n[:n.find('/')] for n in nokakus])
                #     new_dict[case] = nokaku + 'の' + word
                if '<' not in new_dict[case] and '>' not in new_dict[case]:
                    break
            else:
                is_validcf = False
        if is_validcf:
            break
    return selected_entry, new_dict


# 格フレームを用いた文生成
def generate_sentence_with_cf(template):
    memo = {}
    (prem, hyp) = template

    prems = re.split("(?<=。)", prem)[:-1]
    prems_elements = [prem.split(' ') for prem in prems]
    hyp_elements = hyp.split(' ')
    sentences = prems_elements + [hyp_elements]
    case_list = []
    verbs = []
    form_specifies = []
    for element in sum(sentences, []):
        if "vp" in element and "[" in element:
            while True:
                form_specify = set()
                verb, other_case = choice_cf(element)
                verb_base = verb[:verb.find('/')]
                if "+する/する+れる/れる" in verb:
                    verb = verb_base + 'する'
                    form_specify.add("受け身")
                elif "+する/する" in verb:
                    verb = verb_base + 'する'
                else:
                    verb = verb_base
                    try:
                        kotodama.transformVerb(verb, {"過去"} | form_specify)
                    except ValueError:
                        verb += 'する'
                try:
                    kotodama.transformVerb(verb, {"過去"} | form_specify)
                except ValueError:
                    continue
                break
            case_list.append(other_case)
            form_specifies.append(form_specify)
            verbs.append(verb)

    for i, sentence in enumerate(sentences):
        for j, element in enumerate(sentence):
            if element in memo:
                new_element = memo[element]
            elif 'vp' in element:
                ind = int(re.match('.*?(\\d).*?', element).groups()[0]) - 1
                if 'past' in element:
                    new_element = kotodama.transformVerb(verbs[ind], {"過去"} | form_specifies[ind])
                elif 'prog' in element:
                    new_element = kotodama.transformVerb(verbs[ind], {"て"} | form_specifies[ind])
                elif 'coni' in element:
                    new_element = kotodama.transformVerb(verbs[ind], {"です・ます"} | form_specifies[ind])[:-2]
                else:
                    new_element = kotodama.transformVerb(verbs[ind], form_specifies[ind])
            elif '[' in element:
                ind = int(element[element.find(':') + 1:element.find(']')]) - 1
                new_element = case_list[ind][element[element.find('[') + 1:element.find(':')]]
                memo[element[:element.find('[')]] = new_element
            else:
                new_element = assign_word(element, ["agent", "np", "vp", "place"])
                memo[element] = new_element
            sentences[i][j] = new_element

    return [''.join(sentence) for sentence in sentences]


# 格フレームを用いたデータ生成
def generate_dataset_with_cf(out_file, perplexity_check):
    texts = []
    for template in tqdm(templates):
        # 1つのテンプレートあたりいくつのインスタンスを生成するか
        for _ in range(10):
            while True:
                text = generate_sentence_with_cf(template)
                max_perplexity = 0
                if perplexity_check:
                    for t in text:
                        max_perplexity = max(max_perplexity, compute_perplexity(t))
                if max_perplexity < 100:
                    break
                random.seed(max_perplexity)
            texts.append([''.join(text[:-1]), text[-1]])

    with open(out_file, 'w') as outfile:
        outfile.write('premise\thypothesis\n')
        for text in texts:
            outfile.write(f'{text[0]}\t{text[1]}\n')


if __name__ == '__main__':
    random.seed(0)
    tagger = MeCab.Tagger("-p")
    janome_tokenizer = Tokenizer()
    kotodama.setSegmentationEngine(kotodama.SegmentationEngine.JANOME, janome_tokenizer)
    kotodama.disableError(Juman())
    with open("./external_data/kyoto-univ-web-cf-2.0/cf_dict.pickle", "rb") as f:
        cf_dict = pickle.load(f)
    cf_keys = list(cf_dict.keys())
    cf_keys = [set(key.split(',')) for key in cf_keys]

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

    with open('./vocab_list/agent_list.txt', 'r') as infile:
        agent_list = infile.read().splitlines()

    with open('./vocab_list/vocab_black_list.txt', 'r') as infile:
        black_vocab_set = set(infile.read().splitlines())

    with open('./vocab_list/verb_black_list.txt', 'r') as infile:
        black_verb_set = set(infile.read().splitlines())

    bert = "cl-tohoku/bert-base-japanese-whole-word-masking"
    roberta = "nlp-waseda/roberta-large-japanese"

    perplexity_check = False
    if perplexity_check:
        # for model_name_str in ['bert', 'roberta']:
        for model_name_str in ['roberta']:
            if model_name_str == "roberta":
                model_name = roberta
            else:
                model_name = bert
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

            # out_file = 'dataset/' + model_name_str
            # generate_dataset_with_verb_predict(out_file + '/dataset_with_vp_predict')
            # generate_dataset_with_np_predict(out_file + '/dataset_with_np_predict')
            # generate_dataset_with_perplexity(out_file + '/dataset_with_perplexity')

    generate_dataset_with_cf('dataset/cf/dataset_with_cf_agent', perplexity_check)