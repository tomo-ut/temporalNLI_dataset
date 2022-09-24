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
import datetime
import time


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
    if 'time_unit' in element and 'time_unit' not in without:
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


def substitute_time(cond, time_infos, memo):
    for op in ['or', '<=', '<', '>=', '>', '!=', '==', '+', '-', '*']:
        if op in cond:
            return substitute_time(cond.split(op)[0], time_infos, memo) + \
                f' {op} ' + substitute_time(cond.split(op)[1], time_infos, memo)
    if 'tp' in cond:
        if 'start' in cond:
            return str(time_infos[cond.replace(' ', '').replace('.start', '')]['start']['nt'])
        if 'end' in cond:
            return str(time_infos[cond.replace(' ', '').replace('.end', '')]['end']['nt'])
        if 'min_unit' in cond:
            return "'" + str(time_infos[cond.replace(' ', '').replace('.min_unit', '')]['min_unit']) + "'"
    if 'interval' in cond:
        return str(time_infos[cond.replace(' ', '')]['start']['nt'])
    if 'time_unit' in cond:
        d = {'年': "'year'", '月': "'month'", '日': "'day'", '時': "'hour'"}
        return d[memo[cond.replace(' ', '')]]
    c = 1
    for unit in ['hour', 'day', 'month', 'year']:
        if unit in cond:
            return str(c)
        c *= 100

    return cond.replace(' ', '')


def generate_condition(time_infos, cond, memo):
    entail_conds = cond[0].split(',')
    contradict_conds = cond[1].split(',')
    formula = []
    for conds in [entail_conds, contradict_conds]:
        formulas = []
        for cond in conds:
            formulas.append(substitute_time(cond, time_infos, memo))
        formula.append(' and '.join(formulas))

    return formula[0], formula[1]


def sum_time(time_info):
    return time_info['hour'] + 10**2 * time_info['day'] + 10**4 * time_info['month'] + 10**6 * time_info['year']


def extract_time(time_str, element):
    time_info = {}
    time_info['start'] = {}
    time_info['end'] = {}
    time_info['max_unit'] = ''
    time_info['min_unit'] = ''
    time_info['str'] = element
    for unit in ['year', 'month', 'date', 'day', 'hour']:
        time_info['start'][unit] = 0
        time_info['end'][unit] = 0

    if 'tp' in element:
        patterns = ['(\\d+)時', '(.)曜', '(\\d+)日', '(\\d+)月', '(\\d+)年']
        time_info['type'] = 'tp'
    elif 'interval' in element:
        patterns = ['(\\d+)時', '(\\d+)日', '(\\d+)ヶ月', '(\\d+)年']
        time_info['type'] = 'interval'
    jp2en = {'年': 'year', '月': 'month', '日': 'day', '時': 'hour'}

    for pattern in patterns:
        match = re.search(pattern, time_str)
        if match:
            time_info['start'][jp2en[pattern[-1]]] = int(match.groups()[0])
            time_info['max_unit'] = pattern[-1]
            if time_info['min_unit'] == '':
                time_info['min_unit'] = jp2en[pattern[-1]]

    if 'tp' in element:
        if time_info['min_unit'] == 'year':
            for key in time_info['start'].keys():
                time_info['end'][key] = time_info['start'][key]
            time_info['end']['year'] += 1
        elif time_info['min_unit'] == 'month':
            for key in time_info['start'].keys():
                time_info['end'][key] = time_info['start'][key]
            time_info['end']['month'] += 1
            if time_info['end']['month'] == 13:
                time_info['end']['month'] = 1
                time_info['end']['year'] += 1
        elif time_info['min_unit'] == 'day':
            default_values = [2020, 1, 1]
            values = [time_info['start'][key] if time_info['start'][key] else default_values[i]
                      for i, key in enumerate(['year', 'month', 'day'])]
            dt = datetime.date(values[0], values[1], values[2])
            td = datetime.timedelta(1)
            dt = dt + td
            time_info['end']['year'] = dt.year if time_info['start']['year'] else 0
            time_info['end']['month'] = dt.month if time_info['start']['month'] else 0
            time_info['end']['day'] = dt.day
            if dt.month == 1 and dt.day == 1 and not time_info['end']['year']:
                time_info['end']['year'] = 1
            elif dt.day == 1 and not time_info['end']['month']:
                time_info['end']['month'] = 1
            time_info['end']['hour'] = time_info['start']['hour']
        elif time_info['min_unit'] == 'hour':
            for key in time_info['start'].keys():
                time_info['end'][key] = time_info['start'][key]
        time_info['end']['nt'] = sum_time(time_info['end'])
    time_info['start']['nt'] = sum_time(time_info['start'])

    return time_info


def generate_ans(memo, cond):
    time_infos = {}
    for e in memo:
        if 'tp' in e:
            time_str = memo[e]
            time_info = extract_time(time_str, e)
            time_infos[e] = time_info
        elif 'interval' in e:
            time_str = memo[e]
            time_info = extract_time(time_str, e)
            time_infos[e] = time_info
    entail_cond, contradict_cond = generate_condition(time_infos, cond, memo)

    if eval(entail_cond):
        return "entailment"
    elif eval(contradict_cond):
        return "contradiction"
    else:
        return "neutral"


def random_date(start, end):
    stime = time.mktime(time.strptime(start, "%Y,%m,%d,%H"))
    etime = time.mktime(time.strptime(end, "%Y,%m,%d,%H"))
    ptime = stime + random.random() * (etime - stime)
    return time.strftime("%Y,%m,%d,%H", time.localtime(ptime))


def assign_time(element, tp_format, interval_format):
    time = random_date("2000,1,1,0", "2021,1,1,0")
    year, month, day, hour = [e.lstrip('0') for e in time.split(",")]
    if hour == '':
        hour = '0'
    interval = str(random.choice(range(1, 10)))
    if 'tp' in element:
        return tp_format.replace('y', year).replace('m', month).replace('d', day).replace('h', hour)
    if 'interval' in element:
        return interval_format.replace('i', interval)


# 格フレームを用いた文生成
def generate_sentence_with_cf(template, tp_format, interval_format):
    memo = {}
    (prem, hyp) = template[:2]

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
                suffix = ''
                if 'stative' in element and verbs[ind] not in stative_verb_list:
                    element += 'prog'
                    suffix = 'いる'
                    if 'past' in element:
                        suffix = 'いた'
                        element = element.replace('past', '')
                if 'past' in element:
                    new_element = kotodama.transformVerb(verbs[ind], {"過去"} | form_specifies[ind])
                elif 'prog' in element:
                    new_element = kotodama.transformVerb(verbs[ind], {"て"} | form_specifies[ind])
                elif 'coni' in element:
                    new_element = kotodama.transformVerb(verbs[ind], {"です・ます"} | form_specifies[ind])[:-2]
                else:
                    new_element = kotodama.transformVerb(verbs[ind], form_specifies[ind])
                new_element += suffix
            elif 'tp' in element or 'interval' in element:
                new_element = assign_time(element, tp_format, interval_format)
                memo[element] = new_element
            elif '[' in element:
                ind = int(element[element.find(':') + 1:element.find(']')]) - 1
                new_element = case_list[ind][element[element.find('[') + 1:element.find(':')]]
                memo[element[:element.find('[')]] = new_element
            else:
                new_element = assign_word(element, ["agent", "np", "vp", "place", "tp", "interval"])
                memo[element] = new_element
            sentences[i][j] = new_element

    answer = generate_ans(memo, template[2:-1])

    return [''.join(sentence) for sentence in sentences], answer


# 格フレームを用いたデータ生成
def generate_dataset_with_cf(out_file, perplexity_check, data_num):
    texts = []
    time_unit_list = ['year', 'month', 'day', 'hour']
    tu2intervalformat = {"year": "i年間", "month": "iヶ月間", "day": "i日間", "hour": "i時間"}
    tp_format_list = {"i時間": ["y年m月d日h時", "m月d日h時", "d日h時", "h時"],
                      "i日間": ["y年m月d日", "m月d日", "d日"],
                      "iヶ月間": ["y年m月", "m月"],
                      "i年間": ["y年"]}
    template_num = 1
    for template in tqdm(templates):
        for time_unit in time_unit_list:
            if time_unit in template[4].split(','):
                continue
            interval_format = tu2intervalformat[time_unit]
            idx = 0
            for tp_format in tp_format_list[interval_format]:
                idx += 1
                if 'tp' not in template[0] + template[1] and idx > 1:
                    continue
                # 1つのテンプレートあたりいくつのインスタンスを生成するか
                for _ in range(data_num):
                    while True:
                        text, ans = generate_sentence_with_cf(template, tp_format, interval_format)
                        max_perplexity = 0
                        if perplexity_check:
                            for t in text:
                                max_perplexity = max(max_perplexity, compute_perplexity(t))
                        if max_perplexity < 100:
                            break
                        random.seed(max_perplexity)
                    if 'tp' in template[0] + template[1]:
                        time_format = re.sub(r"[a-zA-Z]", "", tp_format)
                    elif 'interval' in template[0] + template[1]:
                        time_format = interval_format[-2:]
                    else:
                        time_format = "None"
                    texts.append([''.join(text[:-1]), text[-1], ans, str(template_num), time_format])
        template_num += 1

    with open(out_file, 'w') as outfile:
        outfile.write('num\tpremise\thypothesis\tgold_label\ttemplate_num\ttime_format\n')
        idx = 1
        for text in texts:
            outfile.write(f'{str(idx)}\t{text[0]}\t{text[1]}\t{text[2]}\t{text[3]}\t{text[4]}\n')
            idx += 1


if __name__ == '__main__':
    random.seed(0)
    tagger = MeCab.Tagger("-p")
    janome_tokenizer = Tokenizer()
    kotodama.setSegmentationEngine(kotodama.SegmentationEngine.JANOME, janome_tokenizer)
    kotodama.disableError(Juman())
    with open("./external_data/kyoto-univ-web-cf-2.0/cf_dict_verb_super_extend.pickle", "rb") as f:
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

    with open('./vocab_list/stative_verb_list.txt', 'r') as infile:
        stative_verb_list = infile.read().splitlines()

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

    generate_dataset_with_cf('dataset/train.tsv', perplexity_check, 80)
    generate_dataset_with_cf('dataset/test.tsv', perplexity_check, 20)
