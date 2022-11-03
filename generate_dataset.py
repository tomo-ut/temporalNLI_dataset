import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from janome.tokenizer import Tokenizer
from kotodama import kotodama
from pyknp import Juman
import pandas as pd
import re
import pickle
import MeCab
import datetime
import time
import os
from collections import defaultdict
from wakati_dataset import wakati


# 単語の品詞を取得する
def get_pos(word):
    p_token = f"{word}\n"
    parsed_lines = tagger.parse(p_token).splitlines()[:-1]
    features = [parsed_line.split("\t")[1] for parsed_line in parsed_lines]
    pos = [feature.split(",")[0] for feature in features]
    return ''.join(pos)


# テンプレートにあった格フレームの選択
def choice_cf(verb, wordlist_dict, cf_dict, cf_keys):
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
            if origin_verb in wordlist_dict['black_verb']:
                continue
            if '時間' not in selected_case:
                continue
            # if 'ノ格' not in ''.join(list(cases)):
            #     for case in cases:
            #         if 'ノ格~' + case in selected_case:
            #             is_in_nokaku = True
            if is_in_nokaku:
                continue
            if '外の関係' in selected_case:
                continue
            # if '+' not in selected_entry.replace('+する/する', '').replace('+れる/れる', ''):
            # 受け身 れる の出現無しパターン
            if '+' not in selected_entry.replace('+する/する', ''):
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
                if word in word_list or word in wordlist_dict['black_vocab'] or '名詞' not in pos or '動詞' in pos:
                    continue
                elif case_wo_ind == 'ガ格' and word not in wordlist_dict['agent']:
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


def generate_condition(time_infos, template, memo):
    entail_conds = template['entailment'].split(',')
    contradict_conds = template['contradiction'].split(',')
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


def generate_ans(memo, template):
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
    entail_cond, contradict_cond = generate_condition(time_infos, template, memo)

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


def assign_time(element, tp_format, interval_format, memo):
    if '-' in element:
        tp = element[:element.find('-')]
        reftime = memo[tp]
        if 'day' in element:
            diff = int(re.search('(\\d+)day', element).groups()[0])
            daynum = int(re.search('(\\d+)日', reftime).groups()[0])
            newday = daynum - diff
            return reftime.replace(f'{daynum}日', f'{newday}日')
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
def generate_sentence_with_cf(template, wordlist_dict, cf_dict, cf_keys, tp_format, interval_format):
    memo = {}
    prem = template['premise']
    hyp = template['hypothesis']
    limited_agent_list = wordlist_dict['agents'].copy()
    random.shuffle(limited_agent_list)

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
                verb, other_case = choice_cf(element, wordlist_dict, cf_dict, cf_keys)
                verb_base = verb[:verb.find('/')]
                if "+する/する+れる/れる" in verb and get_pos(verb_base) == '名詞':
                    verb = verb_base + 'する'
                    form_specify.add("受け身")
                elif "+する/する" in verb and get_pos(verb_base) == '名詞':
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
                if 'stative' in element and verbs[ind] not in wordlist_dict['stative_verb']:
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
                new_element = assign_time(element, tp_format, interval_format, memo)
                memo[element] = new_element
            elif element and element[0] == '[':
                cands = element[1:-1].split(',')
                new_element = random.choice(cands)
                memo[element] = new_element
            elif 'agent' in element:
                new_element = limited_agent_list.pop()
                if '[' in element:
                    memo[element[:element.find('[')]] = new_element
                else:
                    memo[element] = new_element
            elif '[' in element:
                ind = int(element[element.find(':') + 1:element.find(']')]) - 1
                new_element = case_list[ind][element[element.find('[') + 1:element.find(':')]]
                memo[element[:element.find('[')]] = new_element
            else:
                new_element = element
                memo[element] = new_element
            sentences[i][j] = new_element

    answer = generate_ans(memo, template)

    return [''.join(sentence) for sentence in sentences], answer


# 格フレームを用いたデータ生成
def generate_dataset_with_cf(out_file, templates, wordlist_dict, cf_dict, cf_keys, data_num):
    texts = []
    time_unit_list = ['year', 'month', 'day', 'hour']
    tu2intervalformat = {"year": "i年間", "month": "iヶ月間", "day": "i日間", "hour": "i時間"}
    tp_format_list = {"i時間": ["y年m月d日h時", "m月d日h時", "d日h時", "h時"],
                      "i日間": ["y年m月d日", "m月d日", "d日"],
                      "iヶ月間": ["y年m月", "m月"],
                      "i年間": ["y年"]}
    label_dist = defaultdict(int)
    for template in tqdm(templates):
        for time_unit in time_unit_list:
            if time_unit in template['ng time unit'].split(','):
                continue
            interval_format = tu2intervalformat[time_unit]
            idx = 0
            for tp_format in tp_format_list[interval_format]:
                idx += 1
                if 'tp' not in template['premise'] + template['hypothesis'] and idx > 1:
                    continue
                # 1つのテンプレートあたりいくつのインスタンスを生成するか
                for _ in range(data_num):
                    while True:
                        text, ans = generate_sentence_with_cf(
                            template, wordlist_dict, cf_dict, cf_keys, tp_format, interval_format)
                        zero_flag = False
                        for t in text:
                            zero = re.search('([^\\d]0[月,日])|(-1時)|(^0[月,日])', t)
                            if zero:
                                zero_flag = True
                        if not zero_flag:
                            break
                        # random.seed(max_perplexity)
                    if 'tp' in template['premise'] + template['hypothesis']:
                        time_format = re.sub(r"[a-zA-Z]", "", tp_format)
                    elif 'interval' in template['premise'] + template['hypothesis']:
                        time_format = interval_format[-2:]
                    else:
                        time_format = "None"
                    texts.append([''.join(text[:-1]), text[-1], ans, template['id'], time_format])
                    label_dist[ans] += 1

    with open(out_file, 'w') as outfile:
        outfile.write('num\tpremise\thypothesis\tgold_label\ttemplate_num\ttime_format\n')
        idx = 1
        for text in texts:
            outfile.write(f'{str(idx)}\t{text[0]}\t{text[1]}\t{text[2]}\t{text[3]}\t{text[4]}\n')
            idx += 1
    print(out_file + ':\t', end='')
    print(label_dist)


def split_dataset(templates, ver, split, seed=0, mode='random'):
    split_str = mode + '-' + str(split)
    if os.path.exists(f'dataset/ver_{ver}/test_{split_str}.tsv') \
            and os.path.exists(f'dataset/ver_{ver}/test_{split_str}_wakati.tsv'):
        return 0
    if mode == 'random':
        test_size = 1 - float('0.' + str(split))
        templates_train, templates_test = train_test_split(templates, test_size=test_size, random_state=seed)
        templates_train_idx = [template_train['id'] for template_train in templates_train]
        templates_test_idx = [template_test['id'] for template_test in templates_test]
        templates_train_idx.sort()
        templates_test_idx.sort()
    elif mode == 'custom':
        templates_test_idx = pd.read_csv(f'dataset/split_info/{split_str}.txt', header=None).to_dict('list')[0]
        all_ids = [template['id'] for template in templates]
        templates_train_idx = sorted(list(set(all_ids) - set(templates_test_idx)))
    print(f"train template: {len(templates_train_idx)}, test template: {len(templates_test_idx)}")
    for suffix in ['', '_wakati']:
        all_problems = pd.read_csv(f'./dataset/ver_{ver}/all{suffix}.tsv', sep='\t')
        train_problems = all_problems.query('template_num in @templates_train_idx')
        test_problems = all_problems.query('template_num in @templates_test_idx')
        train_problems.to_csv(f'dataset/ver_{ver}/train_{split_str}{suffix}.tsv', sep='\t', index=False)
        test_problems.to_csv(f'dataset/ver_{ver}/test_{split_str}{suffix}.tsv', sep='\t', index=False)
    return 0


def extract_test_problem(path):
    problems = pd.read_csv(path, sep='\t').to_dict('records')
    extracted_problems = []
    for problem in problems:
        pass
    pd.DataFrame(extracted_problems).to_csv(path, sep='\t', index=False)


def initialize(template_ver):
    wordlist_dict = {}

    with open('./vocab_list/agent_list.txt', 'r') as infile:
        wordlist_dict['agent'] = infile.read().splitlines()

    with open('./vocab_list/agents.txt', 'r') as infile:
        wordlist_dict['agents'] = infile.read().splitlines()

    with open('./vocab_list/vocab_black_list.txt', 'r') as infile:
        wordlist_dict['black_vocab'] = set(infile.read().splitlines())

    with open('./vocab_list/verb_black_list.txt', 'r') as infile:
        wordlist_dict['black_verb'] = set(infile.read().splitlines())

    with open('./vocab_list/stative_verb_list.txt', 'r') as infile:
        wordlist_dict['stative_verb'] = infile.read().splitlines()

    cf = "cf_dict_casefreq1000_wordfreq100"
    with open(f"./external_data/kyoto-univ-web-cf-2.0/{cf}.pickle", "rb") as f:
        cf_dict = pickle.load(f)
    cf_keys = list(cf_dict.keys())
    cf_keys = [set(key.split(',')) for key in cf_keys]

    templates = pd.read_csv(f'./dataset/template/template_{template_ver}.tsv', sep='\t').to_dict('records')

    return templates, wordlist_dict, cf_dict, cf_keys


def main():
    random.seed(0)
    template_ver = 'ver_1_1_1'
    ver = '1_2'
    split_dict = {'random': [9, 8, 7], 'custom': [0, 1, 2]}
    do_split = {'random': True, 'custom': True}
    do_update = False
    do_wakati = True
    path = f'dataset/ver_{ver}/'

    os.makedirs(f'./dataset/ver_{ver}', exist_ok=True)
    templates, wordlist_dict, cf_dict, cf_keys = initialize(template_ver)

    generate_dataset_with_cf(path + 'train_all.tsv', templates, wordlist_dict, cf_dict, cf_keys, 50)
    generate_dataset_with_cf(path + 'test_all.tsv', templates, wordlist_dict, cf_dict, cf_keys, 50)
    extract_test_problem(path + 'test_all.tsv')

    for mode in ['train', 'test']:
        if ((not os.path.exists(path + f'{mode}_all_wakati.tsv')) and do_wakati) or do_update:
            wakati(path + f'{mode}_all')

    for mode in ['random', 'custom']:
        if not do_split[mode]:
            continue
        for split in split_dict[mode]:
            split_dataset(templates, ver, split, mode)


if __name__ == '__main__':
    tagger = MeCab.Tagger("-p")
    janome_tokenizer = Tokenizer()
    kotodama.setSegmentationEngine(kotodama.SegmentationEngine.JANOME, janome_tokenizer)
    kotodama.disableError(Juman())
    main()
