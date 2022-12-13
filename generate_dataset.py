from sklearn.model_selection import train_test_split
from tqdm import tqdm
from janome.tokenizer import Tokenizer
from kotodama import kotodama
from pyknp import Juman
from collections import defaultdict
from wakati_dataset import wakati
from ast import literal_eval
from pprint import pprint
import random
import pandas as pd
import re
import pickle
import MeCab
import datetime
import time
import os
import collections


# 単語の品詞を取得する
def get_pos(word):
    p_token = f"{word}\n"
    parsed_lines = tagger.parse(p_token).splitlines()[:-1]
    features = [parsed_line.split("\t")[1] for parsed_line in parsed_lines]
    pos = [feature.split(",")[0] for feature in features]
    return ''.join(pos)


# テンプレートにあった格フレームの選択
def choice_cf(verb, wordlist_dict, cf_dict, cf_keys, is_test):
    include_tp = True if 'tp' in verb else False
    include_nint = True if 'nint' in verb else False
    include_cint = True if 'cint' in verb else False
    cases = re.match(".+?\\[(.+?):[^\\s]*?:\\d+\\]", verb).groups()[0].split(',')
    origin_cases = [case for case in cases]
    cases = set([re.match("(.+?)\\d*$", case).groups()[0] for case in origin_cases])
    case_cands = [cf_key for cf_key in cf_keys if cases <= cf_key]
    # 全ての条件を満たした有効な核フレームが得られるまでループ
    while True:
        is_validcf = True
        # 選んだ格フレーム、動詞がブラックリストに入ってる動詞や変な動詞に該当しなくなるまでループ
        while True:
            is_nokaku = False
            is_unnecessary_case = False
            selected_case_set = random.choice(case_cands)
            selected_case = ','.join(sorted(list(selected_case_set)))
            selected_entry = random.choice(list(cf_dict[selected_case].keys()))
            split_entries = selected_entry.split('?')[-1].split('+')
            origin_verb = ''.join([split_entry[:split_entry.find('/')] for split_entry in split_entries])
            if origin_verb in wordlist_dict['black_verb']:
                continue
            if is_test:
                # if 'ノ格' not in ''.join(list(cases)):
                #     for case in cases:
                #         if 'ノ格~' + case in selected_case:
                #             is_nokaku = True
                if set(['ヲ格', 'ニ格']) & (selected_case_set - cases) != set():
                    is_unnecessary_case = True
                if is_nokaku or is_unnecessary_case:
                    continue
                if include_tp and '時間' not in selected_case_set:
                    continue
                if include_nint:
                    if not ('デ格' in selected_case_set and '<時間>' in cf_dict[selected_case][selected_entry]['デ格']):
                        continue
                if include_cint:
                    for c in ['カラ格', 'マデ格']:
                        if c in selected_case_set and '<時間>' in cf_dict[selected_case][selected_entry][c]:
                            break
                    else:
                        continue
            elif '時間' not in selected_case:
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
                if (case_wo_ind != 'ガ格' and word in word_list) or \
                        word in wordlist_dict['black_vocab'] or '名詞' not in pos or '動詞' in pos:
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


def generate_time(elements, tp_format, interval_format, time_span):
    times = {}
    t = random_date("2000,1,1,0", "2021,1,1,0")
    y, m, d, _ = [e.lstrip('0') for e in t.split(",")]
    start_y = random.choice(range(2000, 2017))
    start_m = random.choice(range(1, 9))
    start_d = random.choice(range(1, 19))
    start_h = random.choice(range(0, 16))
    for element in elements:
        if 'tp' not in element and 'interval' not in element and 'time_unit' not in element:
            continue
        if element in times:
            continue
        if 'time_unit' in element:
            times[element] = interval_format[-2]
            continue
        while True:
            if '-' in element and '=' not in element:
                tp = element[:element.find('-')]
                reftime = times[tp]
                if 'day' in element:
                    diff = int(re.search('(\\d+)day', element).groups()[0])
                    daynum = int(re.search('(\\d+)日', reftime).groups()[0])
                    if '-' in element:
                        newday = daynum - diff
                    times[element] = reftime.replace(f'{daynum}日', f'{newday}日')
                break

            if time_span == 'random':
                time = random_date("2000,1,1,0", "2021,1,1,0")
                interval = str(random.choice(range(1, 10)))
            elif time_span == 'short':
                if tp_format[-1] == '時':
                    time = random_date(f"{y},{m},{d},{str(start_h)}", f"{y},{m},{d},{str(start_h + 8)}")
                elif tp_format[-1] == '日':
                    time = random_date(f"{y},{m},{str(start_d)},0", f"{y},{m},{str(start_d + 10)},0")
                elif tp_format[-1] == '月':
                    time = random_date(f"{y},{str(start_m)},1,0", f"{y},{str(start_m + 4)},1,0")
                else:
                    time = random_date(f"{str(start_y)},1,1,0", f"{str(start_y + 6)},1,1,0")
                interval = str(random.choice(range(1, 4)))
            else:
                time = random_date("2000,1,1,0", "2021,1,1,0")
                interval = str(random.choice(range(1, 10)))

            year, month, day, hour = [e.lstrip('0') for e in time.split(",")]
            if hour == '':
                hour = '0'
            if 'tp' in element:
                new_time = tp_format.replace('y', year).replace('m', month).replace('d', day).replace('h', hour)
                if new_time in times.values():
                    continue
                elif '!=' in element:
                    tp = element.split('!=')[1]
                    if '-' in tp:
                        tp = tp[:tp.find('-')]
                        reftime = times[tp]
                        if 'day' in element:
                            diff = int(re.search('(\\d+)day', element).groups()[0])
                            daynum = int(re.search('(\\d+)日', reftime).groups()[0])
                            if '-' in element:
                                newday = daynum - diff
                            reftime = reftime.replace(f'{daynum}日', f'{newday}日')
                    if new_time == reftime:
                        continue
                    times[element.split('!=')[0]] = new_time
                    break
                else:
                    times[element] = new_time
                    break
            if 'interval' in element:
                times[element] = interval_format.replace('i', interval)
                break
    return times


# 格フレームを用いた文生成
def generate_sentence_with_cf(
        template,
        wordlist_dict,
        cf_dict,
        cf_keys,
        is_test):
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
    cf = ''
    for element in sum(sentences, []):
        if "vp" in element and "[" in element:
            while True:
                form_specify = set()
                verb, other_case = choice_cf(element, wordlist_dict, cf_dict, cf_keys, is_test)
                verb_base = verb.split('?')[-1]
                verb_base = verb_base[:verb_base.find('/')]
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
            cf = cf + verb_base + ','
            for key, value in other_case.items():
                if 'ガ格' in key:
                    continue
                cf = cf + key + ',' + value + ','
            case_list.append(other_case)
            form_specifies.append(form_specify)
            if verb == '居った':
                verb = '居た'
            verbs.append(verb)

    cf = cf.rstrip(',')
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
            elif 'tp' in element or 'interval' in element or 'time_unit' in element:
                if i == j == 0 or (sentences[i] == sentences[-1] and j == 0):
                    new_element = element + ' '
                else:
                    new_element = ' ' + element + ' '
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

    return [''.join(sentence) for sentence in sentences], cf


# 中間データに時間を代入
def fill_times(sentence_wo_time, times):
    prem = sentence_wo_time['premise']
    hyp = sentence_wo_time['hypothesis']
    prems = re.split("(?<=。)", prem)[:-1]
    prems_elements = [prem.split(' ') for prem in prems]
    hyp_elements = hyp.split(' ')
    sentences = prems_elements + [hyp_elements]
    for i, sentence in enumerate(sentences):
        for j, element in enumerate(sentence):
            if 'tp' in element or 'interval' in element or 'time_unit' in element:
                if '!=' in element:
                    new_element = times[element.split('!=')[0]]
                else:
                    new_element = times[element]
                sentences[i][j] = new_element

    return [''.join(sentence) for sentence in sentences]


# 格フレームを用いた中間データ生成
def generate_middata_with_cf(out_file, templates, wordlist_dict, cf_dict, cf_keys, mid_data_num):
    is_test = True if 'test' in out_file else False
    time_unit_list = ['year', 'month', 'day', 'hour']
    ja2en_timeunit = {"年": "year", "月": "month", "日": "day", "時": "hour"}
    tu2intervalformat = {"year": "i年間", "month": "iヶ月間", "day": "i日間", "hour": "i時間"}
    tp_format_list = {"i時間": ["h時"],
                      "i日間": ["d日", "d日h時"],
                      "iヶ月間": ["m月", "m月d日", "m月d日h時"],
                      "i年間": ["y年", "y年m月", "y年m月d日", "y年m月d日h時"]}
    texts = []
    num = 0
    for template in tqdm(templates):
        data_per_template = 0
        ng_time_units = template['ng time unit'].split(',')
        test_break_flag = False
        for time_unit in time_unit_list:
            if time_unit in ng_time_units:
                continue
            else:
                ng_time_formats = [ng_time_unit.split('.') for ng_time_unit in ng_time_units]
            interval_format = tu2intervalformat[time_unit]
            break_flag = False
            for tp_format in tp_format_list[interval_format]:
                cat_ph = template['premise'] + template['hypothesis']
                if 'tp' not in cat_ph and break_flag:
                    break
                if is_test and test_break_flag:
                    break
                if [time_unit, ja2en_timeunit[tp_format[-1]]] in ng_time_formats:
                    continue
                break_flag = True
                test_break_flag = True

                time_spans = ['random', 'short']
                if 'tp' not in cat_ph and 'interval' not in cat_ph:
                    time_spans = ['None']

                ans_list = []
                for time_span in time_spans:
                    for _ in range(mid_data_num * 20):
                        prems = re.split("(?<=。)", template['premise'])[:-1]
                        prems_elements = [prem.split(' ') for prem in prems]
                        hyp_elements = template['hypothesis'].split(' ')
                        sentences = prems_elements + [hyp_elements]
                        time = generate_time(sum(sentences, []), tp_format, interval_format, time_span)
                        ans_list.append(generate_ans(time, template))
                        ans_list = list(set(ans_list))

                    data_per_template += mid_data_num * len(ans_list)

        test_verb = []
        test_cf = []
        for _ in range(data_per_template):
            loop = 0
            valid_text = False
            while True:
                loop += 1
                sentences, cf = generate_sentence_with_cf(template, wordlist_dict, cf_dict, cf_keys, is_test)
                if is_test and loop > 100:
                    break
                if is_test and loop < 50 and cf.split(',')[0] in test_verb:
                    continue
                if is_test and loop < 100 and cf in test_cf:
                    continue
                test_verb.append(cf.split(',')[0])
                test_cf.append(cf)
                valid_text = True
                break

            if valid_text:
                num += 1
                text = {'num': num,
                        'premise': ''.join(sentences[:-1]),
                        'hypothesis': sentences[-1],
                        'template_num': template['id'],
                        'category': template['category'],
                        'cf': cf}
                texts.append(text)

    pd.DataFrame(texts).to_csv(out_file + '.tsv', sep='\t', index=False)
    return


# 格フレームを用いた最終データ生成
def generate_dataset_with_cf(in_file, out_file, templates, data_num):
    is_test = True if 'test' in out_file else False
    time_unit_list = ['year', 'month', 'day', 'hour']
    ja2en_timeunit = {"年": "year", "月": "month", "日": "day", "時": "hour"}
    tu2intervalformat = {"year": "i年間", "month": "iヶ月間", "day": "i日間", "hour": "i時間"}
    tp_format_list = {"i時間": ["h時"],
                      "i日間": ["d日", "d日h時"],
                      "iヶ月間": ["m月", "m月d日", "m月d日h時"],
                      "i年間": ["y年", "y年m月", "y年m月d日", "y年m月d日h時"]}

    texts = pd.read_csv(in_file + '.tsv', sep='\t').to_dict('records')
    template2texts = defaultdict(list)
    for text in texts:
        template2texts[text['template_num']].append(text)

    comp_texts = []
    num = 0
    for template in tqdm(templates):
        ng_time_units = template['ng time unit'].split(',')
        i = 0
        for time_unit in time_unit_list:
            if time_unit in ng_time_units:
                continue
            else:
                ng_time_formats = [ng_time_unit.split('.') for ng_time_unit in ng_time_units]
            interval_format = tu2intervalformat[time_unit]
            break_flag = False
            for tp_format in tp_format_list[interval_format]:
                cat_ph = template['premise'] + template['hypothesis']
                if 'tp' not in cat_ph and break_flag:
                    continue
                break_flag = True

                time_spans = ['random', 'short']
                if 'tp' in cat_ph:
                    time_format = re.sub(r"[a-zA-Z]", "", tp_format)
                elif 'interval' in cat_ph:
                    time_format = interval_format[-2:]
                elif 'time_unit' in cat_ph:
                    time_format = interval_format[-2:]
                    time_spans = ['None']
                else:
                    time_format = "None"
                    time_spans = ['None']

                if is_test and time_format != template['test time format']:
                    continue
                if [time_unit, ja2en_timeunit[tp_format[-1]]] in ng_time_formats:
                    continue

                # 1つのテンプレートあたりいくつのインスタンスを生成するか
                for time_span in time_spans:
                    times_dic = {'entailment': [], 'contradiction': [], 'neutral': []}
                    for _ in range(400):
                        prems = re.split("(?<=。)", template['premise'])[:-1]
                        prems_elements = [prem.split(' ') for prem in prems]
                        hyp_elements = template['hypothesis'].split(' ')
                        sentences = prems_elements + [hyp_elements]
                        time = generate_time(sum(sentences, []), tp_format, interval_format, time_span)
                        ans = generate_ans(time, template)
                        time_cat = ','.join(time.values())
                        zero_flag = True if re.search('([^\\d]0[月,日])|(-1時)|(^0[月,日])', time_cat) else False
                        if len(times_dic[ans]) < data_num and not zero_flag:
                            times_dic[ans].append(time)

                    for ans in times_dic:
                        if times_dic[ans] == []:
                            continue
                        for j in range(data_num):
                            masked_sentences = template2texts[template['id']][i]
                            sentences = fill_times(masked_sentences, times_dic[ans][j])
                            i += 1
                            num += 1
                            comp_text = {'num': num,
                                         'premise': ''.join(sentences[:-1]),
                                         'hypothesis': sentences[-1],
                                         'gold_label': ans,
                                         'template_num': template['id'],
                                         'time_format': time_format,
                                         'time_span': time_span,
                                         'category': template['category']}
                            comp_texts.append(comp_text)

    pd.DataFrame(comp_texts).to_csv(out_file + '.tsv', sep='\t', index=False)
    labels = [comp_text['gold_label'] for comp_text in comp_texts]
    print(len(labels), end='\t')
    print(collections.Counter(labels))
    return


def split_dataset(templates, ver, split, seed=0, mode='random'):
    split_str = mode + '-' + str(split)
    if os.path.exists(f'dataset/ver_{ver}/train_{split_str}.tsv') \
            and os.path.exists(f'dataset/ver_{ver}/train_{split_str}_wakati.tsv'):
        return 0
    if mode == 'random':
        split_info = {'mode': mode, 'time_format': [], 'time_span': []}
        test_size = 1 - float('0.' + str(split))
        templates_train, templates_test = train_test_split(templates, test_size=test_size, random_state=seed)
        templates_train_idx = [template_train['id'] for template_train in templates_train]
        templates_test_idx = [template_test['id'] for template_test in templates_test]
        templates_train_idx.sort()
        templates_test_idx.sort()
        seed = '-' + str(seed)
        split_info['templates'] = templates_test_idx
        with open(f'dataset/split_info/{split_str}{seed}.txt', 'w') as out:
            pprint(split_info, stream=out, compact=True)
    elif mode == 'custom':
        seed = ''
        if not os.path.exists(f'dataset/split_info/{split_str}.txt'):
            return 0
        with open(f'dataset/split_info/{split_str}.txt', 'r') as infile:
            split_info = literal_eval(infile.read())
        templates_test_idx = split_info['templates']
        all_ids = [template['id'] for template in templates]
        templates_train_idx = sorted(list(set(all_ids) - set(templates_test_idx)))
    time_format_train = list(set(["None", "年間", "月間", "日間", "時間", "時", "日", "日時", "月", "月日", "月日時",
                             "年", "年月", "年月日", "年月日時"]) - set(split_info['time_format']))
    time_span_train = list(set(['None', 'random', 'short']) - set(split_info['time_span']))
    print(f"train template: {len(templates_train_idx)}, test template: {len(templates_test_idx)}")
    for suffix in ['', '_wakati']:
        all_problems = pd.read_csv(f'./dataset/ver_{ver}/train_all{suffix}.tsv', sep='\t')
        train_problems = all_problems.query(
            'template_num in @templates_train_idx & time_format in @time_format_train & time_span in @time_span_train')
        train_problems.to_csv(f'dataset/ver_{ver}/train_{split_str}{seed}{suffix}.tsv', sep='\t', index=False)
    return 0


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
    template_ver = 'ver_1_5'
    ver = '2_0'
    split_dict = {'random': [9, 8, 7], 'custom': ['template', 'timeformat', 'timespan']}
    do_split = {'random': False, 'custom': True}
    do_update = True
    do_wakati = True
    path = f'dataset/ver_{ver}/'

    os.makedirs(f'./dataset/ver_{ver}', exist_ok=True)
    templates, wordlist_dict, cf_dict, cf_keys = initialize(template_ver)

    if do_update:
        generate_middata_with_cf(path + 'train_all_temp', templates, wordlist_dict, cf_dict, cf_keys, 20)
        generate_dataset_with_cf(path + 'train_all_temp', path + 'train_all', templates, 10)
        generate_middata_with_cf(path + 'test_2_temp', templates, wordlist_dict, cf_dict, cf_keys, 5)
        generate_dataset_with_cf(path + 'test_filtered', path + 'test_2', templates, 2)

    if do_wakati:
        wakati(path + 'train_all')
        wakati(path + 'test_2')

    for mode in ['random', 'custom']:
        if not do_split[mode]:
            continue
        for split in split_dict[mode]:
            if mode == 'custom':
                for dif in ['easy', 'hard']:
                    split_dataset(templates, ver, str(split) + '-' + dif, 0, mode)
            else:
                split_dataset(templates, ver, split, 0, mode)


if __name__ == '__main__':
    tagger = MeCab.Tagger("-p")
    janome_tokenizer = Tokenizer()
    kotodama.setSegmentationEngine(kotodama.SegmentationEngine.JANOME, janome_tokenizer)
    kotodama.disableError(Juman())
    main()
