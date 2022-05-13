import random


def assign_words(element):
    if 'agent' in element:
        return(random.choice(agent_list))
    if 'np' in element:
        return(random.choice(np_list))
    elif 'tp' in element:
        return(random.choice([str(random.randint(0, 23)) + '時', str(random.randint(1, 12)) + '月']))
    elif 'interval' in element:
        return(str(random.randint(1, 5)) + random.choice(['時間', '日間', '年間']))
    elif 'place' in element:
        return(random.choice(place_list))
    else:
        return(element)


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

texts = []

for template in templates:
    for _ in range(1):
        memo = {}
        (prem, hyp) = template
        prem_elements = prem.split(' ')
        hyp_elements = hyp.split(' ')
        for i, element in enumerate(prem_elements):
            if element in memo:
                prem_elements[i] = memo[element]
            else:
                prem_elements[i] = assign_words(element)
                memo[element] = prem_elements[i]
        for i, element in enumerate(hyp_elements):
            if element in memo:
                hyp_elements[i] = memo[element]
            else:
                hyp_elements[i] = assign_words(element)
                memo[element] = hyp_elements[i]
        texts.append([''.join(prem_elements), ''.join(hyp_elements)])

with open('dataset.tsv', 'w') as outfile:
    outfile.write('premise\thypothesis\n')
    for text in texts:
        outfile.write(f'{text[0]}\t{text[1]}\n')