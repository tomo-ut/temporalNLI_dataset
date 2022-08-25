import pickle

with open('./external_data/pth20210305beta/data.csv', 'r') as infile:
    lines = infile.read().splitlines()
lines = [line.split(',') for line in lines]

inds = []
for i in range(len(lines[0])):
    if '事例' in lines[0][i]:
        inds.append(i)

wordlist = set()
for line in lines[1:]:
    for ind in inds:
        if line[i]:
            suffix_len = len(line[ind-1])
            if line[ind-1] == line[ind][-suffix_len:]:
                wordlist.add(line[ind][:-suffix_len])
            else:
                wordlist.add(line[ind])


with open('./vocab_list/wordlist.pickle', 'wb') as f:
    pickle.dump(wordlist, f)

with open('./vocab_list/wordlist.txt', 'w') as outfile:
    for word in sorted(list(wordlist)):
        outfile.write(word + '\n')
