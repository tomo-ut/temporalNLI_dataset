import pickle

with open('./external_data/kyoto-univ-web-cf-2.0/cf_dict.pickle', 'rb') as f:
    cf_dict = pickle.load(f)

vocab_set = set()

for v1 in cf_dict.values():
    for v2 in v1.values():
        for v3 in v2.values():
            for word in v3:
                vocab_set.add(word)

with open('./vocab_list/vocablist.txt', 'w') as outfile:
    for w in sorted(list(vocab_set)):
        outfile.write(w + '\n')
