import torch
import pdb
import tqdm

def get_words():
    words = set()
    with open('./ccoha1.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            for i in range(len(line)):
                words.add(line[i])
    with open('./ccoha2.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            for i in range(len(line)):
                words.add(line[i])
    return {v: k for k, v in enumerate(words)}

def get_cooccur(words, id):
    triples = {}
    with open('./ccoha' + str(id) + '.txt') as f:
        for line in tqdm.tqdm(f):
            line = line.strip().split(' ')
            for i in range(len(line) - 1):
                triple = (words[line[i]], 0, words[line[i + 1]])
                if triple in triples:
                    triples[triple] += 1
                else:
                    triples[triple] = 1
    
    data = []
    for _, k in enumerate(triples):
        data.append([k[0], k[1], k[2], triples[k]])
    data = torch.tensor(data)
    
    return triples, data


if __name__ == '__main__':
    e_dict = get_words()
    triples_1, data_1 = get_cooccur(e_dict, id=1)
    triples_2, data_2 = get_cooccur(e_dict, id=2)
    triples_common = torch.tensor(list(set(triples_1) & set(triples_2)))
    pdb.set_trace()
    