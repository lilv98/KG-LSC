import torch
import pdb
import tqdm

def get_words():
    words = set()
    with open('./corpus1/lemma/ccoha1.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            for i in range(len(line)):
                words.add(line[i])
    with open('./corpus2/lemma/ccoha2.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            for i in range(len(line)):
                words.add(line[i])
    return {v: k for k, v in enumerate(words)}
    
    # with open('./corpus' + str(id) + '/lemma/ccoha' + str(id) + '.txt') as f:
    #     for line in tqdm.tqdm(f):
    #         line = line.strip().split(' ')
    #         for i in range(len(line) - 1):
    #             triple = (words[line[i]], 0, words[line[i + 1]])
    #             if triple in triples:
    #                 triples[triple] += 1
    #             else:
    #                 triples[triple] = 1
    
    # data = []
    # for v, k in enumerate(triples):
    #     data.append([k[0], k[1], k[2], v])
    # data = torch.tensor(data)
    
    # return words, triples, data



if __name__ == '__main__':
    words_1, triples_1, data_1 = read_corpus(id=1)
    words_2, triples_2, data_2 = read_corpus(id=2)
    pdb.set_trace()
    