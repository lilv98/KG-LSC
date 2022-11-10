import torch
import pdb
import tqdm
import argparse
import numpy as np
import random
import os
import pickle

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_words(cfg):
    if os.path.exists(cfg.save_root + '/e_dict.pkl'):
        e_dict = load_obj(cfg.save_root + '/e_dict.pkl')
    else:
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
        e_dict = {v: k for k, v in enumerate(words)}
        save_obj(e_dict, cfg.save_root + '/e_dict.pkl')
    return e_dict

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


class KnowledgeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, e_dict, r_dict, data, num_ng, common):
        super().__init__()
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.num_ng = num_ng
        if common:
            data = torch.cat([data, torch.ones(len(data), 1).long()], dim=-1)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail, freq = self.data[idx]
        negs = torch.tensor(np.random.choice(len(self.e_dict), self.num_ng))
        neg_t, neg_h = negs[:self.num_ng // 2].unsqueeze(dim=1), negs[self.num_ng // 2:].unsqueeze(dim=1)
        neg_t = torch.cat([torch.tensor([head, rel]).expand(self.num_ng // 2, -1), neg_t, torch.tensor([freq]).expand(self.num_ng // 2, -1)], dim=1)
        neg_h = torch.cat([neg_h, torch.tensor([rel, tail]).expand(self.num_ng // 2, -1), torch.tensor([freq]).expand(self.num_ng // 2, -1)], dim=1)
        sample = torch.cat([torch.tensor([head, rel, tail, freq]).unsqueeze(0), neg_t, neg_h], dim=0)
        return sample


class KnowledgeGraphEmbeddingModel(torch.nn.Module):
    def __init__(self, e_dict, r_dict, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.base_model = cfg.base_model
        self.e_embedding = torch.nn.Embedding(len(e_dict), cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(len(r_dict), cfg.emb_dim)
        self.scoring_fct_norm = cfg.scoring_fct_norm
        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)

    def _DistMult(self, h_emb, r_emb, t_emb):
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def _TransE(self, h_emb, r_emb, t_emb):
        return - torch.norm(h_emb + r_emb - t_emb, p=self.scoring_fct_norm, dim=-1)

    def forward(self, data):
        h_emb = self.e_embedding(data[:, :, 0])
        r_emb = self.r_embedding(data[:, :, 1])
        t_emb = self.e_embedding(data[:, :, 2])
        freq = data[:, 0, -1]
        if self.base_model == 'DistMult':
            return self._DistMult(h_emb, r_emb, t_emb), freq
        elif self.base_model == 'TransE':
            return self._TransE(h_emb, r_emb, t_emb), freq
        else:
            raise ValueError

    def get_loss(self, data):
        pred, freq = self.forward(data)
        pointwise = - torch.nn.functional.logsigmoid(pred[:, 0].unsqueeze(dim=-1) - pred[:, 1:])
        weights = (freq / freq.sum()).unsqueeze(dim=-1)
        return (pointwise * weights).sum() / pointwise.size(dim=-1)

def get_test_data():
    ret = []
    with open('./binary.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            ret.append([line[0], int(line[1])])

    with open('./graded.txt') as f:
        counter = 0
        for line in f:
            line = line.strip().split('\t')
            ret[counter].append(float(line[1]))
            counter += 1
    
    return ret

def js_div(fs_1, fs_2):
    fs_1_normalized = torch.nn.functional.normalize(fs_1, p=1, dim=-1)
    fs_2_normalized = torch.nn.functional.normalize(fs_2, p=1, dim=-1)
    M = 0.5 * (fs_1_normalized + fs_2_normalized)
    kl_1 = torch.nn.functional.kl_div(torch.log(fs_1_normalized + 1e-10), M, reduction='none', log_target=False).sum(dim=-1)
    kl_2 = torch.nn.functional.kl_div(torch.log(fs_2_normalized + 1e-10), M, reduction='none', log_target=False).sum(dim=-1)
    return 0.5 * (kl_1 + kl_2)

def evaluate(e_dict, r_dict, cfg):
    model_1 = KnowledgeGraphEmbeddingModel(e_dict, r_dict, cfg)
    model_2 = KnowledgeGraphEmbeddingModel(e_dict, r_dict, cfg)
    model_1.load_state_dict(torch.load(cfg.save_root + 'corpus_1_' + str(cfg.epochs_respective) + '.pt'))
    model_2.load_state_dict(torch.load(cfg.save_root + 'corpus_2_' + str(cfg.epochs_respective) + '.pt'))
    model_1.eval()
    model_2.eval()
    test_data = get_test_data()
    
    cossim = torch.nn.CosineSimilarity()
    results = []
    for entry in test_data:
        id = e_dict[entry[0]]
        emb_1 = model_1.e_embedding.weight[id]
        emb_2 = model_2.e_embedding.weight[id]
        pred = torch.sigmoid(cossim(emb_1.unsqueeze(dim=0), emb_2.unsqueeze(dim=0)))
        results.append([pred, entry[1], entry[2]])
    results = torch.tensor(results)
    binary_result = ((results[:, 0] > 0.5) == results[:, 1]).sum() / len(results)
    print(f'Task 1: {round(binary_result.item(), 3)}')
    graded_result = js_div(results[:, 0], results[:, -1])
    print(f'Task 2: {round(graded_result.item(), 3)}')

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Tunable
    parser.add_argument('--bs', default=4096, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--base_model', default='DistMult', type=str)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--do', default=0.2, type=float)
    parser.add_argument('--scoring_fct_norm', default=2, type=int)
    # Untunable
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs_common', default=100, type=int)
    parser.add_argument('--epochs_respective', default=20, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_root', default='./', type=str)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    model_log = 'Configurations:\n'
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    
    e_dict = get_words(cfg)
    r_dict = {'cooccur': 0}
    triples_1, data_1 = get_cooccur(e_dict, id=1)
    triples_2, data_2 = get_cooccur(e_dict, id=2)
    triples_common = torch.tensor(list(set(triples_1) & set(triples_2)))
    
    train_dataset_common = KnowledgeGraphDataset(e_dict, r_dict, triples_common, cfg.num_ng, common=True)
    train_dataset_1 = KnowledgeGraphDataset(e_dict, r_dict, data_1, cfg.num_ng, common=False)
    train_dataset_2 = KnowledgeGraphDataset(e_dict, r_dict, data_2, cfg.num_ng, common=False)

    train_dataloader_common = torch.utils.data.DataLoader(dataset=train_dataset_common,
                                                            batch_size=cfg.bs,
                                                            num_workers=cfg.num_workers,
                                                            shuffle=True,
                                                            drop_last=True)
    train_dataloader_1 = torch.utils.data.DataLoader(dataset=train_dataset_1,
                                                            batch_size=cfg.bs,
                                                            num_workers=cfg.num_workers,
                                                            shuffle=True,
                                                            drop_last=True)
    train_dataloader_2 = torch.utils.data.DataLoader(dataset=train_dataset_2,
                                                            batch_size=cfg.bs,
                                                            num_workers=cfg.num_workers,
                                                            shuffle=True,
                                                            drop_last=True)
    
    model = KnowledgeGraphEmbeddingModel(e_dict, r_dict, cfg)
    model = model.to(device)
    optimizer_common = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    for epoch in range(cfg.epochs_common):
        print(f'Common -- Epoch {epoch + 1}:')
        model.train()
        avg_loss = []
        for batch in tqdm.tqdm(train_dataloader_common):
            batch = batch.to(device)
            loss = model.get_loss(batch)
            optimizer_common.zero_grad()
            loss.backward()
            optimizer_common.step()
            avg_loss.append(loss.item())
        print(f'Loss: {round(sum(avg_loss) / len(avg_loss), 6)}')
    torch.save(model.state_dict(), cfg.save_root + 'common_' + str(cfg.epochs_common) + '.pt')
    
    model_1 = KnowledgeGraphEmbeddingModel(e_dict, r_dict, cfg)
    model_1 = model_1.to(device)
    model_1.load_state_dict(torch.load(cfg.save_root + 'common_' + str(cfg.epochs_common) + '.pt'))
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    for epoch in range(cfg.epochs_respective):
        print(f'Corpus 1 -- Epoch {epoch + 1}:')
        model_1.train()
        avg_loss = []
        for batch in tqdm.tqdm(train_dataloader_1):
            batch = batch.to(device)
            loss = model_1.get_loss(batch)
            optimizer_1.zero_grad()
            loss.backward()
            optimizer_1.step()
            avg_loss.append(loss.item())
        print(f'Loss: {round(sum(avg_loss) / len(avg_loss), 6)}')
    torch.save(model_1.state_dict(), cfg.save_root + 'corpus_1_' + str(cfg.epochs_respective) + '.pt')
    
    model_2 = KnowledgeGraphEmbeddingModel(e_dict, r_dict, cfg)
    model_2 = model_2.to(device)
    model_2.load_state_dict(torch.load(cfg.save_root + 'common_' + str(cfg.epochs_common) + '.pt'))
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    for epoch in range(cfg.epochs_respective):
        print(f'Corpus 2 -- Epoch {epoch + 1}:')
        model_2.train()
        avg_loss = []
        for batch in tqdm.tqdm(train_dataloader_2):
            batch = batch.to(device)
            loss = model_2.get_loss(batch)
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()
            avg_loss.append(loss.item())
        print(f'Loss: {round(sum(avg_loss) / len(avg_loss), 6)}')
    torch.save(model_2.state_dict(), cfg.save_root + 'corpus_2_' + str(cfg.epochs_respective) + '.pt')
    
    evaluate(e_dict, r_dict, cfg)
    