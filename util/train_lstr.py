import argparse
import json
import os
import torch
from torch.nn.modules.transformer import F
import torch.utils.data as data_utils
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
import torch.nn as nn
import random
import pandas as pd
import numpy as np

from setup_data import ml_1m, ml_20m, steam, beauty

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class LstrTrainDataset(data_utils.Dataset):
    def __init__(self, dataset: str, max_len: int, mask_prob: float, mask_token: int, num_items: int, rng: random.Random, long_len: int):
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.train_df = pd.read_pickle(f'../data/lstr/{dataset}/seqlen-{max_len}/train.pkl')
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        self.long_len = long_len

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index: int):
        seq = self.train_df.seq[index]

        tokens: list[int] = []
        labels: list[int] = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        # May not need this since the sequences are already cut.
        # tokens = tokens[-self.max_len:]
        # labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        # Split the sequences into two parts.
        # Long and short sequence.
        long_tokens = tokens[:self.long_len]
        short_tokens = tokens[self.long_len:]

        return torch.LongTensor(long_tokens), torch.LongTensor(short_tokens), torch.LongTensor(labels)

class LstrEvalDataset(data_utils.Dataset):
    def __init__(self, dataset: str, answer: str, max_len: int, mask_token: int, long_len: int) -> None:
        self.train_df = pd.read_pickle(f'../data/lstr/{dataset}/seqlen-{max_len}/train.pkl')
        if answer == 'val':
            self.df = pd.read_pickle(f'../data/lstr/{dataset}/seqlen-{max_len}/val.pkl')
        elif answer == 'test':
            self.df = pd.read_pickle(f'../data/lstr/{dataset}/seqlen-{max_len}/test.pkl')
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = pd.read_pickle(f'../data/lstr/{dataset}/seqlen-{max_len}/negative.pkl')
        self.long_len = long_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        
        seq = self.train_df.iloc[index]['seq']
        answer = self.df.iloc[index]['seq']
        negs = self.negative_samples[index]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        long_seq = seq[:self.long_len]
        short_seq = seq[self.long_len:]

        return torch.LongTensor(long_seq), torch.LongTensor(short_seq), torch.LongTensor(candidates), torch.LongTensor(labels)

def fix_random_seed_as(random_seed: int) -> None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class PositionalEmbedding(pl.LightningModule):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class LstrEmbedding(pl.LightningModule):
    """
    Lstr Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of LstrEmbedding
    """

    def __init__(self, vocab_size: int, embed_size: int, max_len: int, dropout: float = 0.1) -> None:
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.token(sequence) + self.position(sequence)  # + self.segment(segment_label)
        return self.dropout(x)


def recalls_and_ndcgs_for_ks(scores: torch.Tensor, labels: torch.Tensor, ks: list[int]):
    metrics: dict[str, torch.NumberType] = {}

    # Why is this being done?
    # scores = scores
    # labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        """
            Recall at k is the proportion of relevant items found in the top-k recommendations.
            Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
        """
        numerator = hits.sum(1)
        denominator = torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
        metrics['Recall@%d' % k] = \
            (numerator / denominator).mean().cpu().item()

        """
            NDCG at k is the average of the normalized DCG scores of the top-k recommendations.
            NDCG@k = DCG@k / IDCG@k
            DCG@K = SUM( recc_i / log2(i + 1))
            IDCG@K = SUM( real_i / log2(i + 1))
        """
        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        # Discounted cumulative gain at k
        dcg = (hits * weights.to(hits.device)).sum(1)
        # Ideal discounted cumulative gain at k
        # idcg = (labels_float * weights.to(labels_float.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                            for n in answer_count]).to(dcg.device)
        # What is the above code doing
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


class LSTR(pl.LightningModule):
    def __init__(self, args: dotdict):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        self.args = args
        long_len: int = args.bert_long_len
        short_len: int = args.bert_short_len
        num_items: int = args.num_items
        n_layers: int = args.bert_num_blocks
        vocab_size: int = num_items + 2
        hidden: int = args.bert_hidden_units
        dropout: float = args.bert_dropout
        self.mask_prob = args.bert_mask_prob
        self.dataset = args.dataset
        self.item_count = num_items
        self.max_len = args.bert_max_len
        self.long_len = long_len
        self.heads: int = args.bert_num_heads
        self.hidden = hidden
        self.metric_ks = args.metric_ks
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        # embedding for Lstr, sum of positional, segment, token embeddings
        self.long_embedding = LstrEmbedding(
            vocab_size=vocab_size, embed_size=self.hidden, max_len=long_len, dropout=dropout)
        self.short_embedding = LstrEmbedding(
            vocab_size=vocab_size, embed_size=self.hidden, max_len=short_len, dropout=dropout)
        # multi-layers transformer blocks, deep network
        self.encoder = nn.TransformerEncoderLayer(hidden, self.heads, hidden * 4, batch_first=True, activation=F.gelu)
        self.transformer = nn.TransformerEncoder(self.encoder, n_layers)
        self.out = nn.Linear(self.hidden, num_items + 1)

    def forward(self, long_seq: torch.Tensor, short_seq: torch.Tensor) -> torch.Tensor:
        full_seq = torch.cat([long_seq, short_seq], dim=1)
        full_mask = (full_seq == 0).unsqueeze(1).repeat(self.heads, full_seq.size(1), 1)
        long_mask = (long_seq == 0).unsqueeze(1).repeat(self.heads, long_seq.size(1), 1)
        long_seq = self.long_embedding(long_seq)
        long_output: torch.Tensor = self.transformer(long_seq, long_mask)


        # short_mask = (short_seq == 0).unsqueeze(1).repeat(self.heads, short_seq.size(1), 1)
        short_seq = self.short_embedding(short_seq)
        seq = torch.cat([long_output, short_seq], dim=1)
        # short_output: torch.Tensor = self.transformer(short_seq, short_mask)
        output: torch.Tensor = self.transformer(seq, full_mask)

        # output = torch.cat([long_output, short_output], dim=1)

        # output = output.masked_fill(torch.isnan(output), 0)
        return self.out(output)

    def init_weights(self):
        pass

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch_size = batch[0].size(0)
        long_seq, short_seq, labels = batch
        logits: torch.Tensor = self(long_seq, short_seq)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss: torch.Tensor = self.ce(logits, labels)
        # log_data = {
        #     'state_dict': (self._create_state_dict()),
        #     "train/ce": loss.item()
        # }
        self.log(
            "train/ce", loss, on_step=True, on_epoch=False, prog_bar=True
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        long_seq, short_seq, candidates, labels = batch
        scores: torch.Tensor = self(long_seq, short_seq)
        scores = scores[:, -1, :]
        scores = scores.gather(1, candidates)
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

        return metrics

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.NumberType]:
        long_seq, short_seq, candidates, labels = batch
        scores: torch.Tensor = self(long_seq, short_seq)
        scores = scores[:, -1, :]
        scores = scores.gather(1, candidates)
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

        # self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def validation_epoch_end(self, outputs: list) -> None:
        # outputs is a list of dicts
        avg_metrics = {}
        for metric in outputs[0].keys():
            avg_metrics[metric] = torch.mean(torch.FloatTensor([x[metric] for x in outputs]))

        self.log_dict(avg_metrics, on_step=False, on_epoch=True, prog_bar=True)

        # Convert metric from tensor to scalar
        save_log = {}
        for metric in avg_metrics.keys():
            save_log[metric] = avg_metrics[metric].item()
        
        # Convert save_log to json
        # If folder does not exist, create it
        if not os.path.exists(f'./logs/{self.dataset}/seqlen-{self.max_len}'):
            os.makedirs(f'./logs/{self.dataset}/seqlen-{self.max_len}')
        with open(f'./logs/{self.dataset}/seqlen-{self.max_len}/{self.mask_prob}-val.json', 'w') as f:
            json.dump(save_log, f)

        
    
    def test_epoch_end(self, outputs: list[dict[str, torch.NumberType]]) -> None:
        # outputs is a list of dicts
        avg_metrics = {}
        for metric in outputs[0].keys():
            avg_metrics[metric] = torch.mean(torch.FloatTensor([x[metric] for x in outputs]))

        self.log_dict(avg_metrics, on_step=False, on_epoch=True, prog_bar=True)

        # Convert metric from tensor to scalar
        save_log = {}
        for metric in avg_metrics.keys():
            save_log[metric] = avg_metrics[metric].item()
        
        # Convert save_log to json
        # If folder does not exist, create it
        if not os.path.exists(f'./logs/{self.dataset}/seqlen-{self.max_len}'):
            os.makedirs(f'./logs/{self.dataset}/seqlen-{self.max_len}')
        with open(f'./logs/{self.dataset}/seqlen-{self.max_len}/{self.mask_prob}-test.json', 'w') as f:
            json.dump(save_log, f)

    def configure_optimizers(self):
        if self.args['optimizer'].lower() == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer'].lower() == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'], momentum=self.args['momentum'])
        else:
            raise ValueError('Optimizer not supported')

    def _create_state_dict(self):
        return {
            'model_state_dict': self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
    
    def setup(self, stage=None):
        self.train_dataset = LstrTrainDataset(self.dataset, self.max_len, self.mask_prob, self.item_count+1, self.item_count, random.Random(0), self.long_len)
        self.val_dataset = LstrEvalDataset(self.dataset, 'val', self.max_len, self.item_count+1, self.long_len)
        self.test_dataset = LstrEvalDataset(self.dataset, 'test', self.max_len, self.item_count+1, self.long_len)

    
    def train_dataloader(self):
        return data_utils.DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data_utils.DataLoader(
            self.val_dataset,
            batch_size=128,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data_utils.DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            pin_memory=True,
        )

sequence_lengths = {
    # 20: [15, 13],
    # 26: [20, 16],
    50: [38, 30],
    # 100: [75, 68],
    # 200: [150, 120],
}


def test_different_sequence_lengths(dataset: str):

    for sequence_length in sequence_lengths.keys():
        # Start data preprocessing.
        print(f'Preprocessing data for sequence length {sequence_length}...')
        # item_count = ml_1m(sequence_length)
        if dataset == 'ml-1m':
            item_count = ml_1m(sequence_length)
        elif dataset == 'ml-20m':
            item_count = ml_20m(sequence_length)
        elif dataset == 'steam':
            item_count = steam(sequence_length)
        else:
            item_count = beauty(sequence_length)
        for long_length in sequence_lengths[sequence_length]:
            args = {
                'dataset': dataset,
                'bert_max_len': sequence_length,
                'bert_long_len': long_length,
                'bert_short_len': sequence_length - long_length,
                'num_items': item_count,
                'bert_num_blocks': 2,
                'bert_num_heads': 4,
                'bert_hidden_units': 64,
                'bert_dropout': 0.1,
                'model_init_seed': 42,
                'bert_mask_prob': 0.2,
                'metric_ks': [1, 5, 10],
                'lr': 0.001,
                'weight_decay': 0.0,
                'optimizer': 'Adam',
            }

            args = dotdict(args)

            # Initialize the model
            model = LSTR(args)

            # Initialize a trainer
            print(f'⚡ Training model for sequence length {sequence_length} and long length {long_length}...')
            trainer = pl.Trainer(
                accelerator='gpu',
                devices=1,
                max_epochs=100,
            )

            # Train the model ⚡
            trainer.fit(model)

            
            

            # Test the model ⚡
            trainer.test(model)


def test_different_ps(dataset: str):
    sequence_length = 50
    long_length = 38
    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f'Preprocessing data for sequence length {sequence_length}...')
    if dataset == 'ml-1m':
        item_count = ml_1m(sequence_length)
    elif dataset == 'ml-20m':
        item_count = ml_20m(sequence_length)
    elif dataset == 'steam':
        item_count = steam(sequence_length)
    else:
        item_count = beauty(sequence_length)
    for p in ps:
        args = {
            'dataset': dataset,
            'bert_max_len': sequence_length,
            'bert_long_len': long_length,
            'bert_short_len': sequence_length - long_length,
            'num_items': item_count,
            'bert_num_blocks': 2,
            'bert_num_heads': 4,
            'bert_hidden_units': 64,
            'bert_dropout': 0.1,
            'model_init_seed': 42,
            'bert_mask_prob': p,
            'metric_ks': [1, 5, 10],
            'lr': 0.001,
            'weight_decay': 0.0,
            'optimizer': 'Adam',
        }

        args = dotdict(args)

        # Initialize the model
        model = LSTR(args)

        # Initialize a trainer
        print(f'⚡ Training model for sequence length {sequence_length} and p: {p}...')
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=100,
        )

        # Train the model ⚡
        trainer.fit(model)
            
        # Test the model ⚡
        trainer.test(model)


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')

    # Test different sequence lengths with --vary-n
    parser.add_argument('--vary-n', action='store_true')

    # Test different p values with --vary-p
    parser.add_argument('--vary-p', action='store_true')

    args = parser.parse_args()

    if args.vary_n:
        test_different_sequence_lengths(dataset=args.dataset)
    elif args.vary_p:
        test_different_ps(dataset=args.dataset)
    