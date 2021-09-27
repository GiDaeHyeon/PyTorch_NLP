import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from pytorch_lightning import LightningDataModule
from konlpy.tag import Mecab

from torchtext.legacy import data
from torchtext.legacy.data import TabularDataset, Iterator


class BERTDataset(Dataset):
    def __init__(self,
                 data_dir='./dataset',
                 weight=None,
                 mode='train',
                 max_len=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(weight)

        with open(f'{data_dir}/ratings_{mode}.txt', 'r') as d:
            self.data = d.readlines()[1:]
            d.close()

        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item, label = self.data[idx].split('\t')[1:]
        encoded_sent = self.tokenizer.encode_plus(text=item,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  truncation=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=True
                                                  )

        input_ids = torch.tensor(encoded_sent.get('input_ids'))
        attention_masks = torch.tensor(encoded_sent.get('attention_mask'))
        token_type_ids = torch.tensor(encoded_sent.get('token_type_ids'))

        return input_ids, attention_masks, token_type_ids, torch.tensor(int(label)).long()


class BERTDataModule(LightningDataModule):
    def __init__(self,
                 weight=None,
                 batch_size=None,
                 num_workers=None,
                 max_len=None):
        super().__init__()
        self.trainset = BERTDataset(weight=weight,
                                    max_len=max_len)
        self.valset = BERTDataset(weight=weight,
                                   mode='test',
                                   max_len=max_len)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.trainset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True)


class BaseLineDataModule(LightningDataModule):
    def __init__(self,
                 batch_size=None,
                 max_length=None):
        super().__init__()
        self.tokenizer = Mecab()
        self.batch_size = batch_size
        self.ID = data.Field(sequential=False,
                             use_vocab=False,
                             is_target=False)
        self.TEXT = data.Field(sequential=True,
                               use_vocab=True,
                               tokenize=self.tokenizer.morphs,
                               lower=False,
                               batch_first=True,
                               fix_length=max_length)
        self.LABEL = data.Field(sequential=False,
                                use_vocab=False,
                                is_target=True)
        self.train_data, self.val_data = TabularDataset.splits(path='./dataset',
                                                               train='ratings_train.txt',
                                                               test='ratings_test.txt',
                                                               format='tsv',
                                                               fields=[('id', self.ID),
                                                                       ('text', self.TEXT),
                                                                       ('label', self.LABEL)],
                                                               skip_header=True)
        self.TEXT.build_vocab(self.train_data, min_freq=5)

    def __len__(self):
        return len(self.TEXT.vocab)

    def train_dataloader(self):
        return Iterator(dataset=self.train_data,
                        batch_size=self.batch_size,
                        shuffle=True)

    def val_dataloader(self):
        return Iterator(dataset=self.val_data,
                        batch_size=self.batch_size,
                        shuffle=False)
