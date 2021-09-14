import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast
from config import DATA_DIR, MAX_LEN, WEIGHT, BATCH_SIZE, NUM_WORKERS, VAL_RATIO, RANDOM_STATE
from pytorch_lightning import LightningDataModule


class BERTDataset(Dataset):
    def __init__(self,
                 data_dir=DATA_DIR,
                 mode='train',
                 max_len=MAX_LEN):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(WEIGHT)

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

        return input_ids, attention_masks, token_type_ids, int(label)


class BERTDataModule(LightningDataModule):
    def __init__(self,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):
        super().__init__()

        self.trainset = BERTDataset(mode='train')
        self.valset = BERTDataset(mode='test')
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
