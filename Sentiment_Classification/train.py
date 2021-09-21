import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from torchmetrics import Accuracy, F1, Recall, Precision

from model import *
from dataloader import *

import warnings
import argparse
from common import str2bool

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Naver Movie Review dataset Classification')
parser.add_argument('--data_directory',
                    help="dataset's directory",
                    default='./dataset', type=str)
# monologg/kobert skt/kobert-base-v1
parser.add_argument('--weight',
                    help="BERT model's weight",
                    default="ainize/klue-bert-base-mrc", type=str)
parser.add_argument('--wordvec_dim',
                    help="word vector's dim",
                    default=512, type=int)
parser.add_argument('--hidden_size',
                    help="lstm's hidden size",
                    default=1024, type=int)
parser.add_argument('--freeze',
                    help='freeze weight',
                    default=True, type=str2bool)
parser.add_argument('--n_class',
                    help='num of classes',
                    default=2, type=int)
parser.add_argument('--learning_rate',
                    help='learning rate',
                    default=1e-4, type=float)
parser.add_argument('--gpus',
                    help='gpu num',
                    default=[2])
parser.add_argument('--max_epochs',
                    help='max epochs',
                    default=10, type=int)
parser.add_argument('--batch_size',
                    help='batch size',
                    default=128, type=int)
parser.add_argument('--num_workers',
                    help='num workers',
                    default=4, type=int)
parser.add_argument('--max_length',
                    help='max length',
                    default=32, type=int)
parser.add_argument('--log_dir',
                    help='tensorboard log directory',
                    default='nsmc_log', type=str)
parser.add_argument('--log_name',
                    help='tensorboard log name',
                    type=str)
parser.add_argument('--ckpt_dir',
                    help='checkpoint file directory',
                    default='nsmc_ckpt', type=str)
parser.add_argument('--ckpt_name',
                    help='checkpoint file name',
                    type=str)
args = parser.parse_args()


class Classifier(LightningModule):
    def __init__(self,
                 model_=None,
                 lr=args.learning_rate,
                 num_classes=args.n_class):
        super().__init__()
        self.model = model_
        self.loss_fn = nn.NLLLoss()

        self.accuracy = Accuracy()
        self.f1 = F1(num_classes=num_classes)
        self.recall = Recall(num_classes=num_classes)
        self.pre = Precision(num_classes=num_classes)
        self.lr = lr

    def forward(self, input_ids, attention_masks, token_type_ids):
        return self.model(input_ids, attention_masks, token_type_ids)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # text, label = batch.text, batch.label
        input_ids, attention_masks, token_type_ids, label = batch
        y_hat = self(input_ids, attention_masks, token_type_ids)
        loss = self.loss_fn(y_hat, label)

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # text, label = batch.text, batch.label
        input_ids, attention_masks, token_type_ids, label = batch
        y_hat = self(input_ids, attention_masks, token_type_ids)
        loss = self.loss_fn(y_hat, label)

        self.accuracy(preds=y_hat, target=label)
        self.f1(preds=y_hat, target=label)
        self.pre(preds=y_hat, target=label)
        self.recall(preds=y_hat, target=label)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('Accuracy', self.accuracy, on_step=False, on_epoch=True)
        self.log('F1', self.f1, on_step=False, on_epoch=True)
        self.log('Precision', self.pre, on_step=False, on_epoch=True)
        self.log('Recall', self.recall, on_step=False, on_epoch=True)
        return {'loss': loss}


#######################
# pytorch lightning config
#######################

logger = TensorBoardLogger(
    save_dir=args.log_dir,
    name=args.log_name,
    default_hp_metric=False
)

checkpoint_callback = ModelCheckpoint(
    monitor='Accuracy',
    dirpath=args.ckpt_dir,
    filename=args.ckpt_name,
    mode='max'
)

early_stop_callback = EarlyStopping(
    monitor='Accuracy',
    min_delta=1e-4,
    patience=5,
    verbose=False,
    mode='max'
)

trainer = Trainer(max_epochs=args.max_epochs,
                  logger=logger,
                  gpus=args.gpus,
                  accelerator='ddp',
                  gradient_clip_val=3.,
                  gradient_clip_algorithm='norm',
                  plugins=DDPPlugin(find_unused_parameters=False),
                  callbacks=[early_stop_callback,
                             checkpoint_callback])

if __name__ == '__main__':
    # BaseLine
    # datamodule = BaseLineDataModule(
    #     batch_size=args.batch_size,
    #     max_length=args.max_length
    # )
    # classifier = Classifier(model_=BaseLineModelForSentimentClassification(
    #     vocab_size=len(datamodule),
    #     wordvec_dim=args.wordvec_dim,
    #     hidden_size=args.hidden_size,
    # ))

    # Bert
    model = BertModelForSentimentClassification(weight=args.weight,
                                                n_classes=args.n_class)
    classifier = Classifier(model_=model)
    datamodule = BERTDataModule(weight=args.weight,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                max_len=args.max_length)
    trainer.fit(model=classifier, datamodule=datamodule)
