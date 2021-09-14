import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from torchmetrics import Accuracy

from network import BertModelForSentimentClassification
from dataloader import BERTDataModule
from config import LEARNING_RATE, NUM_CLASSES, GRADIENT_CLIP_ALGORITHM, GRADIENT_CLIP_VAL, GPUS, MAX_EPOCHS

import warnings
warnings.filterwarnings('ignore')


class Network(LightningModule):
    def __init__(self,
                 model=BertModelForSentimentClassification(),
                 loss_fn=nn.NLLLoss(),
                 lr=LEARNING_RATE,):
        super(Network, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.metrics = Accuracy(num_classes=NUM_CLASSES)

    def forward(self, input_ids, input_mask, input_token):
        return self.model(input_ids, input_mask, input_token)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, input_tokens, labels = batch
        y_hat = self(input_ids,
                     input_mask,
                     input_tokens)
        loss = self.loss_fn(y_hat, labels)
        self.log('train_loss', loss)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, input_tokens, labels = batch
        y_hat = self(input_ids,
                     input_mask,
                     input_tokens)
        loss = self.loss_fn(y_hat, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.metrics(y_hat, labels)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        self.log('acc', self.metrics.compute())


#######################
# pytorch lightning config
#######################

logger = TensorBoardLogger(
                            save_dir='nmsc',
                            name='nmsc log',
                            default_hp_metric=False
                          )

checkpoint_callback = ModelCheckpoint(
                                        monitor='acc',
                                        dirpath='nmsc_ckpt',
                                        filename='nmsc_ckpt_file',
                                        mode='max'
                                     )

early_stop_callback = EarlyStopping(
                                        monitor='acc',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=False,
                                        mode='max'
                                   )

trainer = Trainer(max_epochs=MAX_EPOCHS,
                  logger=logger,
                  gpus=GPUS,
                  accelerator='ddp',
                  num_sanity_val_steps=0,
                  gradient_clip_val=GRADIENT_CLIP_VAL,
                  gradient_clip_algorithm=GRADIENT_CLIP_ALGORITHM,
                  plugins=DDPPlugin(find_unused_parameters=False),
                  callbacks=[early_stop_callback,
                             checkpoint_callback])

if __name__ == '__main__':
    model = Network()
    datamodule = BERTDataModule()
    trainer.fit(model=model, datamodule=datamodule)
