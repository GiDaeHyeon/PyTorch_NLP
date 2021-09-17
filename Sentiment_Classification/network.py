from torch.nn import Module, Linear, Dropout, LeakyReLU, Sequential
from torch.nn.functional import log_softmax
from transformers import BertModel


class BertModelForSentimentClassification(Module):
    def __init__(self,
                 weight=None,
                 n_classes=None,
                 freeze=None):
        super().__init__()
        self.model = BertModel.from_pretrained(weight)

        if freeze:
            for param in self.model.encoder.parameters():
                param.require_grad = False

        self.clf = Sequential(
            Linear(768, 768),
            Dropout(p=.3),
            LeakyReLU(),
            Linear(768, 256),
            Dropout(p=.3),
            LeakyReLU(),
            Linear(256, n_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        x = self.clf(x.pooler_output)
        return log_softmax(x)
