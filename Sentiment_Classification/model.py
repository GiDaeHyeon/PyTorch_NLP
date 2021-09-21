from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LeakyReLU, Sequential, LogSoftmax
from transformers import AutoModel
from common import weight_he_init, weight_xavier_init


class BertModelForSentimentClassification(Module):
    def __init__(self,
                 weight=None,
                 n_classes=None,
                 freeze=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(weight)

        if freeze:
            for param in self.model.parameters():
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
        self.activation = LogSoftmax()

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        x = self.clf(x.pooler_output)
        y_hat = self.activation(x)
        return y_hat


class BaseLineModelForSentimentClassification(Module):
    def __init__(self,
                 vocab_size=None,
                 wordvec_dim=None,
                 hidden_size=None,
                 num_layers=4,
                 dropout_p=.3,
                 bidirectional=True):
        super().__init__()
        self.embedding = Embedding(vocab_size, wordvec_dim)
        self.lstm = LSTM(input_size=wordvec_dim,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout_p,
                         batch_first=True,
                         bidirectional=bidirectional)
        self.activation = LogSoftmax(dim=-1)

        if bidirectional:
            hidden_size *= 2

        self.clf = Sequential(
            Linear(hidden_size, 768),
            Dropout(p=.3),
            LeakyReLU(),
            Linear(768, 256),
            Dropout(p=.3),
            LeakyReLU(),
            Linear(256, 2)
        )
        # weight_he_init(self.clf)
        # weight_xavier_init(self.clf)

    def forward(self, x):
        wordvec = self.embedding(x)
        hidden_state, _ = self.lstm(wordvec)
        logit = self.clf(hidden_state)
        y_hat = self.activation(logit[:, -1])
        return y_hat
