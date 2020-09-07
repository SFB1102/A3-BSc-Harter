from allennlp import *

from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel

from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np
from allennlp.training.optimizers import DenseSparseAdam

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
#from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile

from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure

from allennlp.data.iterators import BasicIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor



class EDDatasetReader(DatasetReader):
    def __init__(self, token_indexers)->None:
        super().__init__(lazy=False)
        #self.token_indexers = token_indexers or {"tokens":SingleIdTokenIndexer()}
        self.token_indexers = token_indexers
    def text_to_instance(self,tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        if tags:
            label_field = SequenceLabelField(labels=tags,sequence_field=sentence_field)
            fields["labels"] = label_field
        return Instance(fields)
    def _read(self,file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                #pairs = line.strip().split()
                pairs = line.split()
                #print(pairs)
                try:
                    sentence, tags = zip(*(pair.split("###") for pair in pairs))
                    print(sentence)
                    yield self.text_to_instance([Token(word) for word in sentence], tags)
                except ValueError:
                    print(pairs)
                


class EventTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 #encoder: Seq2SeqEncoder,
                 encoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
     
        
        self.accuracy = CategoricalAccuracy()
        self.fBeta = FBetaMeasure()
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
            self.fBeta(tag_logits,labels,mask)
        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
    
reader = EDDatasetReader({"bert" : PretrainedBertIndexer(pretrained_model="bert-base-uncased")})
validation_dataset = reader.read("InScriptESDs.txt")
train_dataset = reader.read("DeScript_train.txt")
vocab = Vocabulary.from_instances(validation_dataset+train_dataset)

bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-uncased")
word_embeddings = BasicTextFieldEmbedder({"bert":bert_embedder},allow_unmatched_keys = True,embedder_to_indexer_map={"bert":["bert","bert-offsets"]})


HIDDEN_DIM = 50
#lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(hidden_size = HIDDEN_DIM,input_size=word_embeddings.get_output_dim(),batch_first=True,bias=False,bidirectional=True))
gru = PytorchSeq2SeqWrapper(torch.nn.GRU(hidden_size = HIDDEN_DIM,input_size=word_embeddings.get_output_dim(),batch_first=True,bidirectional=True))
#rnn = PytorchSeq2SeqWrapper(torch.nn.RNN(hidden_size = HIDDEN_DIM,input_size=word_embeddings.get_output_dim(),bias=False,batch_first=True,bidirectional=True))
model = EventTagger(word_embeddings, gru, vocab)
#optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = DenseSparseAdam(model.parameters(),lr=0.001)
iterator = BasicIterator(batch_size=1)
iterator.index_with(vocab)



trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  #patience=5,
                  num_epochs=10)
trainer.train()
performanz = model.fBeta.get_metric()
precisions = performanz["precision"]
recalls = performanz["recall"]
fscores = performanz["fscore"]
precision,recall,fscore = 0,0,0
for p in precisions:
    precision+=p
precision/=len(precisions)
for r in recalls:
    recall+=r
recall/=len(recalls)
for f in fscores:
    fscore+=f
fscore/=len(fscores)
print("precision:",precision,"recall:",recall,"fscore:",fscore)

weiter = input("Zum weiter machen klicken")

predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
richtig = 0
alle = 0
zeile = 1

verbtest = []
with open("evaluierungshilfe.txt","r") as valh:
    for line in valh:
        verbtest.append(line.strip().split())
with open("InScriptESDs.txt","r") as val:
    richtig = 0
    alle = 0
    scores = dict()
    for line in val:
        print(zeile)
        zeile+=1
        labels = []
        satz = ""
        for token in line.split():
            (word,label) = token.split("###")
            satz += word+" "
            labels.append(label)
        tag_logits = predictor.predict(satz)['tag_logits']
        tag_ids = np.argmax(tag_logits, axis=-1)
        liste = [model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]
        print(len(liste),len(labels),len(verbtest[zeile-2]))
        #print(len(liste),len(labels),len(verbtest))
        for tag, gold, verb, wort in zip(liste, labels,verbtest[zeile-2],satz.split()):
            print(tag,gold,verb,wort)
            if verb=="V":
                alle += 1
                if not tag in scores:
                    scores[tag] = dict()
                    scores[tag]["truepositiv"]=0
                    scores[tag]["falsepositiv"]=0
                    scores[tag]["falsenegativ"]=0
                if not gold in scores:
                    scores[gold] = dict()
                    scores[gold]["truepositiv"]=0
                    scores[gold]["falsepositiv"]=0
                    scores[gold]["falsenegativ"]=0
                if tag == gold:
                    richtig+=1
                    scores[gold]["truepositiv"]+=1
                else:
                    scores[gold]["falsenegativ"] += 1
                    scores[tag]["falsepositiv"] += 1
                

accuracy = richtig/alle

precision = 0
recall = 0
f1 = 0
for label in scores:
    if scores[label]["truepositiv"] == 0:
        if scores[label]["falsepositiv"] == 0:
            p = 1
        else:
            p = 0
        if scores[label]["falsenegativ"] == 0:
            r = 1
        else:
            r = 0
    else:
        p = scores[label]["truepositiv"]/(scores[label]["truepositiv"]+scores[label]["falsepositiv"])
        r = scores[label]["truepositiv"]/(scores[label]["truepositiv"]+scores[label]["falsenegativ"])
    precision += p
    recall += r
    if (p+r) == 0:
        f1 += 0
    else:
        f1 += 2*((p*r)/(p+r))
precision = precision/len(scores)
recall = recall / len(scores)
f1 = f1/len(scores)
print("acc",accuracy,"prec",precision,"recall",recall,"f1",f1)
    



