from collections import defaultdict

from allennlp import *

from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel

from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np
from allennlp.training.optimizers import DenseSparseAdam

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ListField, ArrayField,  SequenceField, MetadataField

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

from collections import defaultdict

import time



fold = 10
dataset = "InScriptESDs.txt" 


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
    def _read(self,lines: list) -> Iterator[Instance]:
        #print("pairs",pairs)
        for line in lines:
            try:
                sentence,tags = [],[]
                for pair in line.split():
                    w,t=pair.split("###")
                    sentence.append(w)
                    tags.append(t)
                yield self.text_to_instance([Token(word) for word in sentence], tags)
            except ValueError:
                print(line)
                


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
 
    


instanzen = dict()
verbinfo = dict()
zeile = 0
with open(dataset,"r") as f:
    for line in f:
        instanzen[zeile]=line.strip()
        zeile+=1
print("zeile1",zeile)
zeile = 0
with open("evaluierungshilfe.txt","r") as f:
    for line in f:
        verbinfo[zeile] = line.strip()
        zeile += 1
print("zeile2",zeile)

        
        
reader = EDDatasetReader({"bert" : PretrainedBertIndexer(pretrained_model="bert-base-uncased")})
bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-uncased")
word_embeddings = BasicTextFieldEmbedder({"bert":bert_embedder},allow_unmatched_keys = True,embedder_to_indexer_map={"bert":["bert","bert-offsets"]})


HIDDEN_DIM = 50


foldlen = round(len(instanzen)/fold)
acc_gesamt,prec_gesamt,recall_gesamt,f_gesamt = 0,0,0,0
start = time.time()
for i in range(0,fold):
    if i == fold-1:
        keys = sorted(instanzen)[i*foldlen:]
        verbkeys  = sorted(verbinfo)[i*foldlen:]
    else:
        keys = list(range(i*foldlen,i*foldlen+foldlen))
        verbkeys = list(range(i*foldlen,i*foldlen+foldlen))
    test = []
    train = []
    verbtest = []
    verbtrain = []
    for z in instanzen:
        if z in keys:
            test.append(instanzen[z])
            verbtest.append(verbinfo[z])
        else:
            train.append(instanzen[z])
            verbtrain.append([verbinfo[z]])
    validation_dataset = reader.read(test)
    train_dataset = reader.read(train)
    vocab = Vocabulary.from_instances(validation_dataset+train_dataset)
    gru = PytorchSeq2SeqWrapper(torch.nn.GRU(hidden_size = HIDDEN_DIM,input_size=word_embeddings.get_output_dim(),batch_first=True,bidirectional=True))
    model = EventTagger(word_embeddings,gru,vocab)
    iterator = BasicIterator(batch_size=1)
    iterator.index_with(vocab)
    optimizer = DenseSparseAdam(model.parameters(),lr=0.001)
    trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  #patience=5,
                  num_epochs=10)
    trainer.train()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    richtig = 0
    alle = 0
    scores = dict()
    zeile = 1
    for line in test:
        print(zeile)
        zeile += 1
        labels = []
        indices = []
        satz = ""
        for token in line.split():
            (word,label) = token.split("###")
            satz += word+" "
            labels.append(label)
        satz = satz.strip()
        tag_logits = predictor.predict(satz)['tag_logits']
        tag_ids = np.argmax(tag_logits, axis=-1)
        liste = [model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]
        print(len(liste),len(labels),len(verbtest[zeile-2]).split())
        for tag,gold,verb,wort in zip(liste, labels,verbtest[zeile-2].split(),satz.split()):
            print(tag,gold,verb,wort)
            if verb == "V":
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
    

    precision = precision/len(scores)
    recall = recall / len(scores)
    f1 = 2*(precision*recall)/(recall+precision)
    acc_gesamt += accuracy
    prec_gesamt += precision
    recall_gesamt += recall
    f_gesamt += f1
    print(accuracy,precision,recall,f1)

stop = time.time()
    
print("accuracy:",str(acc_gesamt/fold))
print("precision:",str(prec_gesamt/fold))
print("recall:",str(recall_gesamt/fold))
print("f1 score:",str(f_gesamt/fold))

sekunden = stop-start
minuten = int(sekunden/60)
sekunden=60*((sekunden/60)-minuten)
stunden = int(minuten/60)
minuten = 60*((minuten/60)-stunden)
print(str(stunden),"stunden",str(minuten),"minuten und",str(sekunden),"sekunden")



