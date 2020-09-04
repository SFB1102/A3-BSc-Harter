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

class EDDatasetReader(DatasetReader):
    def __init__(self, token_indexers)->None:
        super().__init__(lazy=False)
        #self.token_indexers = token_indexers or {"tokens":SingleIdTokenIndexer()}
        self.token_indexers = token_indexers
    def text_to_instance(self,indices,tokens: List[Token], tags: List[str] = None) -> Instance:
        labels = []
        vorher = 0
        j = 0
        for index in indices:
            for i in range(0,index-vorher):
                if tags:
                    labels.append(tags[j])
            j += 1
            vorher = index

            
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        if tags:
            label_field = SequenceLabelField(labels=labels,sequence_field=sentence_field)
            fields["labels"] = label_field
          
        fields["indices"]=MetadataField(indices)
        return Instance(fields)
    def _read(self,file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                #pairs = line.strip().split()
                esd = line.strip().split("*")
                #print(pairs)
                try:
                    #sentence, tags = zip(*(pair.split("###") for pair in eds))
                    woerter = []
                    tags = []
                    indices = []
                    index = 0
                    for element in esd:
                        ed,tag = element.split("###")
                        woerter += ed.split()
                        tags.append(tag)
                        index += len(ed.split())
                        indices.append(index)
                    #yield self.text_to_instance([Token(ed) for ed in eds], tags)
                    yield self.text_to_instance(indices,[Token(wort) for wort in woerter],tags)
                except ValueError:
                    print(woerter)
                


class EventTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 #encoder: Seq2SeqEncoder,
                 encoder1,
                 encoder2,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        print(encoder2.get_output_dim())
        self.hidden2tag = torch.nn.Linear(in_features=encoder2.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        #self.hidden2tag = torch.nn.Sigmoid()
     
        
        self.accuracy = CategoricalAccuracy()
        self.fBeta = FBetaMeasure()
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                indices:List,
                labels: torch.Tensor = None,
                ) -> torch.Tensor:
                
        #mask = get_text_field_mask(sentence,num_wrapping_dims=1)
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        #print("embeddings",embeddings.shape,mask.shape)
        encoder1_out = self.encoder1(embeddings, mask)
        #print("encoder1out",encoder1_out.shape,indices[0])
        tensors = torch.split(encoder1_out, 1, 1)
        m = torch.split(mask, 1, 1)
        sh = tensors[0].shape
        liste = []
        liste_m = []
        vorher = 0
        for index in indices[0]:
            #print(tensors[index-1].shape,tensors[index-1])
            t = tensors[vorher]
            vorher +=1
            for i in range(vorher,index):
                t = t.add(tensors[i])
            liste.append(t)
            liste_m.append(m[index - 1])
            vorher = index
        encoder2_in = torch.cat(liste,1)
        mask2 = torch.cat(liste_m,1)
        #print("encoder2in",encoder2_in.shape,encoder1_out.shape)
        encoder2_out=self.encoder2(encoder2_in, mask2)
        tag_logits = self.hidden2tag(encoder2_out)
        output = {"tag_logits": tag_logits}
        
        if labels is not None:
            #print("labels",labels.shape)
            l = torch.split(labels,1,1)
            liste_l = []
            for index in indices[0]:
                liste_l.append(l[index-1])
            labels = torch.cat(liste_l,1)
            #print("labels2",labels.shape)
            self.accuracy(tag_logits, labels, mask2)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask2)
            self.fBeta(tag_logits,labels,mask2)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
        
reader = EDDatasetReader({"bert" : PretrainedBertIndexer(pretrained_model="bert-base-uncased")})
validation_dataset = reader.read("InScriptESDs.txt")
train_dataset = reader.read("DeScript_train.txt")
"""for instanz in train_dataset:
    print(instanz)
    print(instanz["indices"].metadata)"""
vocab = Vocabulary.from_instances(validation_dataset+train_dataset)
#glove_daten = reader.read("glove_voc.txt")


#token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            #embedding_dim=EMBEDDING_DIM,trainable = True, weight = wmatrix)
bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-uncased")
word_embeddings = BasicTextFieldEmbedder({"bert":bert_embedder},allow_unmatched_keys = True,embedder_to_indexer_map={"bert":["bert","bert-offsets"]})
#print(bert_embedder)
#word_embeddings = TextFieldEmbedder(bert_embedder)

HIDDEN_DIM1 = 50
HIDDEN_DIM2 = 45 #45
#lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(hidden_size = HIDDEN_DIM,input_size=word_embeddings.get_output_dim(),batch_first=True,bias=False,bidirectional=True))
gru = PytorchSeq2SeqWrapper(torch.nn.GRU(hidden_size = HIDDEN_DIM1,input_size=word_embeddings.get_output_dim(),batch_first=True,bidirectional=True))
gru2 = PytorchSeq2SeqWrapper(torch.nn.GRU(hidden_size = HIDDEN_DIM2,input_size=gru.get_output_dim(),batch_first=True,bidirectional=True))
#print("outputdim",gru.get_output_dim())
#rnn = PytorchSeq2SeqWrapper(torch.nn.RNN(hidden_size = HIDDEN_DIM,input_size=word_embeddings.get_output_dim(),bias=False,batch_first=True,bidirectional=True))
model = EventTagger(word_embeddings,gru,gru2,vocab)
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
    #print("p",p)
    precision+=p
precision/=len(precisions)
for r in recalls:
    #print("r",r)
    recall+=r
recall/=len(recalls)
for f in fscores:
    #print("f",f)
    fscore+=f
fscore/=len(fscores)
print("precision:",precision,"recall:",recall,"fscore:",fscore)

weiter = input("Zum weiter machen klicken")

predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
richtig = 0
alle = 0
zeile = 1

richtig = 0
alle = 0
scores = dict()
with open("InScriptESDs.txt","r") as val:
    for line in val:
        #print(zeile)
        zeile+=1
        labels = []
        indices = []
        satz = ""
        vorher = 0
        eds = []
        for token in line.strip().split("*"):
            (ed,label) = token.split("###")
            satz += ed+" "
            eds.append(ed)
            labels.append(label)
            indices.append(len(ed.split())+vorher)
            vorher = indices[-1]
        satz = satz.strip()
        satztoken = [Token(wort) for wort in satz.split()]
        satzinstanz = reader.text_to_instance(indices,satztoken)
        tag_logits = model.forward_on_instance(satzinstanz)
        tag_ids = np.argmax(tag_logits["tag_logits"],-1)
        liste = [model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]
        print(len(liste),len(tag_ids),len(eds),len(labels))
        ergebnisse = zip(eds,liste,labels)
        for ed,zugewiesen,gold in ergebnisse:
            with open("out_netz.txt","a") as out:
                out.write(gold+"\t"+zugewiesen+"\n")
            if not zugewiesen in scores:
                scores[zugewiesen] = dict()
                scores[zugewiesen]["truepositiv"]=0
                scores[zugewiesen]["falsepositiv"]=0
                scores[zugewiesen]["falsenegativ"]=0
            if not gold in scores:
                scores[gold] = dict()
                scores[gold]["truepositiv"]=0
                scores[gold]["falsepositiv"]=0
                scores[gold]["falsenegativ"]=0
            alle += 1
            if zugewiesen == gold:
                richtig += 1
                scores[gold]["truepositiv"] += 1
            else:
                scores[gold]["falsenegativ"] += 1
                scores[zugewiesen]["falsepositiv"] += 1
            print(ed,zugewiesen,gold)
        with open("out_netz.txt","a") as out:
            out.write("\n")
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
print(accuracy,precision,recall,f1)
    
