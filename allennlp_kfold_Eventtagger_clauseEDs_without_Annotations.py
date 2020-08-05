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

from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure

from allennlp.data.iterators import BasicIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

from collections import defaultdict

EPOCHS = 10


                
class Satz_Eventtagger(Model):
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
            
class SatzDatasetReader(DatasetReader):
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
    def _read(self,zeilen:list) -> Iterator[Instance]:
        for line in zeilen:
            #print(line)
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

val,train = [],[]
with open("Vergleich_Satz_val.txt","r") as f:
    for line in f:
        val.append(line.strip())
with open("Vergleich_Satz_train.txt","r") as f:
    for line in f:
        train.append(line.strip())
    
Satz_reader =  SatzDatasetReader({"bert" : PretrainedBertIndexer(pretrained_model="bert-base-uncased",truncate_long_sequences=False)})
Satz_validation_dataset = Satz_reader.read(val)
Satz_train_dataset = Satz_reader.read(train)
    

Satz_vocab = Vocabulary.from_instances(Satz_validation_dataset+Satz_train_dataset)
bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-uncased")
Satz_word_embeddings = BasicTextFieldEmbedder({"bert":bert_embedder},allow_unmatched_keys = True,embedder_to_indexer_map={"bert":["bert","bert-offsets"]})
Satz_HIDDEN_DIM1 = 65
Satz_HIDDEN_DIM2 = 45
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(hidden_size = Satz_HIDDEN_DIM1,input_size=Satz_word_embeddings.get_output_dim(),batch_first=True,bidirectional=True))
lstm2 = PytorchSeq2SeqWrapper(torch.nn.LSTM(hidden_size = Satz_HIDDEN_DIM2,input_size=lstm.get_output_dim(),batch_first=True,bidirectional=True))
Satz_model = Satz_Eventtagger(Satz_word_embeddings,lstm,lstm2,Satz_vocab)
Satz_optimizer = DenseSparseAdam(Satz_model.parameters(),lr=0.001)
Satz_iterator = BasicIterator(batch_size=1)
Satz_iterator.index_with(Satz_vocab)
Satz_trainer = Trainer(model=Satz_model,
                  optimizer=Satz_optimizer,
                  iterator=Satz_iterator,
                  train_dataset=Satz_train_dataset,
                  validation_dataset=Satz_validation_dataset,
                  #patience=5,
                  num_epochs=EPOCHS)
Satz_trainer.train()

with open("InScript_ganzeSaetze_ohneAnnotationenTest.txt","r") as f:
    with open("output_ohneAnnotation.txt","w") as out:
        text = 0
        ix = 0
        for line in f:
            text += 1
            out.write(str(text)+"\n")
            eds = []
            vorher = 0
            indices = []
            satz = ""
            for ed in line.strip().split("*"):
                satz += ed+" "
                eds.append(ed)
                indices.append((len(ed.split())+vorher))
                vorher = indices[-1]
            ix += 1
            satz = satz.strip()
            satztoken = [Token(wort) for wort in satz.split()]
            instanz_mitReihenfolge = Satz_reader.text_to_instance(indices,satztoken)
            tag_logits = Satz_model.forward_on_instance(instanz_mitReihenfolge)        
            tag_ids = np.argmax(tag_logits["tag_logits"],-1)  
            liste = [Satz_model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]
            for zugewiesen,ed in zip(liste,eds):
                out.write(zugewiesen+"\t"+ed+"\n")
                print(zugewiesen+"\t"+ed+"\n")
