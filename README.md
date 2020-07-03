# Scriptparsing-with-allenNLP
## additionally required
### InScript Corpus
The data from which the Eds are extracted comes from the InScript corpus. The Corpus can be found [here](https://my.hidrive.com/lnk/AgAzHi6B#file).
In set_path_to_InScript you have to specify the path to the Corpus data, so that the programms which are 
responsible for extracting EDs out of the InScript texts can find them.
### allenNLP
### Python packages
### Stanford CoreNLP Server
## ED Extraction
The files extract_only_relevant_EDs.py and extract_unrelevant_EDs_too.py are responsible for the ED Exraction 
out of the InScript texts. Both programms will ask you which scenario you want to look at and will print the 
extracted EDs to the file InScript.txt, which can be used as input for allennlp_kfold_Eventtagger.py. There is 
a complete ESD in each line consisting of all EDs extracted from one InScript text. The difference between
extract_only_relevant_EDs.py and extract_unrelevant_EDs_too.py is, as the name indicates, that the first one
only extract the scenario relevant EDs and the second one extract all EDs.
## Event-Tagger
The Event-Tagger allennlp_kfold_Eventtagger.py reads in the file InScriptESDs.txt which can be produced by 
either extract_only_relevant_EDs.py or extract_unrelevent_EDs_too.py, depending on whether you only want to 
label the relevant events or all events. Then allennlp_kfold_Eventtagger.py will do 10-fold-cross-validation
over the read Data. That means for every fold it will train on 90% of the ESDs for one scenario and evaluate 
on 10%. At the end the makro average over all folds will be printed for accuracy, precision recall and f1-score.
For every makro average there will be printed two values. In case you only take the relevant EDs into account,
this values will be equal. But if you consider the unvelevant ones, too, the values will differ from each other.
This is because there are different types of labels for unrelevant EDs. For the first value an unrelevant ED counts
as labeld correctly, if it got assigned one of them. For the second value it has to be the correct unrelevant label.
## Clause-ED Extraction
## Event-Tagger for Clause-EDs

