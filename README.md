# Scriptparsing-with-allenNLP
## additionally required
### InScript Corpus
The data from which the EDs are extracted comes from the InScript corpus. The corpus can be found [here](https://my.hidrive.com/lnk/AgAzHi6B#file).
In set_path_to_InScript you have to specify the path to the corpus data, so that the programms which are 
responsible for extracting EDs out of the InScript texts can find them.
### allenNLP
Since allenNLP doesn't work on Windows machines, I used the allenlp environment on the Coli-Server. 
It works like this:
* First you have to enter . /proj/contrib/anaconda3/etc/profile.d/conda.sh 
* Then you can start the anaconda environment by entering conda activate allennlp

If you want to install allennlp on your own computer, you can find the instructions [here](https://github.com/allenai/allennlp).
### Python packages
This are the python packages which are needed:
* xml.etree.ElementTree (to read InScript data for ED-extraction)
* pycorenlp (only needed for Clause-EDs)
* nltk (only needed for Clause-EDs)

They can be intalled via pip install.
### Stanford CoreNLP Server
For using the pycorenlp package, you have to download Stanford CoreNLP,
which can be found [here](https://stanfordnlp.github.io/CoreNLP/download.html). 
After downloading it, use the command line to navigate to the downloaded Stanford CoreNLP folder.
Ther you can start the server which is used by the python package using this command:

java -mx1300m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

Now the Stanford CoreNLP Server is running and you can use exctract_clause_EDs.py and
extract_clause_EDs_without_annotations.py

## ED Extraction
The files extract_only_relevant_EDs.py and extract_unrelevant_EDs_too.py are responsible for the ED-extraction 
out of the InScript texts. Both programms will ask you which scenario you want to look at and will print the 
extracted EDs to the file InScript.txt, which can be used as input for allennlp_kfold_Eventtagger.py. There is 
a complete ESD in each line consisting of all EDs extracted from one InScript text. The difference between
extract_only_relevant_EDs.py and extract_unrelevant_EDs_too.py is, as the name indicates, that the first one
only extract the scenario relevant EDs and the second one extract all EDs.
## Event-Tagger
The Event-Tagger allennlp_kfold_Eventtagger.py reads in the file InScriptESDs.txt which can be produced by 
either extract_only_relevant_EDs.py or extract_unrelevent_EDs_too.py, depending on whether you only want to 
label the relevant events or all events. Then allennlp_kfold_Eventtagger.py will do 10-fold-cross-validation
over the read data. That means for every fold it will train on 90% of the ESDs for one scenario and evaluate 
on 10%. At the end the makro average over all folds will be printed for accuracy, precision recall and f1-score.
For every makro average there will be printed two values. In case you only take the relevant EDs into account,
this values will be equal. But if you consider the unvelevant ones, too, the values will differ from each other.
This is because there are different types of labels for unrelevant EDs. For the first value an unrelevant ED counts
as labeld correctly, if it got assigned one of them. For the second value it has to be the correct unrelevant label.
## Clause-ED Extraction
With the files extract-clauseEDs.py and extract-clauseEds_without_annotations.py you can produce EDs which consist
of whole clauses instead of the eventverb plus little context. So the complete sentence can be given as input, which
provides more context for the Event-Tagger. extract-clausesEDs.py uses annotations already provided in the InScript
corpus, like extract_only_relevant_EDs.py and extract_unrelevant_EDs_too.py do. Extract-clausesEDs_without_annotations.py
only uses information about the structure of the sentences, the Stanford Parser gives and doesn't need the InScript
annotations about participants or coreferents.
## Event-Tagger for Clause-EDs
Since I used slightly different paramethers for tagging with clause-EDs, you need to use allennlp_kfold_Eventtagger_clauseEDs.py
instead of allennlp_kfold_clauseEDs.py. For the tagging without InScript annotations you have to combine the clause-EDs without
annotations and allennlp_kfold_Eventtagger_clauseEDs_without_Annotations.py, because without the annotations you can't
annotate automatically anymore.
 
