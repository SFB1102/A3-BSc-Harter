#java -mx1300m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
from pycorenlp import StanfordCoreNLP
core_nlp = StanfordCoreNLP('http://localhost:9000')
import xml.etree.ElementTree as ET
import os
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from string import punctuation
import codecs

scenario = input("Please choose one scneario from: cake, library, flight, haircut, grocery, train, tree, bicycle, bus, bath")
from set_path_to_InScript import path_to_InScript as path

class Dependent:
    def __init__(self,infos,satznummer):
        self.text = infos["dependentGloss"]
        self.nummer = infos["dependent"]
        self.satzwort_id = str(satznummer)+"-"+str(self.nummer)
        self.dependenten = set()
        self.governor = infos["governor"]
        self.dependenzlabel = infos["dep"]
        self.nichtaufstack = False
    def add_dependenten(self,dependent):
        self.dependenten.add(dependent)    
    def get_dependenten(self):
        return [(d.nummer,d.text) for d in self.dependenten]
    def set_nichtaufstack(self,value):
        self.nichtaufstack = value

pfade = [path+scenario+"/"+datei for datei in os.listdir("G:/Bachelorarbeit/InScript_LREC2016/InScript/corpus/"+scenario)]
zeilen = []
for pfad in  pfade:
    esd = dict()
    print(pfad)
    tree = ET.parse(pfad)
    eventlabels = dict()
    for event in tree.find("annotations").find("events").findall("label"):
        eventlabel = event.get("name")
        if not eventlabel in ["Evoking","RelNScrEv","Unclear","UnrelEv"]:
            name=eventlabel[6:]
        else:
            name=eventlabel
        nummer = event.get("from")
        eventlabels[nummer] = name
    text = tree.find("text").find("content").text
    sätze = sent_tokenize(text)
    satznummer = 0
    for s in sätze:
        satznummer += 1
        print(satznummer)
        esd[satznummer]= dict()
        out = core_nlp.annotate(s, properties={'annotators': 'tokenize,ssplit,pos,lemma,parse,depparse','outputFormat': 'json','ssplit.isOneSentence': True})
        dependenten = dict()
        for depinfo in out["sentences"][0]["basicDependencies"]:
            print(depinfo)
            dependenten[depinfo["dependent"]]=Dependent(depinfo,satznummer)
        for d_id in dependenten:
            d = dependenten[d_id]
            if int(d.governor) > 0:
                dependenten[d.governor].add_dependenten(d)
            else:
                start = d
        eventstack = [start]
        while eventstack:
            event = eventstack.pop(0)
            ednummer = int(event.nummer)
            ed = [(ednummer,event.text)]
            depstack = list(event.dependenten)
            try:
                label = eventlabels[event.satzwort_id]
            except KeyError:
                for x in event.dependenten:
                    if (x.dependenzlabel == "cop" or x.dependenzlabel == "xcomp") and x.satzwort_id in eventlabels:
                        label = eventlabels[x.satzwort_id]
                        ed.append((x.nummer,x.text))
                        x.set_nichtaufstack(True)
                        if x.dependenzlabel == "xcomp":
                            for xdep in x.dependenten:
                                depstack.append(xdep)
                        break
           
            
            while depstack:
                dep = depstack.pop(0)
                wort = dep.text
                if wort in punctuation:
                    pass
                elif dep.satzwort_id in eventlabels:
                    if not dep.nichtaufstack:
                        eventstack.append(dep)
                else:
                    depstack+=list(dep.dependenten)
                    ed.append((dep.nummer,wort))
            edtext = ""
            for n,wort in sorted(ed):
                wort = wort.lower().strip()
                edtext+=wort+" "
            edtext = edtext.strip()
            esd[int(satznummer)][int(ednummer)]=(edtext,label)
    zeile = ""
    for satz in sorted(esd):
        for x in sorted(esd[satz]):
            ed,label = esd[satz][x]
            zeile+=ed+"###"+label+"*"
    zeilen.append(zeile[:-1])

with codecs.open("InScript_clauseEDs.txt","w","utf-8") as file:
    for z in zeilen:
        file.write(z+"\n")
                    
                
                
        
             
             
            
        







