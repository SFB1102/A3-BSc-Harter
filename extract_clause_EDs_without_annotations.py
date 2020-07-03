#java -mx1300m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
from pycorenlp import StanfordCoreNLP
core_nlp = StanfordCoreNLP('http://localhost:9000')
import xml.etree.ElementTree as ET
import os
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from string import punctuation
import codecs

from set_path_to_InScript import path_to_InScript as path
scenario = input("Please choose one scneario from: cake, library, flight, haircut, grocery, train, tree, bicycle, bus, bath")

class Dependent:
    def __init__(self,infos,satznummer,verb):
        self.text = infos["dependentGloss"]
        self.nummer = infos["dependent"]
        self.satzwort_id = str(satznummer)+"-"+str(self.nummer)
        self.dependenten = set()
        self.governor = infos["governor"]
        self.dependenzlabel = infos["dep"]
        self.nichtaufstack = False
        self.verb = verb
    def add_dependenten(self,dependent):
        self.dependenten.add(dependent)    
    def get_dependenten(self):
        return [(d.nummer,d.text) for d in self.dependenten]
    def set_nichtaufstack(self,value):
        self.nichtaufstack = value


pfade = [path+scenario+"/"+datei for datei in os.listdir("G:/Bachelorarbeit/InScript_LREC2016/InScript/corpus/"+scenario)]

zeilen = []
z = 1
for pfad in  pfade:
        z = 1
        esd = dict()
        print(pfad)
        tree = ET.parse(pfad)
        """eventlabels = dict()
        for event in tree.find("annotations").find("events").findall("label"):
            eventlabel = event.get("name")
            if not eventlabel in ["Evoking","RelNScrEv","Unclear","UnrelEv"]:
                name=eventlabel[6:]
            else:
                name=eventlabel
            nummer = event.get("from")
            eventlabels[nummer] = name"""
        text = tree.find("text").find("content").text
        sätze = sent_tokenize(text)
        satznummer = 0
        for s in sätze:
            satznummer += 1
            print(satznummer)
            esd[satznummer]= dict()
            out = core_nlp.annotate(s, properties={'annotators': 'tokenize,ssplit,pos,lemma,parse,depparse','outputFormat': 'json','ssplit.isOneSentence': True})
            dependenten = dict()
            verben = []
            for token in out["sentences"][0]["tokens"]:
                if token["pos"].startswith("V"):
                    verben.append(token["index"])
                    print(token)
            for depinfo in out["sentences"][0]["basicDependencies"]:
                print(depinfo)
                dependenten[depinfo["dependent"]]=Dependent(depinfo,satznummer,depinfo["dependent"] in verben)
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
                if not event.verb:
                    for x in event.dependenten:
                        if (x.dependenzlabel == "cop" or x.dependenzlabel == "xcomp") and x.verb:
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
                    elif dep.verb:
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
                esd[int(satznummer)][int(ednummer)]= edtext
        zeile = ""
        for satz in sorted(esd):
            for x in sorted(esd[satz]):
                ed = esd[satz][x]
                zeile+=ed+"*"
        zeilen.append(zeile[:-1])





with codecs.open("InScript_clauseEDs_without_Annotations.txt","w","utf-8") as file:
    for z in zeilen:
        file.write(z+"\n")





        
    
