import xml.etree.ElementTree as ET
import os
import string

from set_path_to_InScript import path_to_InScript

scenario = input("Please choose one scneario from: cake, library, flight, haircut, grocery, train, tree, bicycle, bus, bath")

with open("evaluierungshilfe.txt","w"):
    pass


with open("InScriptESDs.txt","w") as file:
    pfade = [path_to_InScript+scenario+"/"+datei for datei in os.listdir(path_to_InScript+scenario)]
    for pfad in pfade:
        print(pfad)
        zeile = ""
        testzeile = ""
        try:
            tree = ET.parse(pfad)
        except FileNotFoundError:
            continue
        events = dict()
        partizipanten = dict()
        corefs = dict()
        ids = dict()
        for event in tree.find("annotations").find("events").findall("label"):
            eventlabel = event.get("name")
            if not eventlabel in ["Evoking","RelNScrEv","Unclear","UnrelEv"]:
                satznummer = event.get("from").split("-")[0]
                index = (event.get("from"),event.get("to"))
                if not index[1]:
                    wortnummern = [index[0].split("-")[1]]
                else:
                    wortnummern = [str(n) for n in range(int(index[0].split("-")[1]),int(index[1].split("-")[1])+1)]
                if not satznummer in events:
                    events[satznummer]=dict()
                events[satznummer][index]=dict()
                events[satznummer][index]["label"]=eventlabel[6:]
                events[satznummer][index]["text"]=event.get("text")
                events[satznummer][index]["wortindices"]=wortnummern
        for partizipant in tree.find("annotations").find("participants").findall("label"):
            satznummer = partizipant.get("from").split("-")[0]
            index = (partizipant.get("from"),partizipant.get("to"))
            if not index[1]:
                wortnummern = [index[0].split("-")[1]]
            else:
                wortnummern = [str(n) for n in range(int(index[0].split("-")[1]),int(index[1].split("-")[1])+1)]
            original_text = partizipant.get("text")
            id_ = partizipant.get("id")
            if not satznummer in partizipanten:
                partizipanten[satznummer] = dict()
            partizipanten[satznummer][index]=dict()
            partizipanten[satznummer][index]["label"]=partizipant.get("name")
            partizipanten[satznummer][index]["text"]= original_text
            partizipanten[satznummer][index]["wortindices"]=wortnummern
            ids[id_]= original_text
            partizipanten[satznummer][index]["id"]= id_
        del_liste = []
        for s in partizipanten:
            lang = []
            for p in partizipanten[s]:
                if len(partizipanten[s][p]["wortindices"])>1:
                    lang.append(partizipanten[s][p])
            for p in partizipanten[s]:
                if len(partizipanten[s][p]["wortindices"])==1:
                    index = partizipanten[s][p]["wortindices"][0]
                    for l in lang:
                        if index in l["wortindices"]:
                            del_liste.append((s,p))
                            break
        for d in del_liste:
            del partizipanten[d[0]][d[1]]
        del_liste = []
        for s in events:
            lang = []
            for e in events[s]:
                if len(events[s][e]["wortindices"])>1:
                    lang.append(events[s][e])
            for e in events[s]:
                if len(events[s][e]["wortindices"])==1:
                    index = events[s][e]["wortindices"][0]
                    for l in lang:
                        if index in l["wortindices"]:
                            del_liste.append((s,e))
                            break
        for d in del_liste:
            del events[d[0]][d[1]]
        pronouns = "yourselves Yourselves Themselves themselves myself Myself yourself Yourself himself Himself Herself herself itself Itself ourselves Ourselves I you he she it we they me him her us them my your his her its our their mine yours hers ours theirs this that these those You He She It We They Me Him Her Us Them My Your His Her Its Our Their Mine Yours Hers Ours Theirs This That These Those".split()
        for kette in tree.find("annotations").find("chains").findall("chain"):
            c_ids = kette.get("elements").split()
            x = ""
            for i in c_ids:
                if not ids[i] in pronouns:
                    x = ids[i]
                    break
            if x:
                for i in c_ids:
                    corefs[i]=x
        for satz in tree.find("text").find("sentences").findall("sentence"):
            t = satz.get("id")
            dep = dict()
            lemmata = dict()
            wörter = dict()
            for token in satz.findall("token"):
                token_id = token.get("id")
                wörter[token_id]=token.get("content")
                try:
                    lemmata[token_id]=token.find("lemma").get("type")
                except AttributeError:
                    pass
                try:
                    d = token.find("dep")
                    head = d.get("head")
                    relation = d.get("type")
                    if not head in dep:
                        dep[head]=dict()
                    dep[head][token_id]=relation
                except AttributeError:
                    pass
            if t in events:
                for event in events[t]:
                    evaluierungsid = event[0].split("-")[1]
                    eventlabel = events[t][event]["label"]
                    eventteile = dict()
                    try:
                        eventteile[int(event[0].split("-")[1])]=(lemmata[event[0]],"V")
                    except KeyError:
                        eventteile[int(event[0].split("-")[1])]=(wörter[event[0]],"V")
                    if event[1]:
                        eventteile[int(event[1].split("-")[1])]= (wörter[event[1]],len(wörter[event[1]].split()))
                        
                    for e in event:
                        if e in dep:
                            for dependent in dep[e]:
                                partizipant=None
                                for p in partizipanten[t]:
                                     if dependent in p:
                                         partizipant=p
                                if partizipant:
                                    if partizipanten[t][partizipant]["text"] in pronouns:
                                        if partizipanten[t][partizipant]["id"] in corefs:
                                            eventteile[int(dependent.split("-")[1])] = (corefs[partizipanten[t][partizipant]["id"]],len(corefs[partizipanten[t][partizipant]["id"]].split()))
                                    else:
                                
                                        eventteile[int(dependent.split("-")[1])] = (partizipanten[t][partizipant]["text"],len(partizipanten[t][partizipant]["text"].split()))
                            
                                else:
                                    if dependent in dep:
                                        liste = dep[dependent]
                                        liste2 = []
                                        for l in liste:
                                            partizipant = None
                                            for p in partizipanten[t]:
                                                if dependent in p:
                                                    partizipant=p
                                            if partizipant:
                                                liste2.append(partizipant)
                                        if liste2:
                                            eventteile[int(dependent.split("-")[1])] = (wörter[dependent],len(wörter[dependent].split()))
                                            for l in liste2:
                                                if partizipanten[t][l]["text"] in pronouns:
                                                    if partizipanten[t][l]["id"] in corefs:
                                                        eventteile[int(l.split("-")[1])] = (corefs[partizipanten[t][l]["id"]],len(corefs[partizipanten[t][l]["id"]].split()))
                                                        
                                                else:
                                                    eventteile[int(l.split("-")[1])] = (partizipanten[t][l]["text"],len(partizipanten[t][l]["text"].split()))
                                                    
                                                    
                                                    
                                                 
                                                    
                        
                    eventtext = ""
                    eventtest = ""
                    sort = sorted(eventteile)
                    for s in sort:
                        test = eventtext.split()
                        test2 = eventteile[s][0].split()
                        if not test2 == test[-len(test2):]:
                            eventtext += eventteile[s][0]+" "
                            if eventteile[s][1]=="V":
                                eventtest+="V"
                            else:
                                for strich in range(0,eventteile[s][1]):
                                    eventtest+="-"

                    #print(eventtext,eventtest)
                    indexz = 0
                    for w in eventtext.split():
                        wortneu = ""
                        for char in w.lower():
                            if not char in string.punctuation:
                                wortneu += char
                        if wortneu:
                            if not wortneu in ["i","we","they","then"]:
                                zeile += w+"###"+eventlabel+" "
                                testzeile += eventtest[indexz]+" "
                        indexz += 1
        print("------------",len(zeile.split()),len(testzeile.split()),"------------")
        if zeile:
            file.write(zeile.strip()+"\n")
        
        with open("evaluierungshilfe.txt","a") as g:
            if testzeile:
                g.write(testzeile.strip()+"\n")
                    
