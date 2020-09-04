import xml.etree.ElementTree as ET
import random
from esd import labels 
from set_path_to_DeScript import path_to_DeScript
import string

scenario = input("Please choose one scneario from: baking a cake, borrowing a book from the library, flying in an airplane, getting a hair cut, going grocery shopping, going on a train, planting a tree, repairing a flat bicycle tire , riding on a bus, taking a bath")


scripts = []
#Wortebene
"""def auslesen(baum,source):
    for script in baum.findall("script"):
        zeile = ""
        nummer = script.get("id")
        for event in script.findall("item"):
            try:
                label = labels[scenario,source,nummer,event.get("slot")]    
                for wort in event.get("original").split():
                    if wort == "I":
                        wort_ = "I"
                    else:
                        wort_ = ""
                        for char in wort.lower():
                            if not char in string.punctuation:
                                wort_ += char
                        zeile += wort_+"###"+label+" "
            except KeyError:
                pass
        zeile = zeile.strip()
        scripts.append(zeile)"""
#ED Ebene
def auslesen(baum,source):
    for script in baum.findall("script"):
        zeile = ""
        nummer = script.get("id")
        for event in script.findall("item"):
            try:
                label = labels[scenario,source,nummer,event.get("slot")]
                for wort in event.get("original").split():
                    if wort == "I":
                        wort_ = "I"
                    else:
                        wort_ = ""
                        for char in wort.lower():
                            if not char in string.punctuation:
                                wort_ += char
                        zeile += wort_+" "
                zeile=zeile.strip()
                zeile+="###"+label+"*"
            except KeyError:
                pass
        zeile = zeile[:-1]
        scripts.append(zeile)
        


            
try:
    tree1 = ET.parse(path_to_DeScript+"/esds/pilot_esd/"+scenario+".pilot.xml")
    auslesen(tree1,"1")
except FileNotFoundError:
    pass
tree2 = ET.parse(path_to_DeScript+"/esds/second_esd/"+scenario+".new.xml")
auslesen(tree2,"2")
    


random.shuffle(scripts)
with open("DeScript_train.txt","w") as file:
        for script in scripts:
            file.write(script+"\n")
            
