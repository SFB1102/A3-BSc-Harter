
import sys
import xml.etree.ElementTree as ElementTree

from set_path_to_DeScript import path_to_DeScript

from pathlib import Path
from collections import defaultdict

GOLD_PATH1 = Path(path_to_DeScript+'/gold_paraphrase_sets/first_gold_annotation/')
GOLD_PATH2 = Path(path_to_DeScript+'/gold_paraphrase_sets/second_gold_annotation/')

ESD1_PATH = Path(path_to_DeScript+'/esds/pilot_esd/')
ESD2_PATH = Path(path_to_DeScript+'/esds/second_esd/')

ANNOTATED_FILES = [
    "baking a cake",
    "borrowing a book from the library",
    "flying in an airplane",
    "fyling\ in\ an\ airplane",
    "getting a hair cut",
    "going grocery shopping",
    "going on a train",
    "planting a tree",
    "repairing a flat bicycle tire",
    "riding on a bus",
    "taking a bath"
]


labels = dict()

for filename in ANNOTATED_FILES:
    for gold in [GOLD_PATH1, GOLD_PATH2]:
        path = gold.joinpath(Path(filename).with_suffix('.new.xml'))
        if path.exists():
            tree = ElementTree.parse(str(path))
            for cluster in tree.getroot():
                for item in cluster:
                    labels[filename, item.attrib['source'], item.attrib['script'], item.attrib['slot']] = cluster.attrib['event']

"""for filename in ANNOTATED_FILES:
    for path, suffix in [(ESD1_PATH, ".pilot.xml"), (ESD2_PATH, ".new.xml")]:
        path = path.joinpath(Path(filename).with_suffix(suffix))
        if path.exists():
            tree = ElementTree.parse(str(path))
            for esd in tree.getroot():
                try:
                    for ed in esd:
                        print('%s\t%s/%s\t%s\t%s' % (filename, esd.attrib['id'], esd.attrib['source'], ed.attrib['original'], labels[filename, esd.attrib['source'], esd.attrib['id'], ed.attrib['slot']]))
                except KeyError:
                    pass"""

