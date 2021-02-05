"""
CSC2611 Exercise
Construct various language models and investigate word pair similarity
Wendy Qiu 2021.02.01
"""

import nltk

word_pairs = [["cord", "smile"], ["hill", "woodland"],
              ["rooster", "voyage"], ["car", "journey"],
              ["noon", "string"], ["cemetery", "mound"],
              ["fruit", "furnace"], ["glass", "jewel"],
              ["autograph", "shore"], ["magician", "oracle"],
              ["automobile", "wizard"], ["crane", "implement"],
              ["mound", "stove"], ["brother", "lad"],
              ["grin", "implement"], ["sage", "wizard"],
              ["asylum", "fruit"], ["oracle", "sage"],
              ["asylum", "monk"], ["bird", "crane"],
              ["graveyard", "madhouse"], ["bird", "cock"],
              ["glass", "magician"], ["food", "fruit"],
              ["boy", "rooster"], ["brother", "monk"],
              ["cushion", "jewel"], ["asylum", "madhouse"],
              ["monk", "slave"], ["furnace", "stove"],
              ["asylum", "cemetery"], ["magician", "wizard"],
              ["coast", "forest"], ["hill", "mound"],
              ["grin", "lad"], ["cord", "string"],
              ["shore", "woodland"], ["glass", "tumbler"],
              ["monk", "oracle"], ["grin", "smile"],
              ["boy", "sage"], ["serf", "slave"],
              ["automobile", "cushion"], ["journey", "voyage"],
              ["mound", "shore"], ["autograph", "signature"],
              ["lad", "wizard"], ["forest", "woodland"],
              ["forest", "graveyard"], ["implement", "tool"],
              ["food", "rooster"], ["cock", "rooster"],
              ["cemetery", "woodland"], ["boy", "lad"],
              ["shore", "voyage"], ["cushion", "pillow"],
              ["bird", "woodland"], ["cemetery", "graveyard"],
              ["coast", "hill"], ["automobile", "car"],
              ["furnace", "implement"], ["midday", "moon"],
              ["crane", "rooster"],  ["coast", "shore"], ["gem", "jewel"]]
