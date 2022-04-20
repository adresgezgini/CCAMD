# CCAMD

This repo contains supplementary material for our manuscript called *A Novel Semi-supervised Framework 
for Call Center Agent Malpractice Detection via Neural Feature Learning* to be published in Expert Systems 
with Applications Journal.

## File List
1.  [training.csv](data/training.csv) under **data** folder is the training data we used for training our 
proposed frameworks. Each row represents one training sample. The file comprises of 5 columns named record, speech, silence, noise, and music, respectively. 
    - **noise :**
    - **noise :**
2.  [validation.csv](data/validation.csv) under **data** folder is the training data we used for training our 
proposed frameworks.
3. [20220411_141801.gsm](20220411_141801.gsm) is a hypothetical call center recording between two people,
one acting as a customer and the other as a customer representative.
4. [extract_features.py](extract_features.py) this file generates labels for an input audio file by
using [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter).

