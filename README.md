# CCAMD

This repo contains supplementary material for our manuscript called *A Novel Semi-supervised Framework 
for Call Center Agent Malpractice Detection via Neural Feature Learning* to be published in Expert Systems 
with Applications Journal.

## Abstract

The corresponding work presents a practical solution to the problem of call center agent malpractice. A semi-supervised framework comprising of non-linear power transformation, neural feature learning with  k-means  and agglomerative clustering is outlined. We put these building blocks together and tune the parameters so that the best performance was obtained. The data used in the experiments is obtained from our in-house call center. It is made up of recorded agent-customer conversations which have been annotated using a convolutional neural network (CNN) based segmenter. The methods provided a means of tuning the parameters of the neural network to achieve a desirable result. We show that, using our proposed framework, it is possible to significantly reduce the malpractice classification error of a  clustering model (either k-means or agglomerative). By presenting the amount of silence per call as a key performance indicator, we show that the proposed system has increased the efficiency of quality control managers thus enhancing agents performance at our call center since deployment.

## Citation

    @article{OzanIheme20222,
      title={A Novel Semi-supervised Framework for Call Center Agent Malpractice Detection via Neural Feature Learning},
      author={Ozan, Şükrü and Iheme, Leonardo O.},
      journal={Expert Systems with Applications},
      volume={378},
      pages={686--707},
      year={2022},
      publisher={Elsevier}
    }


## File List
1.  [training.csv](data/training.csv) under **data** folder is the training data we used for training our 
proposed frameworks. Each row represents one training sample. The file comprises of 5 columns named record, speech, silence, noise, and music, respectively. These columns hold the corresponding percentage values which were calculated by preprocessing 
call center recordings using  [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter).
    - **record :** This column holds the index values of the data samples.
    - **speech :** This column  holds the percentages of speech segments in data samples.
    - **silence :** This column holds the percentages of silence segments in data samples.
    - **noise :** This column holds the percentages of noise segments in data samples.
    - **music :** This column holds the percentages of music segments in data samples.

2.  [validation.csv](data/validation.csv) under **data** folder is the validation data we used for testing our 
proposed frameworks. This file does have an additional *malpractice* column which represents a label for each data sample in the data set. Each row represents one validation sample. The file comprises of 6 columns named record, speech, silence, noise, music, and malpractice respectively. These columns hold the corresponding percentage values which were calculated by preprocessing 
call center recordings using  [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter).
    - **record :** This column holds the index values of the data samples.
    - **speech :** This column  holds the percentages of speech segments in data samples.
    - **silence :** This column holds the percentages of silence segments in data samples.
    - **noise :** This column holds the percentages of noise segments in data samples.
    - **music :** This column holds the percentages of music segments in data samples.
    - **malpractice :** This column holds a boolean flag value as *TRUE* or *FALSE*. If the value is *TRUE* the data sample is considered to be a malpractice.
3. [20220411_141801.gsm](20220411_141801.gsm) is a hypothetical call center recording between two people,
one acting as a customer and the other as a customer representative.
4. [extract_features.py](extract_features.py) this file generates labels for an input audio file by
using [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter).

