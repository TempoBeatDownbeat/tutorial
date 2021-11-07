(dnns:overview)=
# Deep learning approaches 

The goal of these sections is to provide the theoretical underpinnings of deep learning models or _deep neural networks_ (DNNs) commonly used for tempo, beat and downbeat tracking, to build an understanding of why these models work well in these tasks and what are their limitations. Here we discuss the main concepts and definitions,
but for those interested in reading more about the use of DNNs in audio processing and MIR related tasks, we suggest them to take a look at works like McFee {cite}`McFee2018`, Purwins et al. {cite}`purwins2019deep` and Choi et al. {cite}`choi2017tutorial`.

The field moved quite fast these past years and the amount different design choices and approaches can be somehow overwhelming!
We tried to come up with a way of summarizing the different moving parts of beat and downbeat deep learning systems to help
digest this, as we explain below. 

Recent beat and downbeat tracking approaches can be structured in three main stages: 1) A first stage of _low-level feature_ computation or _feature extraction_, where feature vectors that represent the content of musical audio are 
extracted from the raw audio signal (e.g. Spectrogram, Chromagram); 2) A second step that usually consists of a stage of feature learning, whose outcome is an activation function that indicates the most likely candidates
for beats and/or downbeats among the input audio observations; 3) Finally, a post-processing stage is often used, usually consisting of a probabilistic graphical model which encodes some relevant musical rules 
to select the final beat/downbeat candidates.

```{figure} ../assets/ch3_going_deep/figs/diagram.png
---
alt: General pipeline commonly used for beat and/or downbeat tracking systems.
---
General pipeline commonly used for beat and/or downbeat tracking systems.
```


Different alternatives were proposed for the distinct stages among beat and downbeat tracking systems. Here we give an overview of the main ideas 
presented in the literature.

## Feature extraction 
It is common to exploit music knowledge for feature design using signal processing techniques. 
The three most explored categories of musically inspired features in the literature for both beat and downbeat estimation are: chroma (CH) {cite}`hockman2012one, papadopoulos2010joint, peeters2010simultaneous,  khadkevich2012probabilistic, Krebs2016, fuentes2018analysis, fuentes2019_structure` ---used to reflect the harmonic content of the signal---, 
onset detection function (ODF) {cite}`hockman2012one, zapata2014multi, Durand2015` or spectral flux (SF) {cite}`krebs2013rhythmic, khadkevich2012probabilistic, Holzapfel2014, fuentes2018analysis, fuentes2019_structure` ---as event-oriented indicators--- and timbre inspired features {cite}`hockman2012one, ajay2014, durand2014enhancing` such as spectral coefficients or MFCCs. 
For beat, the main features exploited are those related to event-oriented indicators, assuming that changes in the spectral energy relate to 
beat positions are {cite}`degara2012reliability, krebs2013rhythmic, Holzapfel2014, krebs2015inferring, Nunes2015, ajay2015, fillon2015`. For downbeat, harmonic-related features showed to be relevant to estimate downbeats reliably across music genres.

The feature extraction is usually based on a single feature {cite}`papadopoulos2010joint, peeters2010simultaneous, zapata2014multi, Holzapfel2014, korzeniowski2014probabilistic, Krebs2015, ajay2015, Bock2014`, with some exceptions exploiting more than one music property at the same time {cite}`Durand2015, durand2016feature, Durand2017, Krebs2016,fuentes2018analysis, fuentes2019_structure, zapata2014multi`, which results in systems robust to different music genres {cite}`Durand2017`. 
Recently, approaches based on deep learning exploring combinations of logarithmic spectrograms with different resolutions showed to perform competently {cite}`bock2016joint, korzeniowski2014probabilistic,bock2020deconstruct, bock2019multi`.

```{figure} ../assets/ch3_going_deep/figs/features_example.png
---
alt: Example of features used for downbeat tracking.
---
Example of features, from left to right: melodic constant-Q transform, onset detection function, chromagram, low-frequency spectrogram. Adapted from {cite}`Durand2017`.
```


## Likelihood estimation

The objective of this stage is to map the input representation into a beat/downbeat likelihood that indicates which are the most likely candidates to be a beat or a downbeat
in a given temporal sequence. There are two main groups of approaches in this respect: the first one uses "heuristics" to perform the mapping, while the second
group exploits machine learning approaches. The latter group is the most popular one in the literature in the last years and also the state of the art. 

The estimation of a likelihood with heuristics is performed differently depending on the features used. For instance, a common approach is to pre-define a template of _expected_ features such as 
spectral-flux or chroma, and to measure the distance between this template to the features computed from the audio signal {cite}`peeters2010simultaneous, Nunes2015`. Within the group of 
machine learning approaches, we could identify two subgroups: a first one that exploits "traditional" learning techniques and a second one with focus on deep learning models.

```{figure} ../assets/ch3_going_deep/figs/rhythmic_patterns.png
---
alt: Example of rhythmic patter learning.
---
Example of rhythmic patter learning from {cite}`krebs2013rhythmic`.
```

Before deep learning, machine learning systems often focus on recognizing rhythm patterns in data, for instance by using _Gaussian Mixture Models_ (GMM) and k-means 
{cite}`ajay2015, ajay2016, ajay2017, krebs2015inferring, krebs2013rhythmic, Holzapfel2014, Nunes2015`. This usually required making some assumptions of 
style or genre (e.g. to define the length of the patterns to be learned), and for these models to be effective the music should have distinctive rhythmic patterns. 
Deep learning approaches propose an alternative to such limitations given their capacity to learn complex function mappings, and systems exploiting DNNs have became the state of the art in
recent years {cite}`jia2019deep`.  

```{figure} ../assets/ch3_going_deep/figs/feature_extraction.png
---
alt: Example of likelihood estimation.
---
Different stages of feature extraction. Left: input spectrogram, middle: intermediate DNN outputs, right: the final beat and downbeat likelihoods. Adapted from {cite}`bock2016joint`.
```


## Inference
 
The aim of this stage is to obtain the final downbeat sequence by selecting the most likely candidates in the downbeat likelihood given some model or criteria. Probabilistic graphical models (PGMs) are the most used 
post-processing techniques since 2010. This might be due to two main reasons: PGMs offer a flexible framework to incorporate music knowledge and then exploit interrelated structure {cite}`papadopoulos2010joint, peeters2010simultaneous`, and the 
Bar Pointer Model (BPM) {cite}`Whiteley2006` stands as a very effective and adaptable model for meter tracking, being popular for beat and downbeat tracking. 

PGMs proved to be adaptable to cultural-aware systems in diverse music cultures {cite}`ajay2014, Nunes2015, Holzapfel2014`, being for instance extendable to track longer meter cycles and different meters than the widely explored 3/4 and 4/4 {cite}`ajay2016, ajay2017`. 
Considerable efforts have been made towards improving the use of these models in practice, by reducing computational cost via an efficient state-space definition {cite}`Krebs2015` or proposing _sequential Monte Carlo_ methods (also called _particle filters_) for inference {cite}`ajay2015, krebs2015inferring`. 


## Next

After discussing the usual pipelines of beat and downbeat tracking systems, let's dig a bit more in the different deep learning architectures used for likelihood estimation these past years.


 