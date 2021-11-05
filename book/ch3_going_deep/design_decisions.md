Design decisions for tempo, beat, and downbeat
==============================================

To this point we hope you have enough of an overview of the **many** different variations of systems for beat and downbeat
tracking. In the following we discuss the importance of the different design choices for these systems, to gain some insights of
the more critical aspects of it.

```{image} ../assets/ch3_going_deep/figs/design_choices.png
---
alt: some of the choices needed for the design of beat/downbeat systems.
```

# Input representation

As mentioned in **SECTION OVERVIEW** several different input representations have been studied in the context of both 
beat and downbeat tracking: harmonic-based (e.g. chroma), event-oriented (e.g. spec-flux), spectrograms, etc. 

** Even though there are works that show advantages of using multiple input representations to increase robustness 
across different music genres {cite}`durand2016feature, Durand2017, Krebs2016,fuentes2018analysis, fuentes2019_structure, zapata2014multi`, results
obtained with spectrograms have shown similar results, and thus is not clear that adding multiple features is an advantage,
especially since it increases the system's complexity considerably (e.g. if the multiple representations are included as an
ensamble of networks like {cite}`durand2016feature, Durand2017, Krebs2016,fuentes2018analysis, fuentes2019_structure`).`

# Granularity

The temporal granularity of the input observations (or temporal grid) relates to important aspects of the design of beat and downbeat tracking systems. 
It determines the length of the context taken into account around musical events, which controls design decisions in the network architecture, such as 
filter sizes in a CNN, or the length of training sequences in an RNN. 

Among the different systems, several granularities have been used. In particular, some systems use either musically motivated temporal grids (such as tatums
or beats) {cite}`durand2016feature, Krebs2016,fuentes2018analysis`  or fixed length frames {cite}`bock2016joint`.  Systems that use beat- or tatum-synchronous
input depend on reliable beat/tatum estimation upstream, so they are inherently more complex, and prone to error propagation. On the other hand, frame-based
systems are not subject to these problems, but the input dimensionality is much higher due to the increased observation rate, which causes difficulties 
when training the models.

It has been shown that RNNs work better with coarser granularities {cite}`fuentes2018analysis` as they have more difficulty to model long temporal 
dependencies which are inherent to finer granularities, whereas CRNNs or LSTMs seem more robust to different granularities.

** What is your takeaway with using the TCNs??????** 


# DNN architecture

Given the same data, input features, post-processing and evaluation scheme different architectures can achieve similar performance,
if the input granularity and architecture is optimized for performance. In other words, architecture matters more because of model 
complexity, portability or interpretability than for performance. 


# Post-processing

The importance of the post-processing stage has been mentioned in multiple works {cite}`Durand2017,Krebs2016, fuentes2018analysis`. 
So far, post-processing with most PGMs discussed in *SECTION PGMS* almost always helps improve the model. However, the importance of this stage 
and how much it improves performance depends on the temporal granularity, the network architecture and the dataset. In particular, for datasets with
mostly Pop music, or very danceable genres post-processing can boost performance considerably. **NOT SUE WHAT I WANNA SAY HERE**