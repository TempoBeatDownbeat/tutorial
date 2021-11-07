# Concluding remarks

It's been a lot of fun, and a lot of work (most of it right at the last minute!) 
to put this tutorial together. Our sincere hope is that the
content becomes a useful resource for anyone looking
to become immersed into the topics of tempo, beat, and downbeat
estimation from musical audio signals.
We have sought to provide a broad outlook across this domain
touching upon annotation, model design (both past and present),
and evaluation. Indeed, we maintain that undertaking
successful research in this area requires effort and understanding
across this pipeline.  

In particular, we hope that the practical code examples
provide the means for anyone to build, train, and test
their own systems (with their own annotated data!) 
for analysing the metrical structure of music signals, 
and even for combining
them with other MIR techniques towards 
even more holistic approaches to the analysis
of musical audio. 

Even though the provided multi-task formulation using
TCNs is representative of the current state of the art,
it is extremely important to recognise that
the state of the art in a community like ISMIR is in 
constant evolution. To this end, it would 
remiss not to consider at least some (but certainly not all) 
of the open challenges which remain in this area. 

```{note}
Many of the open challenges in rhythm analysis are not new! 
```

* Moving away from constant tempo, 4/4 metre, western music:
With these advanced deep learning models, we may be moving
very close to a position where, up to the limit of metrical
ambiguity, the tasks of tempo, beat, and downbeat estimation
are close to being solved. Some open and interesting 
challenges lie include: 
	* tracking (tappable) pieces with high musical expression 
	* building inference models that can cope with changes in metre
	* pursuing models which are effective in multiple
	musical cultural contexts (i.e., without need to retrain
	from scratch for a singular nonwestern specific approach). 

* Incoporating long-term temporal dependencies: 
Given the effort in annotation (especially at scale)
it's not an accident that many annotated datatsets contain
short excerpts. This has the negative outcome that 
current state-of–the-art systems really aren't able
to track the beat/downbeat in a structurally coherent way. 
Thus, there is a huge opportunity in building models
which can understand temporal inerrelations at longer
time scales. This may exist at the point of the 
network architecture and/or at inference. 

* Strategies for adaptation to new content: This can
almost be understood as *What should I do when (even)
the state of the art doesn't work?* There is an important
opportunity here for developing learning techniques
which can readily adapt to new content given only
with minimal new information, e.g., having a user
tap a couple of bars only, and update the weights
of the network so this new specific piece can be 
analysed to a very high degree of accuracy.
Such an approach also has important implications
for semi-automatic annotation. 

* Decoupling from supervised learning: So much of the deep learning
research in this field depends heavily on high quality annotated data
and supervised learning. As we've seen, annotation in general is hard, and 
annotation that's useful for learning is even harder. Thus
a grand challenge may be the pursuit of approaches
which eliminate entirely or greatly minimise the
need for supervised learning.  


