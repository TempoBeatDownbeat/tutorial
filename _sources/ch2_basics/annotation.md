(annotatemap)=
# How do we annotate?

Within the context of the kind of data-driven approaches for the estimation
of tempo, beats, and downbeats we explore in this tutorial, a critical aspect is 
how to acquire some data to learn from, and ultimately to evaluate upon.

The "data" in question refers to annotations of beat and downbeat
locations from which either: i) a global value tempo for a roughly
constant tempo piece of music; or ii) a local tempo contour can be derived.

The workflow by which beat and downbeat annotations can be obtained
typically involves an interative process departing from an initial estimate,
e.g., marking beat locations only, and then correcting timing errors, followed by
a labelling process to mark the metrical position of each beat. 

This initial estimate could be obtained by hand, i.e., by tapping
along with the musical audio excerpt in a software such as `Sonic Visualiser`,
or alternatively, by running an existing beat estimation algorithm, e.g., from `madmom` or `librosa` and then loading this annotation layer. 

We can consider these approaches to be **manual** or **semi-automatic**.
Within the beat tracking literature and the creation of datasets,
both approaches have been used. To begin with, we'll focus on the fully manual approach.


(annotatemap:example)=
## Manual annotation example
The figure below gives an illustration of a typical manual annotation process 
in `Sonic Visualiser`. 
The excerpt in quesiton is the straightforward musical excerpt 
from the previous section that we've already here, and is around 25s in duration
with a constant tempo and 4/4 metre.

```{note}
The clip is sped up by a factor of 5, so you shouldn't expect to hear anything!
``` 

```{figure} ../assets/ch2_basics/figs/annotation_process.gif
---
alt: Annotation example in Sonic Visualiser.
width: 1200px
align: center
name: annotate
---
Manual annotation example in Sonic Visualiser.
```


The stages of the process as follows:
1. Listening to the excerpt and tapping along in real time to mark the beat locations. 
Note, usually it's not possible to start tapping straight away as it takes some time
for a listener to infer the beat (even for familiar pieces of music). 
In this case, the tapper begins at the start of the 3rd bar, meaning the first two bars will need to be filled in later. 
2. Having completed one real-time pass over the musical excerpt, the next stage is to go 
back and listen again, but this time with the beat annotations rendered as audible 
clicks of short duration. As becomes clear from watching the clip above, 
the timing of the taps is not super precise! As such many of the beats need to be altered to compensate for temporal imprecision. 
While this could simply be slopping timing on the part of the 
tapper, in practice it is likely a a combination of human motor noise and jitter [^1] in the acquisition of the keyboard taps. 
In this case, there are no duplicated or missing taps 
(besides those of the first two bars), and so the tap-editing operations are exclusively 
performed by shifting the annotations -- using the waveform as a guide -- and listening back for perceptual accuracy.  
3. Once done, the annotations for the first two bars are marked approximately by hand 
and the main listening and editing process in the previous step is applied again.
4. In this excerpt, there is a constant 4/4 metre throughout, thus it is straightforward 
to have Sonic Visualiser apply two-level labels to the beat locations, 1.1, 1.2., 1.3, 
1.4, 2.1, 2.2., etc. where each 'x.1' corresponds to a downbeat. However, in more complex cases containing changes in metre, it may be necessary to edit the annotation labels by hand. Having performed this labelling, a final listen and minor edits are made, the process is complete and the annotations can be exported. 

(annotatemap:perspectives)=
## Perspectives on manual annotation

At this point it is worth a little reflection the practical aspects of the manual annotation and editing process. The musical excerpt is under 25s in duration
yet the total time taken to complete the annotation is a little over 4 minutes 
(approximately a **10x** overhead). Concerning the number and type of edits, we find 8 
insertions: corresponding to the first two bars, 0 deletions, and 21 shifting operations (including some beats shifted more than once) for a total of 32 beat annotations.  

For brevity, the annotation corrections were made rather quickly with an emphasis on 
approximate perceptual accuracy as opposed to extremely precise hand-labelling. 
A more "forensic" analysis of the waveform (perhaps supported by other 
time-frequency representations) and additional listening back would further increase the 
annotation time. Of course, the better the real-time taps the fewer repeated listens and 
editing operations, but in the limit even when no edits are required, this would still 
requires two complete listens (once to tap, once to confirm). 

If we then begin to consider more complex musical material, e.g., with challenging
musical properties such as syncopation, expressive timing, metrical changes, lack of 
percussion etc. together the cognitive burden of annotating and annotator fatigue it's 
easy to imagine that the annotation process could be 1-2 orders of magnitude more 
time-consuming. As we've seen in the case of the expressive excerpt in the previous
section, specific aspects of the annotation may require access to a musical score (if it exists and is available). 

```{note}
For comparison, the expressive piece in its entirety (4m51s) which is used extensively in {cite}`pinto2021user` took around 15 hours to annotate (spread over 3 days), and included frequent discussion with musical experts. 
```

**Why does this matter?** This matters in the context of deep learning, 
since we'd typically like to acquire as much high-quality annotated data as possible
when training models. Thus if it is very expensive and time-consuming to accurately annotate then this may intrinsically limit the potential of deep learning approaches. 

Furthermore, it is also worthwhile to consider the type of musical material will be annotated since it's not just about "how much" but also "what." While straightforward
musical excerpts like the one shown above are essentially easy to annotate
there may be little benefit in annotating this kind of musical content since it
is already "trackable." On this basis, the added benefit in annotation likely
resides in more challenging musical material which takes longer to annotate
and may be more ambiguous.  


(annotatemap:automating)=
## Towards automating the annotation process
Given the labour-intensive nature of the annotation process, it is useful
to consider possible steps for at least partial automation. 

* Instead of performing real-time tapping to make an initial estimate
of the beat, it's possible to execute an existing beat tracking algorithm. 
For easier examples this may be highly beneficial as the issues
relating to (human) motor noise and jitter can be avoided. 
However, we must accept that the temporal accuracy of the beat locations will be 
quantised to the frame rate at which the beats are estimated (e.g., every 10ms),
and so may still require some fine temporal adjustment. Perhaps more
troubling is that the choice of metrical level will be determined
by an algorithm and thus may bias the annotator who may have 
otherwise chosen to tap at a different metrical level. 
Finally, if the material is extremely complex, and "beyond the scope"
of what existing approaches can reliably annotate, then there
may be very little value in an initial automatic first pass
if all beat estimates need subsequent correction.

* A promising approach for the automatic correction of annotations
appeared in ISMIR 2019 paper by Drieger et al {cite}`DriedgerSHM19_TapCorrect_ISMIR` which relies on a smart snapping of manually-tapped beat locations
to peaks in a beat activation function (i.e., the prediction
of a deep neural network) or an onset detection function, and is shown below.   
While shown to be successful in improving the subjective quality
of annotations according to musical experts, the full work-flow 
in the paper still recommends a final human intervention to confirm
the precise annotations. Furthermore the success of the approach
depends on the presence of peaks in the beat activation function 
at, or near, the "correct" locations. 

```{figure} ../assets/ch2_basics/figs/tapcorrect.png
---
alt: Tap Correction Procedure Overivew.
width: 400px
align: center
name: tapcorrect
---
Tap Correction Procedure Overivew. Image taken from {cite}`DriedgerSHM19_TapCorrect_ISMIR`
```

* Finally, in the case of sequenced music, (e.g., electronic dance music) it may possible
to be obtain a global tempo, beat, and downbeat locations automatically from
the project settings. For example, in the Giant Steps tempo dataset {cite}`knees2015ismir`
tempo labels were obtained directly from the online music service Beatport, but 
but many cases were found to be ambiguous, or incorrect and required relabelling.
Of course, it's worth remembering that not all sequenced music need be of constant tempo and metre, nor are they necessarily easy to analyse (even in cases of constant tempo).


(annotatemap:datasets)=
## Example annotated datasets
To provide some additional perspective on annotation, and continue the thread
from the previous point about the Giant Steps tempo dataset, we can take a look at a 
small set of example datasets and highlight some relevant insights concerning
the type of musical material they contain, and the manner in which they 
were annotated. 

```{note}
This list is in no way intended to be exhaustive,
merely a subset that help support the tutorial content. For a list
of MIR datasets of all kinds, please see Alexander Lerch's Audio Content Analysis [website](https://www.audiocontentanalysis.org/data-sets/). 

We'll learn more about `mirdata` in the practical part of the tutorial
and the importance of loading and working with the correct version of a dataset. 
```


* **Hainsworth** {cite}`hainsworth2004particle`
The **Hainsworth** dataset comprises 222 musical excerpts of around 1 minute
each in length, which were organised into six categories by genre label:
rock/pop, dance, jazz, folk, classical, and choral. It was created by 
Stephen Hainsworth as part of his PhD thesis on automatic music transcription.
He produced the annotations in a two-stage process, first
recording initial taps and then subsequently using a custom interface
in Matlab to load and then manually correct the annotations guided
by a time-frequency representation. Note, this dataset 
pre-dates even the earliest versions of Sonic Visualiser by a few years 
and thus Stephen needed to make his own tool for annotation correction. 
Of particular note was the inclusion of 20 or so choral examples
which drew first attention in the beat tracking community to a particularly 
challenging class of musical signals to annotate, and analyse. 
The Hainsworth dataset remained in its original incarnation 
until 2014 when Böck et al {cite}`bock2014ismir` performed
a set of revisions on the beat and downbeat annotations
to correct some errors, and resulted in an increase in performance. 

* **HJDB** {cite}`hockman2012one`
The **HJDB** dataset has 236 excerpts taken from 80s and 90s
electronic dance music, specifically in the sub-genres of hardcore,
jungle, and drum and bass. Initially it was only annotated
in terms of downbeat positions (with no beat positions).
Subsequently a set of beat annotations, and revisions to the 
downbeats were made, and the current "reference" set of annotations
can be found among the supplementary material for {cite}`bock2019multi`
which can be found [here](https://github.com/superbock/ISMIR2019).
This dataset is somewhat noteworthy as being among the first datasets
to push back against the notion that electronic dance music is essentially
very straightforward from the perspective of computational
rhythm analysis.  

* **SMC** {cite}`holzapfel12taslp`
The **SMC** dataset contains 217 excerpts of 40s each in duration and 
was designed with a methodology for selecting the audio examples to annotate.
Specifically, it was based on the idea of building a dataset
out of musical audio examples that would be difficult for (then) state-of-the-art
beat tracking algorithms to analyse. Normally, we can discover
when the state of the art fails by running it on an audio excerpt
for which ground truth annotations exist, and then use one or more
evaluation methods to estimate performance. Thus, the challenge here
was to identify these kinds of excerpts **without** annotating them first.
The approach was to build a committee of "good, but different" beat
tracking algorithms and run them over a large collection of unannotated
musical audio signals. Next, a subset of excerpts was selected based
on the lack of concensus among the estimates of each committee member.
In effect, if all algorithms gave a different answer, then
this would at least hint at interesting properties in the music
that would make it worthwhile to annotate. 
Despite this dataset being compiled around 10 years ago (and thus predating
almost all work in deep learning applied to rhythm), it remains
highly challenging even for the most recent state-of-the-art
approaches of the kind we'll explore later in the tutorial.

(annotatemap:summary)=
## Summary
* Annotation is hard! 
* It takes a long time, and the more challenging
the material to annotate the greater the likelihood of this being
helpful for learning. 
* On the plus side, annotation is a fantastic
way to learn about the task of beat and downbeat estimation
so it's a really great excercise. 
* We always need more data,
so do consider doing some annotating!  
* As hard as we try, annotation "mistakes" are made, so they made need correcting.
* This makes comparative evaluation more challenging, so it's always worthwile
to ensure you are using the most up to date version of any annotations.  

[^1]: Recent work by Anglada-Tort et al {cite}`anglada2021repp` has propsed a means to eliminate jitter through a novel signal acquisition approach.

