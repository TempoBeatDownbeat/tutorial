#!/usr/bin/env python
# coding: utf-8

# # How do we evaluate?

# Having looked at annotation and some baseline approaches for the estimation
# of beat and tempo from music signals, let's turn our attention towards
# evaluation. 
# 
# So, why do we conduct evaluation?
# 
# * To know the performance of an algorithm.
# * To compare against other algorithms.
# * To understand its strengths and weakness towards building better models.
# 
# In some sense we've been doing a kind of implicit evaluation already by listening
# to the output of beat estimates mixed back with the input musical audio signal,
# albeit not in a quantitative way. 
# Likewise performing annotation is accompanied by a kind of constant evaluation 
# (i.e., "accuracy-maximizing") process which assesses the choice of metrical level, 
# the phase of the beats, and their temporal localisation.  
# 
# Indeed, there is nothing wrong *per se* with doing evaluation this way and 
# in line with our understanding the beat as a perceptual construct, this 
# is our fall-back... essentially "does it sound right when I hear it played back?"
# 
# However the design and execution of large scale evaluation of beat (or downbeat)
# tracking algorithms where rating scales are used to grade performance 
# is time-consuming and expensive (it requires real-time listening possibly multiple times).
# It also difficult to repeat in an exact way, and to determine 
# a set of unambiguous judging criteria.
# 
# Within the beat tracking literature, more effort is devoted to the use 
# of objective evaluation methods which can relate a sequence of beat annotations
# (the so-called "ground truth") with a sequence of estimates from an algorithm (or a tapper),
# as shown in the figure below. In the ideal we're looking for some 
# number (or set of numbers) that we can calculate by comparing
# the two temporal sequences in such a way as to reflect general consensus
# of human listeners. 
# 
# Whether that goal is realisable or not, is certainly debatable 
# (we could even have that debate in the discussion part of the tutorial!)
# but for now, let's explore a few (of the many) existing objective approaches.
# 
# Following the approach in the previous parts of the basics section, 
# we'll avoid notation and instead rely on intuition and observation. 

# ```{figure} ../assets/ch2_basics/figs/objective_v_subjective.png
# ---
# alt: Objective vs Subjective
# width: 600px
# align: center
# name: eval_objective_v_subjective
# ---
# Overview of objective methods vs subjective ratings
# ```
# 
# 

# ## F-measure

# Among the most used methods is the F-measure, which is widely used
# in onset detection, structural boundary detection and other 
# temporal MIR tasks. A high-level graphical overview is shown below.
# 
# ```{figure} ../assets/ch2_basics/figs/fmeasure.png
# ---
# alt: F-measure
# width: 600px
# align: center
# name: eval_fmeasure
# ---
# F-measure
# ```
# 
# Typically we use a tolerance window of +/- 70ms around 
# each ground truth annotation. Beats that uniquely
# fall into these tolerance windows are **true positives** (or **hits**),
# with any additional beats in the tolerance window
# or those outside of any tolerance window are
# counted as **false positives**, and any empty
# tolerance windows with no beats are counted as **false negatives**.
# With these quantities we can calculate the **precision**
# and **recall** and then the F-measure.
# 
# However, if we choose the vary the size of the tolerance
# window we can dramatically change these quantities.
# Too wide, and we'll treat very poorly localised
# beats as accurate, and too narrow we may mark
# perceptually accurate beats as errors.
# 
# ```{figure} ../assets/ch2_basics/figs/tol.gif
# ---
# alt: F-measure anim
# width: 450px
# align: center
# name: eval_fmeasure_anim
# ---
# Animation of F-measure tolerance window
# ```
# 
# Indeed the use of a fixed tolerance window means that
# wherever the beats are inside, they're considered accurate.
# We can push this to an absurd extreme, by 
# anticipating the even beat annotations by just under 70ms
# and offseting the odd beat annotations by the same amount.
# 
# 

# In[1]:


import warnings
warnings.simplefilter('ignore')
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import mir_eval
import matplotlib.pyplot as plt
import scipy.stats


filename = '../assets/ch2_basics/audio/easy_example'

sr = 44100
FIGSIZE = (14,3)

# read audio and annotations
y, sr = librosa.load(filename+'.flac', sr = sr)
ref_beats = np.loadtxt(filename+'.beats')
ref_beats = ref_beats[:,0]
bad_beats = ref_beats.copy()
bad_beats[::2] = bad_beats[::2]-0.069
bad_beats[1::2] = bad_beats[1::2]+0.069

y_good_beats = librosa.clicks(times=ref_beats, sr=sr, click_freq=1000.0, 
                              click_duration=0.1, length=len(y))

y_bad_beats = librosa.clicks(times=bad_beats, sr=sr, click_freq=1000.0, 
                              click_duration=0.1, length=len(y))

plt.figure(figsize=FIGSIZE)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(ref_beats, 1.1*y.min(), 1.1*y.max(), label='Good Beats', color='r', linestyle=':', linewidth=2)
plt.vlines(bad_beats, 1.1*y.min(), 1.1*y.max(), label='Bad beats', color='green', linestyle='--', linewidth=2)

plt.legend(fontsize=12); 
plt.title('Easy Example: audio waveform with "good" beats and "bad" beats', fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Time (s)', fontsize=13)
plt.legend(fontsize=12); 
plt.xlim(1.5, 10);


# Let's listen first the unmodified beat annotations.

# In[2]:


ipd.Audio(0.6*y[1*sr:10*sr]+0.25*y_good_beats[1*sr:10*sr], rate=sr)


# and then the limit of accurate f-measure

# In[3]:


ipd.Audio(0.6*y[1*sr:10*sr]+0.25*y_bad_beats[1*sr:10*sr], rate=sr)


# We can confirm the f-measure score using `mir_eval`

# In[4]:


print('Fmeasure:', round(mir_eval.beat.f_measure(ref_beats, bad_beats), 3))


# Of course, this is a deliberately bad output at the limit
# of what the f-measure calculation will allow, and it's
# probably better not to dwell on cases like this, but 
# rather think that this seemingly big tolerance window
# is useful to catch either: i) natural human variation in tapping
# and not punish it, or ii) contend with cases like arpeggiated chords
# where it's difficult to mark a single beat location. 

# ## Cemgil
# 
# An alternative to using a "top-hat" tolerance window is instead to use a Gaussian distribution 
# placed around each annotation. Here we no longer have the notion of a true or false positive,
# but rather each beat can be assigned the value corresponding to the Guassian located
# at the nearest annotation.
# In this way, poor localisation like in the "bad beats" example
# would be punished since the value of the Gaussian (with standard deviation 40ms as in 
# the original implementation) at an offset of +/-70ms will be rather low. 
# 

# ```{figure} ../assets/ch2_basics/figs/cemgil.png
# ---
# alt: Cemgil
# width: 600px
# align: center
# name: eval_cemgil
# ---
# Cemgil evaluation approach
# ```
# 

# We can verify this with `mir_eval` 

# In[5]:


print('Fmeasure:', round(mir_eval.beat.cemgil(ref_beats, bad_beats)[0], 3))


# What's potentially of interest here is that through the combination
# of these two scores we can begin to make some interpretation about
# the qualitative nature of the relationship between estimated beats
# and annotations.

# ## Continuity-based

# ```{figure} ../assets/ch2_basics/figs/continuity1.png
# ---
# alt: Continuity overview
# width: 600px
# align: center
# name: eval_continuity1
# ---
# Overview of contuity based approach
# ```
# 
# ```{figure} ../assets/ch2_basics/figs/continuity_levels.png
# ---
# alt: Continuity levels
# width: 600px
# align: center
# name: eval_cont_levels
# ---
# Example alternative metrical levels for continuity-based evaluation
# ```
# 

# ## Objective vs Subjective

# 
# 
# 

# In[ ]:




