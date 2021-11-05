#!/usr/bin/env python
# coding: utf-8

# # Data and metrics

# Some text here explaining stuff
# - A list maybe

# In[1]:


import mirdata
import librosa
import mir_eval
from tqdm import tqdm


# In[2]:


dataset = mirdata.initialize('groove_midi')


# In[3]:


# dataset.download()  # downloading a dataset to work with


# In[4]:


dataset.validate()  # checking data were downloaded correctly


# In[5]:


scores = {}  # compute scores for a beat tracker
for track_id in tqdm(dataset.track_ids[:15]):  # first 15 files for illustration
    track = dataset.track(track_id)
    audio, sr = track.audio
    ref_beats = track.beats.times
    est_tempo, est_beats = librosa.beat.beat_track(y=audio, sr=sr)
    est_beats = librosa.frames_to_time(est_beats, sr=sr)
    track_scores = {'f1': mir_eval.beat.f_measure(ref_beats, est_beats),
                    'cemgil': mir_eval.beat.cemgil(ref_beats, est_beats)[0]}
    track_scores.update({m:v for m,v in zip(['CMLc', 'CMLt', 'AMLc', 'AMLt'], 
                                            mir_eval.beat.continuity(ref_beats, est_beats))})
    scores[track_id] = track_scores


# In[6]:


scores[track_id], track_id

