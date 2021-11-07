Table of reference works
========================

```{note}
This table is not exahustive. We did our best to compile relevant works in the past few years, but
we welcome contributions to update this table with missing references or corrections. Please open 
an issue in the main repository!
```
```{tip}
Definitions of the acronyms are at the bottom of this page.
```


```{list-table}
:widths: 15 5 5 5 5 5 5 5
:header-rows: 1

* - Author
  - Year
  - 
  - Task
  -
  - 
  - Approach
  -   
* - 
  -
  - _Tempo_
  - _Beat_
  - _Downbeat_
  - _Features_
  - _Likelihood_
  - _Post-proc_

* - Pinto {cite}`pinto2021user`
  - 2021
  - 
  - ✅ 
  -  
  - log-STFT
  - TCN
  - DBN
* - Steinmetz {cite}`steinmetz2021wavebeat`
  - 2021
  - 
  - ✅ 
  - ✅ 
  - waveform
  - TCN
  - DBN
* - Böck {cite}`bock2020deconstruct`
  - 2020
  - ✅
  - ✅ 
  - ✅
  - log-STFT
  - TCN
  - DBN
* - Böck {cite}`bock2019multi`
  - 2019
  - ✅
  - ✅ 
  - 
  - log-STFT
  - TCN
  - DBN
* - Davies {cite}`matthewdavies2019temporal`
  - 2019
  - 
  - ✅ 
  - 
  - log-STFT
  - TCN
  - DBN
* - Fuentes {cite}`fuentes2019_microtiming`
  - 2019
  - 
  - ✅ 
  -  
  - mel-STFT + ODF
  - RNNs
  - CRFs
* - Fuentes {cite}`fuentes2019_structure`
  - 2019
  - 
  -  
  - ✅ 
  - CH + SF
  - CRNNs
  - SCCRFs
* - Fuentes {cite}`fuentes2018analysis`
  - 2018
  - 
  -  
  - ✅ 
  - CH + SF
  - CRNNs
  - DBNs
* - Cheng {cite}`cheng2018convolving`
  - 2018
  - 
  - ✅  
  -  
  - mel logSTFT
  - RNNs
  - DBNs
* - Vogl {cite}`vogl2017drum`
  - 2017
  - 
  - ✅  
  -  
  - logSTFT
  - CRNNs
  - peack-pick
* - Srinivasamurthy {cite}`ajay2017`
  - 2017
  - 
  - ✅  
  - ✅  
  - 2D-SF
  - GMMs
  - DBNs
* - Durand {cite}`Durand2017`
  - 2017
  - 
  -   
  - ✅  
  - LFS+ODF+MCQT+CH
  - CNNs
  - HMMs
* - Durand {cite}`Durand2016`
  - 2016
  - 
  -   
  - ✅  
  - LFS+ODF+MCQT+CH
  - CNNs
  - CRFs
* - Holzapfel {cite}`holzapfel2016bayesian`
  - 2016
  - 
  - ✅  
  - ✅  
  - mel log STFT
  - CNNs
  - DBNs
* - Krebs {cite}`Krebs2016`
  - 2016
  - 
  -   
  - ✅  
  - CH+ODF
  - Bi-GRUs
  - DBNs
* - Durand {cite}`Durand2016`
  - 2016
  - 
  -   
  - ✅  
  - LFS+ODF+MCQT+CH
  - CNNs
  - CRFs   
* - Böck {cite}`bock2016joint`
  - 2016
  - 
  - ✅  
  - ✅  
  - mel log STFT
  - Bi-LSTMs
  - DBNs  
* - Srinivasamurthy {cite}`ajay2016`
  - 2016
  - 
  - ✅  
  - ✅  
  - 2D-SF
  - GMMs
  - DBNs  
* - Nunes {cite}`Nunes2015`
  - 2015
  - 
  - ✅  
  - ✅  
  - SF + AC + STFT
  - template
  - HMMs     
* - Fillon {cite}`fillon2015`
  - 2015
  - 
  - ✅  
  -  
  - ODF + tempogram
  - 
  - CRFs   
* - Durand {cite}`Durand2015`
  - 2015
  - 
  -   
  - ✅ 
  - LFS+ODF+MFCC+CH
  - DNNs
  - average   
* - Srinivasamurthy {cite}`ajay2015`
  - 2015
  - 
  - ✅   
  - ✅ 
  - 2D-SF
  - GMMs
  - DBNs   
* - Krebs {cite}`krebs2015inferring`
  - 2015
  - 
  - ✅   
  - ✅ 
  - log SF
  - GMMs
  - DBNs    
* - Krebs {cite}`Krebs2015`
  - 2015
  - 
  - ✅   
  - ✅ 
  - mel log STFT + 2D-SF
  - GMMs + RNNs
  - DBNs  
* - Korzeniowski {cite}`korzeniowski2014probabilistic`
  - 2014
  - 
  - ✅   
  - 
  - logSTFT
  - Bi-LSTMs
  - DBNs  
* - Srinivasamurthy {cite}`ajay2014`
  - 2014
  - 
  - ✅   
  - ✅ 
  - tempogram + ODF
  - SSM
  - peak-pick
* - Holzapfel {cite}`Holzapfel2014`
  - 2014
  - 
  - ✅   
  - ✅ 
  - 2D-SF
  - GMMs
  - DBNs  
* - Durand {cite}`durand2014enhancing`
  - 2014
  - 
  -   
  - ✅ 
  - CH + F0 + MFCCs
  - 
  - SVMs  
* - Böck {cite}`Bock2014`
  - 2014
  - 
  - ✅  
  -  
  - mel log STFT
  - RNNs
  - DBNs  
* - Zapata {cite}`zapata2014multi`
  - 2014
  - 
  - ✅  
  -  
  - SF + EF + CSD + HF + PSF + MAF
  - 
  - HMMs
* - Krebs {cite}`krebs2013rhythmic`
  - 2013
  - 
  - ✅  
  - ✅ 
  - SF
  - GMMs
  - DBN
* - Hockman {cite}`hockman2012one`
  - 2012 
  - 
  -  
  - ✅ 
  - CH + ODF + SC
  - 
  - SVR?????    
* - Khadkevich {cite}`khadkevich2012probabilistic`
  - 2012 
  - 
  - ✅ 
  - ✅ 
  - CH + SF
  - multiple-pass decoding
  - HMMs  
* - Degara {cite}`degara2012reliability`
  - 2011 
  - 
  - ✅ 
  - 
  - CSF
  - comb filterbank
  - HMMs    
* - Peeters {cite}`peeters2010simultaneous` 
  - 2010
  -    
  - ✅ 
  - ✅ 
  - CH + SF
  - template 
  - HMMs 
* - Papadopulos {cite}`papadopoulos2010joint`
  - 2010 
  - 
  - 
  - ✅ 
  - CH
  - template 
  - HMMs 
```

- **CH:** chroma
- **CSD:** complex spectral difference
- **CSF:** complex spectral flux
- **CRFs:** Conditional Random Fields
- **EF:** energy flux
- **F0:** fundamental frequency
- **HMMs:** Hidden Markov models
- **GMMs:** Gaussian mixture models
- **DBN:** Dynamic Bayesian network
- **DNNs:**: Deep Neural Network
- **LFS:** Low frequency spectrogram
- **MAF:** Mel auditory feature
- **MFCCs:** Mel frequency cepstral coefficients
- **ODF:** onset detection function
- **PSF:** phase slope function
- **SCCRF:** Skip-Chain Conditional Random Fields  
- **SF:** spectral flux
- **SSM:** self-similarity matrix   
- **SVMs:** support vector machines

