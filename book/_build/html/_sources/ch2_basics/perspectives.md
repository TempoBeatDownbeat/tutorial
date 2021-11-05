Perspectives
============

<span style="color:blue"> **MD:** _In this section, I think it will be really good to pause for a 
moment and look at the high-level distinction between a non-deep approach
and the transition towards deep learning approaches.
In particular, there's a nice figure that I've used in the past
that contrasts a spectral flux function with a beat activation function
I think this is a really powerful demonstration of what we're aiming
to do, which is to try to make inference as easy as possible._</span>


Having looked in a high-level way at a baseline signal processing-based approach
for beat estimation, it's possible to constrast this with the potential,
and arguably, what might be the naïve expectation of a deep learning approach.

In the example below we can observe the profound difference between the spectral flux 
shown in the middle plot and the output of a trained deep neural network in the lower
plot. 

Spectrogram with overlaid beat annotations
![](../assets/ch2_basics/figs/spectrogram.png)

Superflux function with overlaid beat annotations
![](../assets/ch2_basics/figs/spectral_flux.png)

Neural network output with overlaid beat annotations
![](../assets/ch2_basics/figs/network_output.png)


In the former, the means for recovering the beat locations relies on identifying 
peaks among the superflux function are (quasi-)periodic, strong, and thus likely
to correspond to beat information. Whereas in the output of the trained network,
there is essentially just a single set of peaks that clearly correspond 
to the beat locations, and in this way a simple peak-picking approach would be 
sufficient to return a reliable beat output. 

Of course, this example has been cherry-picked on purpose since it represents
a super-ideal. As we'll see moving forward, things are not always quite so
simple.  
