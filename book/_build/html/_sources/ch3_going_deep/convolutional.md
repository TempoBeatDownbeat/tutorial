
(dnns:cnns)=
# Convolutional approaches: CNNs, TCNs

The main idea behind CNNs is to convolve their input with learnable kernels. They are suitable to problems that have two characteristics {cite}`McFee2018`:  statistically meaningful information tends to concentrate locally (e.g. within a window around an event),
and shift-invariance (e.g. in time or frequency) can be used to reduce model complexity by reusing kernels' weights with multiple inputs.

CNNs can be designed to perform either 1-d or 2-d convolutions, or a combination of both. In the context of audio, in general 1-d convolutions are used in the temporal domain, 
whereas 2-d convolutions are usually applied to exploit time-frequency related information. We will focus on models that perform 2-d convolutions. The output of a convolutional layer is usually called _feature map_. 
In the context of audio applications, it is common to use CNN architectures combining convolutional and pooling layers. Pooling layers are used to down-sample feature maps between convolutional layers, so that deeper layers integrate larger extents of data. 
The most widely used pooling operator in the context of audio is _max-pooling_, which samples ---usually--- non-overlapping patches by keeping the biggest value in that region. 

**MAGDA to add figure here**


A convolutional layer is given by the following expression:

$$
\mathbf{Y}^j = f(\sum_{k=0} ^{K-1} \mathbf{W}^{kj}\: *\: \mathbf{X}^k + \mathbf{b}^j),
$$

where all $\mathbf{Y}^j$, $\mathbf{W}^{jk}$, and $\mathbf{X}^k$ are 2-d, $\mathbf{b}$ is the bias vector, $j$ indicates the j-th output channel, and $k$ indicates the k-th input channel. 
The input is a tensor $\mathbf{X} \in \mathbb{R}^{T\times F \times d}$, where $T$ and $F$ refer to the temporal and spatial---usually frequency---axes, and $d$ denotes a non-convolutional dimension or _channel_. 
In most audio applications $d$ usually equals one, though sometimes is used to encode multiple channels or multiple representations of the input (e.g. each channel is one representation). 
Note that while $\mathbf{Y}$ and $\mathbf{X}$ are 3-d arrays (with axes for height, width and channel), $\mathbf{W}$ is a 4-d array, so $\mathbf{W} \in \mathbb{R}^{h \times l 
\times d_{in} \times d_{out}}$, $h$ and $l$ being the dimensions of the convolution, and the 3rd and 4th dimensions account for the relation between input and output channels.

**Magda to add instantiation, eg Simons work)