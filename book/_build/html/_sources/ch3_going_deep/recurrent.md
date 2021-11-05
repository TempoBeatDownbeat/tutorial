(dnns)=
# Recurrent approaches: RNNs, LSTMs, and GRUs


The goal of this section is to provide the theoretical underpinnings of DNN models commonly used for tempo, beat and downbeat tracking, to build an understanding of why these models work well in these tasks and what are their limitations. Here we discuss the main concepts and definitions,
but for those interested in reading more about the use of DNNs in audio processing and MIR related tasks,  we suggest them to take a look at works like McFee {cite}`McFee2018`, Purwins et al. {cite}`purwins2019deep` and Choi et al. {cite}`choi2017tutorial`.

**Sebastian do you have further suggestions for resources about DNNs for audio/MIR here**?

(dnns:context)=
### A bit of context


Motivated by the huge success of deep neural networks (DNNs) in Computer Vision, and due to recent advances that allow for faster training and scalability, DNNs have been widely used in many other domains, in particular in audio related tasks {cite}`goodfellow2016deep`. 
The inclusion of these models in MIR tasks has meant a considerable improvement in the performance of automatic systems, in particular tempo, beat and downbeat tracking ones, as can be seen from the MIREX campaigns {cite}`jia2019deep`. 
Moreover, the use of deep learning models presents other advantages over traditional machine learning methods used in MIR, i.e. they are flexible and adaptable across tasks. As an example, convolutional neural network based models from Computer Vision were adapted for 
onset detection {cite}`schluter2014improved`, and then for  segment boundary detection {cite}`ullrich2014boundary`. Furthermore, DNNs reduce ---or allow to remove completely--- the stage of hand-crafted feature design, by including the feature learning as part of the learning
process. 

However, the use of supervised deep learning models presents some disadvantages, one of the main ones being their dependence on annotated data. Annotated data is an important bottleneck in MIR especially due to copyright issues, and because annotating a musical piece requires
expert knowledge and is thus expensive. Besides, solutions obtained in a data-driven fashion suffer from bias depending on the dataset used, a problem that also occurs in other learning-based approaches. Besides, deep-learning based methods are usually less interpretable than
signal processing methods, making it a bit hard to predict the type of mistakes a DNN would do when presented with e.g. unseen music tracks or genres. 



In general terms, a deep neural network consists of a composition of non-linear functions that acts as a function approximator $F_\omega: \mathbf{X} \rightarrow \mathbf{Y}$, for given input and output data $\mathbf{X}$ and $\mathbf{Y}$. 
The network is parametrized by its weights $\omega$, whose values are optimized so the estimated output $\hat{\mathbf{Y}}=F_\omega (\mathbf{X})$ approximates the desired output $\mathbf{Y}$ given an input $\mathbf{X}$.

(dnns:mlps)=
### Multi-layer perceptrons


Multi-layer perceptrons (MLPs) are the simple and basic modules of DNNs. They are also known as _fully-connected layers_ or _dense layers_, and consist of a sequence of layers, each defined by an affine transformation composed with a non-linearity: 

$$
\mathbf{y} = f(\mathbf{W}^T \mathbf{x} + \mathbf{b}),
$$

where $\mathbf{x} \in \mathbb{R}^{d_{in}}$ is the input, $\mathbf{y} \in \mathbb{R}^{d_{out}}$ is the output, $\mathbf{b} \in \mathbb{R}^{d_{out}}$ is called the _bias vector_ and $\mathbf{W} \in \mathbb{R}^{d_{in} \times d_{out}}$ is the weight matrix. $f()$ is a non-linear activation function, which allows the model to learn non-linear 
representations. Note that for multi-dimensional inputs, e.g. $\mathbf{X} \in \mathbb{R}^{d_1 \times d_2}$, the  input is flattened so $\mathbf{x} \in \mathbb{R}^{d}$ with $d = d_1 \times d_2$. These layers are usually used to map the input to another space where hopefully the problem (e.g. classification or regression) can be solved more easily. However, by definition, this type of layer is not shift or scale invariant, 
meaning that when using this type of network for audio tasks, any small temporal or frequency shift needs dedicated parameters to be modelled, becoming very expensive and inconvenient when it comes to modelling music.  

MLPs have been mainly used in early works before convolutional neural networks (CNNs) and recurrent neural networks (RNNs) became popular {cite}`choi2017tutorial`, and are now used in combination with those architectures, usually as the last layers of a model to map high dimensional intermediate representations
to the output space (e.g. classes), as discussed below.
 
(dnns:rnns)=
### Recurrent networks

Unlike CNNs which are effective at modelling fixed-length local interactions, _recurrent neural networks_ (RNNs) are good in modelling variable-length long-term interactions. RNNs exploit recurrent connections since they are formulated as {cite}`goodfellow2016deep`:

$$
\begin{align}
       \mathbf{y}_t = f_{y}(\mathbf{W}_y\:\mathbf{h}_t + \mathbf{b}_y), \label{eq:rnn_def_1}\\
        \mathbf{h}_t = f_{h}(\mathbf{W}_h\:\mathbf{x}_t + \mathbf{U}\: \mathbf{h}_{t-1} + \mathbf{b}_h),\label{eq:rnn_def_2}
\end{align}
$$


where $\mathbf{h}_t$ is a hidden _state vector_ that stores information at time $t$, $f_y$ and $f_h$ are the non-linearities of the output and hidden state respectively, and $\mathbf{W}_y, \mathbf{W}_h$ and $\mathbf{U}$ are matrices of trainable weights. 
An RNN integrates information over time up to time step $t$ to estimate the state vector $\mathbf{h}_t$, being suitable to model sequential data. 
Note that learning the weights $\mathbf{W}_y, \mathbf{W}_h$ and $\mathbf{U}$ in a RNN is challenging given the dependency of the gradient on the entire state sequence {cite}`McFee2018`. In practice, _back-propagation through time_ is used 
{cite}`werbos1990backpropagation`, which consists in unrolling Equation \ref{eq:rnn_def_2} up to $k$ time steps and applying standard back propagation. Given the accumulative effect of applying $\mathbf{U}$ when unrolling Equation  \ref{eq:rnn_def_2}, the gradient values tend to either vanish or explode if $k$ is too big, a problem known as the \textit{vanishing and exploding gradient problem}. For that reason, in practice the value of $k$ is limited to account for relatively short sequences.

The most commonly used variations of RNNs, that were designed to mitigate the vanishing/exploding problem of the gradient, include the addition of \textit{gates} that control the flow of information through the network. The most popular ones in MIR applications are \textit{long-short memory units} (LSTMs) \cite{hochreiter1997long} and \textit{gated recurrent units} (GRUs) \cite{Cho2014}. We will focus here on GRUs, which we use in our experiments in the following chapters, and mention LSTMs only to draw differences between the two neural networks.


\subsubsection{Gated recurrent units}

In a GRU layer, the \textit{gate} variables $\mathbf{r}_t$ and $\mathbf{u}_t$ ---named as \textit{reset} and \textit{update} vectors--- control the updates to the state vector $\mathbf{h}_t$, which is a combination of the previous state $\mathbf{h}_{t-1}$ and a proposed next state $\hat{\mathbf{h}}_t$. The equations that rule these updates are given by:


\begin{subequations}\label{eq:gru_def}
\begin{align}
        \mathbf{r}_t = f_g(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r), \label{eq:gru_def_1}\\
        \mathbf{u}_t = f_g(\mathbf{W}_u \mathbf{x}_t + \mathbf{U}_u \mathbf{h}_{t-1} + \mathbf{b}_u),\label{eq:gru_def_2}\\
        \hat{\mathbf{h}}_t = f_h(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t\odot \mathbf{h}_{t-1}) + \mathbf{b}_h), \label{eq:gru_def_3}\\
         \mathbf{h}_t = \mathbf{u}_t\odot \mathbf{h}_{t-1} + (1-\mathbf{u}_t)\odot \hat{\mathbf{h}}_t;\label{eq:gru_def_4}
\end{align}
\end{subequations}

\noindent
$\odot$ indicates the element-wise Hadamard product, $f_g$ is the activation applied to the reset and update vectors, and $f_h$ is the output activation. $\mathbf{W}_r, \mathbf{W}_u, \mathbf{W}_h \in \mathbb{R}^{d_{i-1}\times d_i}$ are the input weights, $\mathbf{U}_r, \mathbf{U}_u, \mathbf{U}_h \in \mathbb{R}^{d_{i}\times d_i}$ are the recurrent weights and $\mathbf{b}_r, \mathbf{b}_u, \mathbf{b}_h \in \mathbb{R}^{d_{i}}$ are the biases. The activation functions $f_g$ and $f_h$ are typically sigmoid and tanh, since saturating functions help to avoid exploding gradients in recurrent networks.
 
The GRU operates as follows: when $\mathbf{u}_t$ is close to 1, the previous observation $\mathbf{h}_{t-1}$ dominates in Equation \ref{eq:gru_def_4}. When $\mathbf{u}_t$ gets close to 0, depending on the value of $\mathbf{r}_t$, either a new state is updated with the standard recurrent equation by $\hat{\mathbf{h}}_t = f(\mathbf{W}_h \mathbf{x}_t + \mathbf{v}_h \mathbf{h}_{t-1} + \mathbf{b}_h)$, if $\mathbf{r}_t=1$, or the state is ``reset'' as if the $\mathbf{x}_t$ was the first observation in the sequence by $\hat{\mathbf{h}}_t = f(\mathbf{W}_h \mathbf{x}_t + \mathbf{b}_h)$.\\
The reset variables allow GRUs to successfully model long-term interactions, and perform comparably to LSTMs, but GRUs are simpler since LSTMs have three gate vectors and one extra \textit{memory} gate. Empirical studies show that both networks perform comparably  while GRUs are faster to train \cite{greff2016lstm, jozefowicz2015empirical}, so in this dissertation we will use GRUs for the study of long-term dependencies over time.

\subsubsection{Bi-directional models}

GRUs and RNNs in general are designed to integrate information in one direction, e.g. in an audio application they integrate information forward in time. However, it can be beneficial to integrate information in both directions, and so has been the case for neural networks in audio applications such as beat tracking \cite{bock2011enhanced} or environmental sound detection \cite{parascandolo2016recurrent}. A bi-directional recurrent neural network (Bi-RNN) \cite{schuster1997bidirectional} in the context of audio consists of two RNNs running in opposite time directions with their hidden vectors $\mathbf{h}_t ^f$ and $\mathbf{h}_t ^b$ being concatenated, so the output $h_t$ at time $t$ has information about the entire sequence. Unless the application is online, Bi-RNNs are usually preferred due to better performance \cite{McFee2018}.

\subsection{Hybrid architectures}
\label{ssec:hybrid_dnns}

As mentioned before, MLPs are now usually being used in combination with CNNs, which are able to overcome the lack of shift and scale invariance MLPs suffer. At the same time, MLPs offer a simple alternative for mapping representations from a big-dimensional space to a smaller one, suitable for classification problems. 

Finally, hybrid architectures that integrate convolutional and recurrent networks have recently become popular and have proven to be effective in audio applications, especially in MIR \cite{sigtia2016end, mcfee2017_structured}. They integrate local feature learning with global feature integration, a playground between time scales that is in particular interesting in the scope of this dissertation. In the following chapters, and considering what was exposed in this section, we explore hybrid architectures for the task of rhythm analysis.

\subsection{Learning and optimization}

To optimize the parameters $\omega$, a variant of gradient descent is usually exploited. A \textit{loss function} $J(\omega)$ measures the difference between the predicted and desired outputs $\hat{\mathbf{Y}}$ and $\mathbf{Y}$, so the main idea behind the optimization process is to iteratively update the weights $\omega$ so the loss function decreases, that is:

\begin{equation}\label{eq:gradient_descent}
\omega \leftarrow \omega - \eta \nabla_\omega J(\omega)
\end{equation}

\noindent
Where $\eta$ is the \textit{learning rate} which controls how much to update the values of $\omega$ at each iteration. Because the DNN consists of a composition of functions, the gradient of $J(\omega)$, $\nabla \omega J(\omega)$, is obtained via the chain rule, a process known as \textit{back propagation}. In the last four years, many software packages that implement automatic differentiation tools and various versions of gradient descent were released, \cite{chollet2015keras, tensorflow2015, theano2016}, reducing considerably the time needed for the implementation of such models. 

Since computing the gradient over a large training set is very expensive both in memory and computational complexity, a widely adopted variant of gradient descent is \textit{Stochastic Gradient Descent} (SGD) \cite{bottou1991stochastic}, which approximates the gradient at each step on a mini-batch of training samples, $B$, considerably smaller than the training set. There are other variants of SGD such as \textit{momentum} methods or \textit{adaptive update} schemes that accelerate convergence dramatically, by re-using information of previous gradients (momentum) and reducing the dependence on $\eta$. From 2017, the most popular method for optimizing DNNs has been the adaptive method ADAM \cite{Adam}, which is the one we use in our work. 

Another common practice in the optimization of DNNs is to use \textit{early stopping} as regularization \cite{sjoberg1995overtraining}, which means to stop training if the training --or validation-- loss is not improving after a certain amount of iterations. Finally, \textit{batch normalization} (BN) \cite{batchnorm} is widely used in practice as well, and consists of scaling the data by estimating its statistics during training, which usually leads to better performance and faster convergence. 



\subsection{Activation functions}

The expressive power of DNNs is in great extent due to the use of non-linearities $f()$ in the model. The type of non-linearity used depends on whether it is an internal layer or the output layer. Many different options have been explored in the literature for intermediate-layer non-linearities ---usually named \textit{transfer functions}, the two main groups being saturating or non-saturating functions (e.g. \textit{tanh} or \textit{sigmoid} for saturated, because they saturate in 0 and 1, and \textit{rectified linear units} (ReLUs) \cite{nair2010rectified} for non-saturating ones). Usually non-saturating activations are preferred in practice for being simpler to train and increasing training speed \cite{McFee2018}.