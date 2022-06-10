# polysemantic-neurons

### Aims
Develop a proof-of-concept for an approach to penalising neural networks for learning polysemantic neurons (PNs). There were four aims:

1) pursue a programmatic approach to identifying PNs
2) explore loss terms to penalise a network for learning PNs
3) test whether changing the sign of the regularisation term encourages PNs and 
4) measure the clusterability of the trained models.


### Results

1) PNs were identified as those that fire for inputs of different classes.

2) A loss term was implemented as:

    $$polysemantic\ loss = \frac{\alpha}{n} \sum_{k=1}^l\sum_{i=1}^n\sum_{j=1,j\neq i}^n \beta_{ij} \hat x_{i}^{l} \cdot \hat x_{j}^{lT}$$

$\hat x_{i}^{l}$ is the activation vector of example i at hidden layer l rescaled to [0, 1]; n is the number of choose two pairs in each training batch;  controls the contribution of this term to the training loss. $\beta_{ij}$ is 0 if the labels of example i and j are equal and 1 otherwise.

The polysemantic loss was added to the cross entropy loss.  Control models were trained with $\alpha=0$; PNs were penalised with $\alpha=1$.  Compared to controls, when $\alpha=1$, the models were ~0.1% less accurate and neurons that fired for one class fired less frequently for other classes.

<p float="left">
    <img src="https://github.com/dr-darryl-wright/polysemantic-neurons/blob/main/experiments/mnist/alpha_0.0/trial_1/layer_analysis/activations/test/e1_activations.png" alt="drawing" width="400"/>
    <img src="https://github.com/dr-darryl-wright/polysemantic-neurons/blob/main/experiments/mnist/alpha_1.0/trial_1/layer_analysis/activations/test/e1_activations.png" alt="drawing" width="400"/>
</p>
*Fig 1. (left) Frequency each neuron activated across the test for each class with $\alpha=0.0$. (right) The same but for $\alpha=1.0$.*

3) Models were trained with $\alpha=-1$ to encourage PNs.  Compared to controls  the models were ~17% les accurate and neurons that fired for one class fired for other classes more frequently than in control models.

4) Penalising PNs lead to absolutely more clusterable networks than controls.  Encouraging PNs lead to absolutely less clusterable networks than controls, although the difference was less pronounced.

### Takeaways
These results  indicate that it is possible to identify PNs and too successfully penalise a network for learning them with little loss in classification performance.  Similar results hold for the same experimental set up with [Fashion-MNIST](https://github.com/dr-darryl-wright/polysemantic-neurons/tree/main/experiments/fashion_mnist).

For a network to be absolutely more clusterable it must be more modular, meaning it can be partitioned into subgraphs with strong internal connectivity but only weakly connected to other neurons[^1]. This shares the definition of circuits, “a computational subgraph of a neural network. It consists of a set of features, and the weighted edges that go between them in the original network.”[^2] The circuits agenda aims to improve interpretability of neural networks by identifying subgraphs that are more tractable to rigorously investigate than the whole network. If circuits are coherent with respect to class labels, then network behaviour could be interpreted in terms of individual circuit behaviour. Effectively allowing the interpretability of networks to be decomposed into subproblems. PNs are a problem as they are not coherent with respect to class labels and therefore, neither are circuits that contain them.

### Future directions
*Improving the polysemantic loss function*

For the polysemantic loss in the experiments, the activation vectors are rescaled because the pattern of activations rather than the strength of the activations should be penalised. However, this approach does not just contain information about which neuron fired for which example, but still contains information about the relative strength of the activation.  This results in a “soft” penalisation of PNs since the loss can be reduced by reducing the relative strength of the activation for examples of different classes.  A “hard” penalisation would instead binarise the activation vectors (fired or did not) rather than rescale them.  The loss can only then be reduced by ensuring that a neuron that activated for an example of one class does not activate for an example of another.  This would better target the definition of PNs used here.

*Are the clusters coherent with class labels?*

The clustering results in 4) are promising. The hypothesis was that penalising PNs should lead to networks that are absolutely more clusterable than controls and networks where PNs are encourages. This is what was seen for both MNIST and Fashion-MNIST.  MNIST networks were also relatively more clusterable.  However, networks trained on Fashion-MNIST were not, suggesting that the increased absolute clusterability may simply be due to the distribution of weights in each layer.  It remains to be seen if the circuits identified by clustering are coherent with respect to class labels or with respect to input features as has been observed for the MNIST controls[^3]. 

*Encouraging PNs*

One motivation for aim 3) was to investigate whether encouraging PNs would compress more information into each neuron and allow a smaller network to classify as accurately as the smallest network that achieves the same performance but trained only with cross entropy.  Conversely, penalising PNs might be expected to require a larger network to achieve the same performance as the controls.

When encouraging PNs, all hidden neurons fired for all inputs and the networks only achieve ~82% classification accuracy. This might indicate that $\alpha=-1$ is too aggressive, “washing out” the cross entropy contribution to the loss.

These aspects of encouraging PNs have not yet been explored.

*Scaling up*

I think these initial results are promising enough to motivate effort to scale up the experiments. The first step would be to replicate the local specialization study of [Casper et al. (2022)](https://openreview.net/pdf?id=HreeeJvkue9) with CNNs penalised for learning PNs.  The hope would be that training in this way will yield greater class-specific measures of coherence than has been observed so far. This would then justify a deeper dive into an analysis of the circuits identified in these networks.
At this stage we would have a pipeline for experiments to iterate on improvements for computer vision.  Expanding the pipeline to other networks such as language models would be an obvious next step.

[^1]: https://arxiv.org/pdf/2103.03386.pdf
[^2]: https://distill.pub/2020/circuits/zoom-in/
[^3]: https://openreview.net/pdf?id=HreeeJvkue9