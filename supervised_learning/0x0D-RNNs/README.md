# 0x0D-RNNs

> **Recurrent Neural Network(RNN)** is a type of [Neural Network](https://www.geeksforgeeks.org/tag/neural-network/) where the **output from the previous step are fed as input to the current step**. The main and most important feature of RNN is **Hidden state**, which remembers some information about a sequence.
> RNN have a **“memory”** which remembers all information about what has been calculated. It uses the same parameters for each input as it performs the same task on all the inputs or hidden layers to produce the output. This reduces the complexity of parameters, unlike other neural networks.

- a general (FrameWork) approach to sequence moding problems.
- well suited for sequence of input
- instead of the general 1 to 1, vanilla NN. A RNN can be Many to One for sentiment classification. ej. sequence of word and outputs a sentiment representing the sentence.
- RNN can also be many to many, which returns many outputs at each time step of the sequence like music generation.
![RNN](https://www.simplilearn.com/ice9/free_resources_article_thumb/Network_framework.gif)

- standard vanilla is input to output which can't maintain sequential info
- RNN's has loops in them which allows information to persist. RNN use recurrence relation at every time step to process a sequence.
![[Screen Shot 2023-01-08 at 7.44.18 PM.png]]

- updates hidden state, and have a chain like structure
- Traiing RNN -> Backpropagation through time
  - errors are backpropragated at each individual time step and in all time step.

## How RNN works

- RNN converts the independent activations into dependent activations by providing the same weights and biases to all the layers, thus reducing the complexity of increasing parameters and memorizing each previous output by giving each output as input to the next hidden layer.
- Hence these three layers can be joined together such that the weights and bias of all the hidden layers are the same, in a single recurrent layer.

### Calculating current state

![Calculating current state](https://media.geeksforgeeks.org/wp-content/uploads/Screen-Shot-2018-08-23-at-3.27.11-PM.png)

ht -> current state
ht-1 -> previous state
xt -> input state

**Formula for applying Activation function(tanh):**

![activatioj formulae](https://media.geeksforgeeks.org/wp-content/uploads/Screen-Shot-2018-08-23-at-3.28.27-PM.png)

whh -> weight at recurrent neuron
wxh -> weight at input neuron

**The formula for calculating output:**

![output formula](https://media.geeksforgeeks.org/wp-content/uploads/Screen-Shot-2018-08-23-at-3.29.30-PM.png)

Yt -> output
Why -> weight at output layer

### Training through RNN

1. A single-time step of the input is provided to the network.
2. Then calculate its current state using a set of current input and the previous state.
3. The current ht becomes ht-1 for the next time step.
4. One can go as many time steps according to the problem and join the information from all the previous states.
5. Once all the time steps are completed the final current state is used to calculate the output.
6. The output is then compared to the actual output i.e the target output and the error is generated.
7. The error is then back-propagated to the network to update the weights and hence the network (RNN) is trained.

### **Applications of Recurrent Neural Network**

1. Language Modelling and Generating Text
2. Speech Recognition
3. Machine Translation
4. Image Recognition, Face detection
5. Time series Forecasting
