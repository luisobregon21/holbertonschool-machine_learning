# Attention

## What is Attention mechanism

In machine learning, the attention mechanism is a way for a model to focus on certain parts of its input when processing it. This can be useful in tasks such as machine translation, where the model needs to pay attention to certain words in the input sentence in order to properly translate them. The attention mechanism allows the model to weigh the importance of different parts of the input, rather than processing the entire input equally. This can make the model more efficient and improve its performance.

![[Pasted image 20230127214106.png]]

## Applying Attention to RNNS

In Recurrent Neural Networks (RNNs), the attention mechanism can be applied in several ways. One common method is to use **an attention layer that takes the hidden state of the RNN at each time step, along with the current input, and produces a weighting over the input sequence.** These weights represent the attention that the model is giving to each part of the input at that time step.

Another way to apply attention to RNNs is to use an **attention-based RNN cell**, such as the Attention LSTM (ALSTM) or Attention GRU (AGRU), which have an attention mechanism built directly into the cell. These cells have an extra attention layer that takes the hidden state and the current input, and produces a weighted context vector that is then combined with the hidden state before it is passed to the next time step. In both cases, the **attention mechanism allows the RNN to focus on the most relevant parts of the input**, which can help improve the performance of the model.

### additional resources

- <https://www.youtube.com/watch?v=B3uws4cLcFw>

- <https://youtube.com/watch?v=SysgYptB198>

- <https://www.youtube.com/watch?v=quoGRI-1l0A>
