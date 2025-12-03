# Why and How Does Post-Norm cause instability in the values? Explain Mathematically Where was this insight found and can you find the paper for me to cite it?

Sublayer here can be the :
Positionwise Fully Feedforward Neural Network (FNN)
Multi-Head Self Attention (MHSA)
Mask Multi-Head Self Attention
Multi-Head Cross Attention (MHCA) 


My thought process:
Post-Norm (The Problem)
input_of_sublayer = sublayer(input) + input 
output_of_sublayer = LayerNorm(input_of_sublayer)

Pre-Norm (Actual)

input_to_sublayer = RMSNorm(input)
output_of_sublayer = sublayer(input_to_sublayer) + input_to_sublayer 


Lets say your original input values are very huge,
within the sublayer, there is a chance that it cause the number to overflow to a number which the bits can support because, when we use softmax in the MHSA, it use an exponential function

Lets say your original input values are very small,
within the sublayer, there is a chance that it cause the number to underflow to a number which the bits can support because, when we use softmax in the MHSA, it use an exponential function for the softmax before normalizing,

These both results in overflow.
As we stack more and more of these layers, this is an issue, cause instability

While in Pre-Norm.
You immediately normalize the values, to a specific range in the beginning.

