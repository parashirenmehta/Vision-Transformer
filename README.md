# Vision-Transformer
This code contains my implementation of a vision transformer on MNIST dataset

The vit_call_main file contains the actual code that needs to be executed for the Vision transformer to run. 
It contains the train and test loaders, the model definition, the number of epochs, etc. needed for the program to run.

The ViT.py file contains the model definition for the Vision Transformer.
This file contains the linear projection, transformer encoder, MLP head, etc.

The vit_encoder.py file contains the code for the attention head class, multi-head class, and the congregation of both these components inside the transformer encoder, 
with the MLP inside the encoder.

The vit_blocks.py file specifies the number of encoders stacked on top of each other in the transformer.
