# Dynamic Memory Networks Plus

In this work, I have implemented the Dynamic Memory Network Plus as presented in ["Dynamic Memory Networks for Visual and Textual Question Answering"](https://arxiv.org/abs/1603.01417) by Xiong et al. Here, we made use of the [bAbI dataset](https://research.fb.com/downloads/babi/) for training and validating the model.

Scripts:

1. Config.py: This script contains all the configurations/parameter values to be used

2. babi_input_processor.py: Contains the code for preprocessing of bAbI Dataset

3. custom_attention_cell.py: Contains implementation of the Attention GRU Cell as specified in DMN+

4. dynamicMemoryNetworkModel.py: Implementation of the complete Dynamic Memory Network Plus Model

5. training_dmn.py: Training the DMN on a specific task of the bAbI dataset

6. training_togther.py: Training on all the bAbI tasks
