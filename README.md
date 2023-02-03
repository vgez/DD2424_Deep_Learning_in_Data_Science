# DD2424_Deep_Learning_in_Data_Science - Lab Course

Lab part of the course **DD2424 Deep Learning in Data Science** at KTH Royal Institute of Technology. The labs focused on the theoretical knowledge and practical experience of training networks for deep learning including optimisation using stochastic gradient descent, analysis of models and representations and application examples of deep learning for learning of representations and recognition, among other subjects.


## Technologies

-   [Python3](https://www.python.org/)
-   [NumPy](https://numpy.org/)

## Assignment Details

### Assignment 1

The lab focused on to what extent we can train a one layered network to be able to classify pictures. The data-set came in batches of 10000 data-points with 3072 features each, divided into 10 unique classes. A softmax classification using a mini-batch gradient descent model meant that for each data-point, the model makes a prediction stored in a probability vector and argmax of said vector is predicted to be the class of the data-point. For each mini-batch, the weights and biases is updated. For further reading, see the <a href="https://github.com/vgez/DD2424_Deep_Learning_in_Data_Science/blob/main/labs/assign1/reports/DD2424_DeepLearning_Assign1_ValdemarGezelius_vgez.pdf">report</a>.

### Assignment 2

Adding to the code and theory of Assignment 1, a coarse-to-fine search for optimal hyperparameter values was conducted and a cyclic eta update was implemented. For further reading, see the <a href="https://github.com/vgez/DD2424_Deep_Learning_in_Data_Science/blob/main/labs/assign2/reports/DD2424_DeepLearning_Assign2_ValdemarGezelius_vgez.pdf">report</a>.

### Assignment 3

Apart from the code I implemented for Lab 2, Batch Normalization functionality was added to the network. The task was accomplished using Python3 with NumPy for mathematical computations and Matplotlib for data visualization. For further reading see the <a href="https://github.com/vgez/DD2424_Deep_Learning_in_Data_Science/blob/main/labs/assign3/reports/DD2424_DeepLearning_Assign3_ValdemarGezelius_vgez.pdf">report</a>.

### Assignment 4

The lab focused on training a Recurrent Neural Networks (RNN) to synthesize English text character by character. The assignment was implemented using Python 3, with the support of NumPy for mathematical computations and Matplotlib for creating graphs. For further reading, see the <a href="https://github.com/vgez/DD2424_Deep_Learning_in_Data_Science/blob/main/labs/assign4/reports/DD2424_DeepLearning_Assign4_ValdemarGezelius_vgez.pdf">report</a>. 

