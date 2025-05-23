<h1>Setting up the Virtual Environment</h1>
<p1>Simply run the command: pip install -r requirements.txt</p1>

<h2>Running the models</h2>
<p2>To train and test a model simply run the file the model is contained in. If you wish to change any hyper parameters, you can alter the batch size when the train and test loaders are initialized, the learning rate in the function train_model when the optimizer is defined, and the number of epochs is a parameter of the train_model </p2>


<h2>Helper Functions</h2>

<h4>def custom_collate_fn(batch)</h4>
<p4>
  Made a custom collation file because DeepLake has a default collation method but that tries to stack tensors from each batch.
But this doesn't work well in our case as the labels vary in length. So, this function, instead of stacking the tensors,
gathers the values for each key into a list. This prevents the run time errors that were occurring from attempting to 
stack the tensors.
</p4>

<h4>def convert_labels_to_multihot(raw_labels, num_classes=15)</h4>

<p4>Each image is associated with a variable number of labels, but the code required consistency, so this function is used to
convert these variable length label lists into a fixed length multi-hot vectors. In a multi-hot vector each position 
corresponds to a particular class and a value of 1 indicates that the class is present

</p4>

