# Baseline_CNN_Model
## def custom_collate_fn(batch):

Made a custom collation file because DeepLake has a default collation method but that tries to stack tensors from each batch.
But this doesn't work well in our case as the labels vary in length. So, this function, instead of stacking the tensors,
gathers the values for each key into a list. This prevents the run time errors that were occurring from attempting to 
stack the tensors.
---
## def convert_labels_to_multihot(raw_labels, num_classes=15):

Each image is associated with a variable number of labels, but the code required consistency, so this function is used to
convert these variable length label lists into a fixed length multi-hot vectors. In a multi-hot vector each position 
corresponds to a particular class and a value of 1 indicates that the class is present

