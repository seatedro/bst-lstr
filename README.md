# bst-lstr
Long Short Term Transformer Recommendation System

### Dataset:
[MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/)

### Work Done:

- Implemented Behavior Sequence Transformer in PyTorch
- Combine long and short sequence lengths and their respective positional embeddings and pass to the encoder.
- Predict the `target_rating` using Dense layers

### Work to be done:

[ ] Add padding for the sequences to cover more movies in the train dataset
[ ] Test when overfitting occurs
[ ] Run inference on a deployed model
