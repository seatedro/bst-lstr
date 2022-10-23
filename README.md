# bst-lstr
Long Short Term Transformer Recommendation System

### Dataset:
[MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/)

### Folder Structure:

 ```
  bst-lstr/
  │
  ├── data/ - directory for storing train and test data
  ├── util/ - directory for additional colab notebooks
  └── saved/
      ├── models/ - trained models are saved here
      └── log/ - default logdir for tensorboard and logging output
  ```

### Work Done:

- Implemented Behavior Sequence Transformer in PyTorch
- Combine long and short sequence lengths and their respective positional embeddings and pass to the encoder.
- Predict the `target_rating` using Dense layers

### Work to be done:

- [x] Add padding for the sequences to cover more movies in the train dataset
- [ ] Test when overfitting occurs
- [ ] Adding support for GRU model for matching
- [ ] Run inference on a deployed model
