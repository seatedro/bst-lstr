import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm, trange
import pickle
import numpy as np
tqdm.pandas()


def create_dataset(df: pd.DataFrame, max_len: int, user_count: int, item_count: int, dataset: str = 'ml-1m'):
    user_group = df.groupby('uid')
    if dataset != 'steam' and dataset != 'beauty':
        user2items = user_group.progress_apply(lambda x: list(x.sort_values('timestamp')['sid']))
    else:
        user2items = user_group.progress_apply(lambda x: list(x['sid']))
    train, val, test = {}, {}, {}
    for user in range(user_count):
        items = user2items[user]
        train[user]= items
    
    user_arange = np.arange(0, user_count)
    user_arange.shape

    train_df = pd.DataFrame(user_arange, columns=['user'])
    train_df['seq'] = [[] for _ in range(user_count)]

    val_df = pd.DataFrame(user_arange, columns=['user'])
    # Add new column seq
    # Add new column seq with empty lists
    val_df['seq'] = [[] for _ in range(user_count)]

    test_df = pd.DataFrame(user_arange, columns=['user'])
    test_df['seq'] = [[] for _ in range(user_count)]

    train_df, val_df, test_df = create_sequences(train, train_df, val_df, test_df, user_count, max_len, max_len)

    # Print train_df number of rows
    print(f'Train_df shape: {train_df.shape[0]}')
    train_df = train_df[train_df['seq'].map(len) > 0]
    val_df = val_df[val_df['seq'].map(len) > 0]
    test_df = test_df[test_df['seq'].map(len) > 0]

    train_df = train_df[["user", "seq"]].explode("seq", ignore_index=True)
    val_df = val_df[["user", "seq"]].explode("seq", ignore_index=True)
    test_df = test_df[["user", "seq"]].explode("seq", ignore_index=True)

    return train_df, val_df, test_df


def create_sequences(train, train_df, val_df, test_df, user_count, window_size, step_size) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for i in trange(0, user_count):
        train_sequences = []
        val_sequences = []
        test_sequences = []
        values = train[i]
        start_index = 0
        while True:
            end_index = start_index + window_size
            seq = values[start_index:end_index]
            if len(seq) < window_size:
                # If sequence length is less than half of the window size, dont use it
                if len(seq) < window_size / 2:
                    break
                train_sequences.append(seq[:-2])
                val_sequences.append(seq[-2:-1])
                test_sequences.append(seq[-1:])
                break
            train_sequences.append(seq[:-2])
            val_sequences.append(seq[-2:-1])
            test_sequences.append(seq[-1:])
            start_index += step_size
        # Set the sequence to the correct row
        train_df.at[i, 'seq'] = train_sequences
        val_df.at[i, 'seq'] = val_sequences
        test_df.at[i, 'seq'] = test_sequences

    return train_df, val_df, test_df

def generate_negative_samples(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, item_count: int, seed: int, sample_size: int = 100) -> dict[str, list[int]]:
    np.random.seed(seed)
    negative_samples = {}
    # If sample size is 0 return;
    if sample_size == 0: return
    for row in trange(len(train_df)):
        seen = set(train_df.iloc[row]['seq'])
        seen.update(val_df.iloc[row]['seq'])
        seen.update(test_df.iloc[row]['seq'])
        
        samples = []
        for _ in range(sample_size):
            item = np.random.choice(item_count) + 1
            while item in seen or item in samples:
                item = np.random.choice(item_count) + 1
            samples.append(item)
        
        negative_samples[row] = samples
    
    return negative_samples


def beauty(max_len: int):
    # Load json file
    df = pd.read_csv('../data/beauty/Beauty.txt', sep=' ', header=None, names=['uid', 'sid'])
    num_ratings = len(df)
    # Get unique users and items
    user_count = df['uid'].nunique()
    item_count = df['sid'].nunique()
    umap = {u: i for i, u in enumerate(set(df['uid']))}
    smap = {s: i for i, s in enumerate(set(df['sid']))}
    df['uid'] = df['uid'].map(umap)
    df['sid'] = df['sid'].map(smap)
    max_ratings = user_count * item_count
    density = float(num_ratings) / float(max_ratings)

    # Save the density to a file
    with open('../data/beauty/density.txt', 'w') as f:
        f.write(str(density))

    if not os.path.exists(f"../data/lstr/beauty/seqlen-{max_len}"):
        os.makedirs(f"../data/lstr/beauty/seqlen-{max_len}")

    # Store the item count in a file
    with open(f'../data/lstr/beauty/seqlen-{max_len}/itemcount.pkl', 'wb') as f:
        pickle.dump(item_count, f)

    data_flag = os.path.exists(f'../data/lstr/beauty/seqlen-{max_len}/train.pkl') and os.path.exists(f'../data/lstr/beauty/seqlen-{max_len}/val.pkl') and os.path.exists(f'../data/lstr/beauty/seqlen-{max_len}/test.pkl')
    
    if not data_flag:
        train_df, val_df, test_df = create_dataset(df, max_len, user_count, item_count, 'beauty')
        train_df.to_pickle(f"../data/lstr/beauty/seqlen-{max_len}/train.pkl", protocol=4)
        val_df.to_pickle(f"../data/lstr/beauty/seqlen-{max_len}/val.pkl")
        test_df.to_pickle(f"../data/lstr/beauty/seqlen-{max_len}/test.pkl")
    
    # print(f'Train_df shape: {train_df.shape[0]}')

    
    
    

    dataset_folder = Path(f'../data/lstr/beauty/seqlen-{max_len}')

    train_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('train.pkl'))
    val_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('val.pkl'))
    test_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('test.pkl'))

    negative_test_save_file_path = 'negative.pkl'
    negative_test_save_file = dataset_folder.joinpath(negative_test_save_file_path)
    # If negative samples dont exist, set false
    negative_flag = os.path.exists(negative_test_save_file)

    if not negative_flag:
        negative_test_samples = generate_negative_samples(train_df, val_df, test_df, item_count, 42, max_len)
        with open(negative_test_save_file, 'wb') as f:
            pickle.dump(negative_test_samples, f)

    with open(negative_test_save_file, 'rb') as f:
        negative_test_samples = pickle.load(f)
    

    print(f"🚀🚀🚀 Preprocessing finished for beauty with sequence length: {max_len}")

    return item_count

def steam(max_len: int):
    # Load json file
    df = pd.read_csv('../data/steam/Steam.txt', sep=' ', header=None, names=['uid', 'sid'])
    num_ratings = len(df)
    # Get unique users and items
    user_count = df['uid'].nunique()
    item_count = df['sid'].nunique()
    umap = {u: i for i, u in enumerate(set(df['uid']))}
    smap = {s: i for i, s in enumerate(set(df['sid']))}
    df['uid'] = df['uid'].map(umap)
    df['sid'] = df['sid'].map(smap)
    max_ratings = user_count * item_count
    density = float(num_ratings) / float(max_ratings)

    # Save the density to a file
    with open('../data/steam/density.txt', 'w') as f:
        f.write(str(density))

    if not os.path.exists(f"../data/lstr/steam/seqlen-{max_len}"):
        os.makedirs(f"../data/lstr/steam/seqlen-{max_len}")

    # Store the item count in a file
    with open(f'../data/lstr/steam/seqlen-{max_len}/itemcount.pkl', 'wb') as f:
        pickle.dump(item_count, f)

    data_flag = os.path.exists(f'../data/lstr/steam/seqlen-{max_len}/train.pkl') and os.path.exists(f'../data/lstr/steam/seqlen-{max_len}/val.pkl') and os.path.exists(f'../data/lstr/steam/seqlen-{max_len}/test.pkl')
    
    if not data_flag:
        train_df, val_df, test_df = create_dataset(df, max_len, user_count, item_count, 'steam')
        train_df.to_pickle(f"../data/lstr/steam/seqlen-{max_len}/train.pkl", protocol=4)
        val_df.to_pickle(f"../data/lstr/steam/seqlen-{max_len}/val.pkl")
        test_df.to_pickle(f"../data/lstr/steam/seqlen-{max_len}/test.pkl")
    
    # print(f'Train_df shape: {train_df.shape[0]}')

    
    
    

    dataset_folder = Path(f'../data/lstr/steam/seqlen-{max_len}')

    train_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('train.pkl'))
    val_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('val.pkl'))
    test_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('test.pkl'))

    negative_test_save_file_path = 'negative.pkl'
    negative_test_save_file = dataset_folder.joinpath(negative_test_save_file_path)
    # If negative samples dont exist, set false
    negative_flag = os.path.exists(negative_test_save_file)

    if not negative_flag:
        negative_test_samples = generate_negative_samples(train_df, val_df, test_df, item_count, 42, max_len)
        with open(negative_test_save_file, 'wb') as f:
            pickle.dump(negative_test_samples, f)

    with open(negative_test_save_file, 'rb') as f:
        negative_test_samples = pickle.load(f)
    

    print(f"🚀🚀🚀 Preprocessing finished for steam with sequence length: {max_len}")

    return item_count

def ml_20m(max_len: int):
    df = pd.read_csv('../data/ml-20m/ratings.csv', header=0, names=['uid', 'sid', 'rating', 'timestamp'])
    df = df[df['rating'] >= 0]
    num_ratings = len(df)

    item_sizes = df.groupby('sid').size()
    good_items = item_sizes.index[item_sizes >= 0]
    df = df[df['sid'].isin(good_items)]

    user_sizes = df.groupby('uid').size()
    good_users = user_sizes.index[user_sizes >= 5]
    df = df[df['uid'].isin(good_users)]

    umap = {u: i for i, u in enumerate(set(df['uid']))}
    smap = {s: i for i, s in enumerate(set(df['sid']))}
    df['uid'] = df['uid'].map(umap)
    df['sid'] = df['sid'].map(smap)

    user_count = len(umap)
    item_count = len(smap)
    max_ratings = user_count * item_count
    density = float(num_ratings) / float(max_ratings)
    # Save the density to a file
    with open('../data/ml-20m/density.txt', 'w') as f:
        f.write(str(density))

    if not os.path.exists(f"../data/lstr/ml-20m/seqlen-{max_len}"):
        os.makedirs(f"../data/lstr/ml-20m/seqlen-{max_len}")

    # Store the item count in a file
    with open(f'../data/lstr/ml-20m/seqlen-{max_len}/itemcount.pkl', 'wb') as f:
        pickle.dump(item_count, f, protocol=4)


    # If train.pkl or val.pkl or test.pkl dont exist, set false
    data_flag = os.path.exists(f'../data/ml-20m/seqlen-{max_len}/train.pkl') and os.path.exists(f'../data/ml-1m/seqlen-{max_len}/val.pkl') and os.path.exists(f'../data/ml-1m/seqlen-{max_len}/test.pkl')
    
    if not data_flag:
        train_df, val_df, test_df = create_dataset(df, max_len, user_count, item_count)
        train_df.to_pickle(f"../data/lstr/ml-20m/seqlen-{max_len}/train.pkl", protocol=4)
        val_df.to_pickle(f"../data/lstr/ml-20m/seqlen-{max_len}/val.pkl", protocol=4)
        test_df.to_pickle(f"../data/lstr/ml-20m/seqlen-{max_len}/test.pkl", protocol=4)
    
    # print(f'Train_df shape: {train_df.shape[0]}')

    
    
    

    dataset_folder = Path(f'../data/lstr/ml-20m/seqlen-{max_len}')

    train_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('train.pkl'))
    val_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('val.pkl'))
    test_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('test.pkl'))

    negative_test_save_file_path = 'negative.pkl'
    negative_test_save_file = dataset_folder.joinpath(negative_test_save_file_path)
    # If negative samples dont exist, set false
    negative_flag = os.path.exists(negative_test_save_file)

    if not negative_flag:
        negative_test_samples = generate_negative_samples(train_df, val_df, test_df, item_count, 42, max_len)
        with open(negative_test_save_file, 'wb') as f:
            pickle.dump(negative_test_samples, f, protocol=4)

    with open(negative_test_save_file, 'rb') as f:
        negative_test_samples = pickle.load(f)
    

    print(f"🚀🚀🚀 Preprocessing finished for ml-20m with sequence length: {max_len}")

    return item_count

def ml_1m(max_len: int):
    df = pd.read_csv("../data/ml-1m/ratings.dat", sep="::", names=["uid", "sid", "rating", "timestamp"])
    df = df[df['rating'] >= 0]
    num_ratings = len(df)

    item_sizes = df.groupby('sid').size()
    good_items = item_sizes.index[item_sizes >= 0]
    df = df[df['sid'].isin(good_items)]

    user_sizes = df.groupby('uid').size()
    good_users = user_sizes.index[user_sizes >= 5]
    df = df[df['uid'].isin(good_users)]

    umap = {u: i for i, u in enumerate(set(df['uid']))}
    smap = {s: i for i, s in enumerate(set(df['sid']))}
    df['uid'] = df['uid'].map(umap)
    df['sid'] = df['sid'].map(smap)

    user_count = len(umap)
    item_count = len(smap)
    max_ratings = user_count * item_count
    density = float(num_ratings) / float(max_ratings)
    # Save the density to a file
    with open('../data/ml-1m/density.txt', 'w') as f:
        f.write(str(density))

    if not os.path.exists(f"../data/lstr/ml-1m/seqlen-{max_len}"):
        os.makedirs(f"../data/lstr/ml-1m/seqlen-{max_len}")

    # Store the item count in a file
    with open(f'../data/lstr/ml-1m/seqlen-{max_len}/itemcount.pkl', 'wb') as f:
        pickle.dump(item_count, f)


    # If train.pkl or val.pkl or test.pkl dont exist, set false
    data_flag = os.path.exists(f'../data/ml-1m/seqlen-{max_len}/train.pkl') and os.path.exists(f'../data/ml-1m/seqlen-{max_len}/val.pkl') and os.path.exists(f'../data/ml-1m/seqlen-{max_len}/test.pkl')
    
    if not data_flag:
        train_df, val_df, test_df = create_dataset(df, max_len, user_count, item_count)
        train_df.to_pickle(f"../data/lstr/ml-1m/seqlen-{max_len}/train.pkl")
        val_df.to_pickle(f"../data/lstr/ml-1m/seqlen-{max_len}/val.pkl")
        test_df.to_pickle(f"../data/lstr/ml-1m/seqlen-{max_len}/test.pkl")
    
    print(f'Train_df shape: {train_df.shape[0]}')

    
    
    

    dataset_folder = Path(f'../data/lstr/ml-1m/seqlen-{max_len}')

    train_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('train.pkl'))
    val_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('val.pkl'))
    test_df: pd.DataFrame = pd.read_pickle(dataset_folder.joinpath('test.pkl'))

    negative_test_save_file_path = 'negative.pkl'
    negative_test_save_file = dataset_folder.joinpath(negative_test_save_file_path)
    # If negative samples dont exist, set false
    negative_flag = os.path.exists(negative_test_save_file)

    if not negative_flag:
        negative_test_samples = generate_negative_samples(train_df, val_df, test_df, item_count, 42, max_len)
        with open(negative_test_save_file, 'wb') as f:
            pickle.dump(negative_test_samples, f)

    with open(negative_test_save_file, 'rb') as f:
        negative_test_samples = pickle.load(f)
    

    print(f"🚀🚀🚀 Preprocessing finished for ml-1m with sequence length: {max_len}")

    return item_count