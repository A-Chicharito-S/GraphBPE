from sklearn.model_selection import train_test_split


def split_dataset(idx_array, dataset_name, dataset):
    len_dataset = len(idx_array)
    if len_dataset <= 2000:
        val_test_size = int(len_dataset * 0.2)  # 0.6/0.2/0.2
        val_test_percentage = 0.2
    else:
        val_test_size = int(len_dataset * 0.1)  # 0.8/0.1/0.1
        val_test_percentage = 0.1

    if dataset_name in ['esol', 'freesolv', 'lipophilicity', 'qm9']:  # regression
        train_index, val_test = train_test_split(idx_array, test_size=2*val_test_percentage, random_state=0)
        val_index, test_index = train_test_split(val_test, test_size=0.5, random_state=0)
    else:  # classification
        # mutag: {1: 125, 0: 63}
        # enzymes: {5: 100, 4: 100, 0: 100, 1: 100, 2: 100, 3: 100}
        # proteins: {0: 663, 1: 450}
        if dataset_name == 'mutag' or dataset_name == 'proteins':  # binary dataset
            pos_idx, neg_idx = [], []
            for idx, data in enumerate(dataset):
                y = data.y.item()
                if y == 1:
                    pos_idx.append(idx)
                else:
                    neg_idx.append(idx)
            # add positive data
            val_index = [pos_idx.pop(0) for _ in range(int(val_test_size / 2))]  # half positive
            test_index = [pos_idx.pop(0) for _ in range(int(val_test_size / 2))]
            train_index = pos_idx
            # add negative data
            val_index = val_index + [neg_idx.pop(0) for _ in range(val_test_size - len(val_index))]  # half negative
            test_index = test_index + [neg_idx.pop(0) for _ in range(val_test_size - len(test_index))]
            train_index = train_index + neg_idx
            assert len(val_index) == val_test_size
            assert len(test_index) == val_test_size
        elif dataset_name == 'enzymes':
            class2idx = {}
            for idx, data in enumerate(dataset):
                label = data.y.item()
                if label not in class2idx.keys():
                    class2idx[label] = [idx]
                else:
                    class2idx[label].append(idx)
            train_index, val_index, test_index = [], [], []
            for label in class2idx.keys():
                idx_list = class2idx[label]
                assert len(idx_list) == 100
                val_index = val_index + idx_list[:20]  # 0.2
                test_index = test_index + idx_list[20:40]  # 0.2
                train_index = train_index + idx_list[40:]  # 0.6
            assert len(train_index) == 360
            assert len(val_index) == 120
            assert len(test_index) == 120
        else:
            raise NotImplementedError

    return train_index, val_index, test_index
