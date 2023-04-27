
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch

#load labels --> I already generated this and saved it to a file #will have to pass in. cringe
all_labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',\
          'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', \
          'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

exclusion_labels = [ 'backward', 'bed', 'bird', 'cat', 'dog', 'eight', 'five', 'follow', 'forward',\
                    'four', 'happy', 'house', 'learn', 'marvin', 'nine', 'one', \
                    'seven', 'sheila', 'six', 'three', 'tree', 'two', 'visual', 'wow', 'zero']

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


'''
    Will filter the data based on which labels we are focusing on!
    I made up an exclusion list because of the problem of 'on' being found in 'one'
'''
def filter_selected_labels(file_set):
    if file_set == None:
        print("filter_selected_labels: train_set empty!")
        exit(0)

    filtered_set = []
    append_file = True

    for file in file_set:
        stripped_file = os.path.dirname(file)
        for e_label in exclusion_labels:
            if e_label in stripped_file:
                append_file = False
                break

        if append_file == True:
            filtered_set.append(file)
        append_file = True

    return filtered_set

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
            self._walker = filter_selected_labels(self._walker)
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
            self._walker = filter_selected_labels(self._walker)
        elif subset == "training_rir":
            excludes = load_list("rir_validation.txt") + load_list("rir_testing.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
            self._walker = filter_selected_labels(self._walker)
        elif subset == "testing_rir":
            excludes = load_list("testing_list.txt")
            self._walker = filter_selected_labels(self._walker)
            


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def getTrainLoader(train_set, batch_size, shuffle, num_workers, pin_memory, collate_fn = collate_fn):

    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def getTestLoader(test_set, batch_size, shuffle, drop_last, num_workers, pin_memory, collate_fn = collate_fn):

    return torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def getValidationLoader(validation_set, batch_size, shuffle, drop_last, num_workers, pin_memory, collate_fn = collate_fn):

    return torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )