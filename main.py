import torch
import torchaudio
import torch.optim as optim
from tqdm import tqdm
import os

import audioModel
import importDataset
import matplotlib.pyplot as plt
import IPython.display as ipd
import argparse

torch.set_printoptions(threshold=torch.inf)

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',\
          'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', \
          'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device_type():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device_info(device):
    if device == "cuda":
        return 1, True
    else:
        return 0, False

def set_tranform_function(origin_frequency, new_frequency, device):
    transformFunction = torchaudio.transforms.Resample(orig_freq=origin_frequency, new_freq=new_frequency)
    return transformFunction.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="420 Audio Adverseiral Model")
    parser.add_argument("--create_model", default=False, type=bool, help="Create Audio model")
    parser.add_argument("--epoch_stop", default=40, type=int, help="Epoch stop for training")
    parser.add_argument("--save_model", default=False, type=bool, help="Number of particles")
    parser.add_argument("--save_file", default="model.pt", type=str, help="Model file name")
    parser.add_argument("--load_model", default=False, type=bool, help="Load a model")
    parser.add_argument("--load_file", default="", type=str, help="File to load")
    parser.add_argument("--sample_rate", default=8000, type=int, help="Sample rate for transformation")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for model training")
    parser.add_argument("--test_model", default=False, type=bool, help="Test the model")
    parser.add_argument("--run_full_test", default=False, type=bool, help="Test the model on all test data")
    # parser.add_argument("--sample_rate", default=8000, type=int, help="Sample rate for transformation")
    # parser.add_argument("--sample_rate", default=8000, type=int, help="Sample rate for transformation")
    

    args = parser.parse_args()

    log_interval     = 20
    origin_frequency = 16000
    new_frequency    = args.sample_rate
    n_epoch          = args.epoch_stop
    batch_size       = args.batch_size
    load_model_file  = args.load_file
    save_model_file  = args.save_file
    save_model       = args.save_model
    load_model       = args.load_model
    create_model     = args.create_model
    test_model       = args.test_model
    run_full_test    = args.run_full_test
    train_model      = False

    device = get_device_type()
    num_workers, pin_memory = get_device_info(device)


    #get training and testing set
    train_set = importDataset.SubsetSC("training")
    test_set = importDataset.SubsetSC("testing")
    validation_set = importDataset.SubsetSC("validation")

    #waveform, sample_rate, utterance, *_ = train_set[50]
    
    transform = set_tranform_function(origin_frequency, new_frequency, device)

    #get testing and training loaders --> Will have to look at what this actually is!
    train_loader = importDataset.getTrainLoader(train_set, batch_size, True, num_workers, pin_memory)
    test_loader = importDataset.getTestLoader(test_set, batch_size, False, False, num_workers, pin_memory)

    #get model --> if statement for either loading in, or creating a new one!
    model = audioModel.M5()
    print(load_model)
    if(load_model == True):
        if(os.path.isfile(load_model_file) == False):
            print("\'%s\' does not exist, exiting" % load_model_file) #Force to rerun
            exit(1)
        else:
            model.load_state_dict(torch.load("test.ptf"))
            model.eval()
            model.to(device)
            train_model = False
    else:
        print("Creating new model")
        model.to(device)
        train_model = True
        test_model = False
    
    optimizer = audioModel.setOptimizer(model, learn_rate = 0.01, weight_decay=0.0001)
    # reduce the learning after 20 epochs by a factor of 10
    scheduler = audioModel.setScheduler(optimizer, step_size=20, gamma=0.1)

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    if(train_model == True):
        #training and testing model!
        with tqdm(total=n_epoch) as pbar:
            for epoch in range(1, n_epoch + 1):
                audioModel.train(model, epoch, log_interval, train_loader, device, transform, optimizer, pbar, pbar_update)
                audioModel.test(model, epoch, test_loader, device, transform, pbar, pbar_update)
                scheduler.step()

    elif(test_model == True): #get rid of pbar, not needed, only running one test on it. 
        with tqdm(total=1) as pbar:
            audioModel.test(model, 1, test_loader, device, transform, pbar, pbar_update)
            scheduler.step()
    
    if(run_full_test == True):
        for fileIndex in range(0, len(train_set)):
            waveform, sample_rate, utterance, *_ = train_set[fileIndex]
            ipd.Audio(waveform.numpy(), rate=sample_rate)

            output = audioModel.predict(waveform, model, device, transform, importDataset.index_to_label)
    
    if(save_model == True):
        torch.save(model.state_dict(), save_model_file)


# stop, go
# on, off
# yes, no