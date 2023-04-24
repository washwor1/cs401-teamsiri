import matplotlib.pyplot as plt
import IPython.display as ipd
import torch.optim as optim
from tqdm import tqdm
import torchaudio
import argparse
import torch
import os
import tensorflow as tf
#local files
import audioModel
import importDataset
import pertubation
import librosa
import wave
import numpy as np


testAudio = None

graphDir = "graphs/"

torch.set_printoptions(threshold=torch.inf)

all_labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',\
          'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', \
          'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# arg is a string that is a label from labels array, see above 
#
#   return new target label index
#
#   if left is input, should return index of 'right' from labels
#
def get_target_label(old_label):

    return 1


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

    device = 'cpu' #get_device_type()
    num_workers, pin_memory = get_device_info(device)
    print("Device: " + str(device))

    #get training and testing set
    train_set = importDataset.SubsetSC("training")
    
    # with open("files.txt", "w") as f:
    #     f.writelines(train_set._walker)

    test_set = importDataset.SubsetSC("testing")


    #validation_set = importDataset.SubsetSC("validation")

    #train_set = importDataset.filter_selected_labels(train_set)

    # exit(0)
    np.savetxt("original.txt", test_set[0][0][0].numpy())
    fig = plt.figure()
    plt.plot(test_set[0][0][0].numpy())
    plt.title("Original audio: Right")
    fig.savefig("original.png")

    #waveform, sample_rate, utterance, *_ = train_set[50]
    
    transform = set_tranform_function(origin_frequency, new_frequency, device)

    #get testing and training loaders --> Will have to look at what this actually is!
    train_loader = importDataset.getTrainLoader(train_set, batch_size, True, num_workers, pin_memory)
    test_loader = importDataset.getTestLoader(test_set, batch_size, False, False, num_workers, pin_memory)

    # print(test_loader)
    # exit(0)

    #get model --> if statement for either loading in, or creating a new one!
    model = audioModel.M5()
    print(load_model)
    
    if(load_model == True):
        if(os.path.isfile(load_model_file) == False):
            print("\'%s\' does not exist, exiting" % load_model_file) #Force to rerun
            exit(1)
        else:
            model.load_state_dict(torch.load(load_model_file))
            # model.eval()
            model.to(device)
            train_model = False
    else:
        pass
        #print("Creating new model")
        #model.to(device)
        #train_model = True
        #test_model = False
        #data, target = next(iter(test_loader))
    
    # for i in range(0, 256):
    #     target[0] = 4
    waveform, *_ = train_set[2]
    # data, target = next(iter(test_loader))
    # for i in range(0, 256):
    #     target[0] = 4

    attack_model = pertubation.M5(n_input=waveform.shape[0], n_output=10)
    attack_model.load_state_dict(torch.load('model2.ptf'))
    # attack_model.train()
    attack_model = attack_model.to(device)
    # attack_model.eval()

    optimizer = audioModel.setOptimizer(model, learn_rate = 0.01, weight_decay=0.0001)
    # reduce the learning after 20 epochs by a factor of 10
    scheduler = audioModel.setScheduler(optimizer, step_size=10, gamma=0.1)

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

    output_y = [[0] * 10 for i in range(0, 10)]
    
    if(run_full_test == True): #will update later on!
        # attack_model.eval()
        for fileIndex in range(0, len(test_set)):
            waveform, sample_rate, utterance, *_ = test_set[fileIndex]
            if(waveform.size()[1] != 16000):
                pad_vals = (0, 16000 - waveform.size()[1])
                padded_waveform = torch.nn.functional.pad(waveform, pad_vals, mode='constant', value=0)
                output = audioModel.predict(padded_waveform, attack_model, device, transform, importDataset.index_to_label)
            else:
                output = audioModel.predict(waveform, attack_model, device, transform, importDataset.index_to_label)
            output_y[labels.index(utterance)][labels.index(output)] += 1
        #end for
        
        graph_data = []
        for y in output_y:
            total_tests = np.sum(y)
            graph_data.append([num / total_tests for num in y])
        # print(graph_data)
        graph_data = np.array(graph_data)
        fig = plt.figure()

        plt.imshow(graph_data, cmap='gist_earth', interpolation='nearest')
        plt.xticks(range(0, 10), labels=labels)
        plt.yticks(range(0, 10), labels=labels)
        plt.colorbar()
        plt.show()
        fig.savefig(graphDir + "heatGraph2.png")
    # exit(0)
    if(save_model == True):
        torch.save(model.state_dict(), save_model_file)
    
    waveform, *_ = train_set[0]

    attack_model = pertubation.M5(n_input=waveform.shape[0], n_output=10)
    attack_model.load_state_dict(torch.load('model2.ptf'))
    attack_model = attack_model.to(device)
    test_loader_iterator = iter(test_loader)
    currentBatch = 0
    pertubation_results = []
    batch_number = 0

    correct_label = []
    pertubated_label = []
    while True:
        try:
            #extract batches of wave_forms, and target labels
            data, target = next(test_loader_iterator)
            #changes target values to be what we want based on the utterance of current wave_form
            #
            # print(data.shape[0])
            for i in range(currentBatch, currentBatch + data.shape[0]):
                wave_form, sample_rate, utterance, *_ = test_set[i]
                correct_label.append(utterance)
                currentTarget = i % 256
                new_label_index = get_target_label(utterance)
                target[currentTarget] = new_label_index

            currentBatch += 256
            data, target = data.to(device), target.to(device)
            pertubation_results.append(pertubation.attack(attack_model, device, data, target, targeted=True))
            print(target)
            for p in range(0, pertubation_results[batch_number].shape[0]):
                new_prediction = audioModel.predict(pertubation_results[batch_number][p], attack_model, device, transform, importDataset.index_to_label)
                pertubated_label.append(new_prediction)
            # print(pertubation_results[0][0])
            batch_number += 1
            print("batch: " + str(batch_number))
            break
        except StopIteration:

            print("end")
            break
        
    mismatches = 0
    left_count = 0
    right = 0
    for i in range(0, len(pertubated_label)):
        if(correct_label[i] != pertubated_label[i]):
            mismatches += 1
        if(pertubated_label[i] == "no"):
            left_count += 1
        if(correct_label[i] == "right"):
            right += 1
    #print(correct_label[i] + " --> " + pertubated_label[i])
    print("mismatches" + str(mismatches))
    print("no count: " + str(left_count))
    print("right count: " + str(right))
    print(len(pertubated_label))
    
    
    exit(0)
    print(target)
    print(correct_label)    
    print(data[0][0][2])
    # print(target[0])
    target[0] = 6
    print(target[0])
    # print(type(adv_data))
    exit(0)
    # with open("t.txt", 'w') as f:
    #     f.write(str(adv_data))


    testThing = adv_data[0][0].numpy()

    fig = plt.figure()
    
    plt.plot(testThing)
    plt.title("Pertubation Audio: Right --> Left")
    fig.savefig("test.png")
    #librosa.display.waveshow(testThing)
    
    np.savetxt("pertubation.txt", testThing)

    #ipd.Audio(testThing, rate=len(testThing))

    with wave.open('output.wav', 'wb') as wave_file:
        wave_file.setnchannels(1)  # mono audio
        wave_file.setsampwidth(2)  # 16-bit audio
        wave_file.setframerate(16000)  # sampling rate
        wave_file.setnframes(testThing.shape[0])  # number of frames
        wave_file.writeframes(testThing.astype(np.int16))  # write audio data


    output = audioModel.predict(adv_data[0], model, device, transform, importDataset.index_to_label)

    print(output)

    
    #get target, and data!


    #adv_data = pertubation.attack(model, data, target, eps=EPS, alpha=ALPHA, iters=ITERS, targeted=CT)



# stop, go
# on, off
# yes, no
