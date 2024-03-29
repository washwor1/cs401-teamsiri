import matplotlib.pyplot as plt
import IPython.display as ipd
import torch.optim as optim
from tqdm import tqdm
import torchaudio
import argparse
import os
import tensorflow as tf
#local files
import audioModel
import importDataset
import pertubation
import librosa
import wave
import numpy as np
import torch
import audioModel as am
from torch.utils.mobile_optimizer import optimize_for_mobile


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
    mappedIndex = {"yes":   1, "no":    0,
                   "up":    3, "down":  2,
                   "left":  5, "right": 4,
                   "on":    7, "off":   6,
                   "stop":  9, "go":    8}
    return mappedIndex[old_label]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device_type():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device_info(device):
    if str(device) == "cuda":
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

    device = get_device_type()
    num_workers, pin_memory = get_device_info(device)
    print("Device: " + str(device))
    print("Worker: " + str(num_workers))
    print("Memory: " + str(pin_memory))
    print("Getting training, testing sets")
    #get training and testing set
    train_set = importDataset.SubsetSC("training")
    test_set = importDataset.SubsetSC("testing")
    print("Getting Transform function")
    #get transform function
    transform = set_tranform_function(origin_frequency, new_frequency, device)
    print("Getting loaders")
    #get testing and training loaders --> Will have to look at what this actually is!
    train_loader = importDataset.getTrainLoader(train_set, batch_size, True, num_workers, pin_memory)
    test_loader = importDataset.getTestLoader(test_set, batch_size, False, False, num_workers, pin_memory)
    
    model = audioModel.M5()

    if(load_model == True):
        if(os.path.isfile(load_model_file) == False):
            print("\'%s\' does not exist, exiting" % load_model_file)
            exit(1)
        else:
            model.load_state_dict(torch.load(load_model_file))
            model.eval()
            model.to(device)
            train_model = False
    else:
        print("Creating new model")
        train_model = True
        test_model = False
        data, target = next(iter(test_loader))
    
    model.to(device)

    print("Setting optimizer")
    optimizer = audioModel.setOptimizer(model, learn_rate = 0.01, weight_decay=0.0001)
    print("Setting scheduler")
    # reduce the learning after 20 epochs by a factor of 10
    scheduler = audioModel.setScheduler(optimizer, step_size=10, gamma=0.1)
    
    pbar_update = 1 / (len(train_loader) + len(test_loader))
    
    if(train_model == True):
        with tqdm(total=n_epoch) as pbar:
            for epoch in range(1, n_epoch + 1):
                audioModel.train(model, epoch, log_interval, train_loader, device, transform, optimizer, pbar, pbar_update)
                audioModel.test(model, epoch, test_loader, device, transform, pbar, pbar_update)
                scheduler.step()

    elif(test_model == True):
        with tqdm(total=1) as pbar:
            audioModel.test(model, 1, test_loader, device, transform, pbar, pbar_update)
            scheduler.step()
    
    if(run_full_test == True):
        output_y = [[0] * 10 for i in range(0, 10)]
        for fileIndex in range(0, len(test_set)):
            waveform, sample_rate, utterance, *_ = test_set[fileIndex]
            #pad waveforms that are not length 16000
            if(waveform.size()[1] != 16000):
                pad_vals = (0, 16000 - waveform.size()[1])
                padded_waveform = torch.nn.functional.pad(waveform, pad_vals, mode='constant', value=0)
                output = audioModel.predict(padded_waveform, model, device, transform, importDataset.index_to_label, perform_transform=True)
            else:
                output = audioModel.predict(waveform, model, device, transform, importDataset.index_to_label, perform_transform=True)
            output_y[labels.index(utterance)][labels.index(output)] += 1
        #end for
        
        graph_data = []
        for y in output_y:
            total_tests = np.sum(y)
            graph_data.append([num / total_tests for num in y])

        graph_data = np.array(graph_data)
        fig = plt.figure()

        plt.imshow(graph_data, cmap='gist_earth', interpolation='nearest')
        plt.xticks(range(0, 10), labels=labels)
        plt.yticks(range(0, 10), labels=labels)
        plt.colorbar()
        plt.show()
        fig.savefig(graphDir + "heatGraph2.png")

    if(save_model == True):
        model = model.to('cpu')
        torch.save(model.state_dict(), save_model_file)
        example_input = torch.randn(1, 1, 16000)
        # Convert the PyTorch model to TorchScript
        traced_script_module = torch.jit.trace(model, example_input)
        optimized = optimize_for_mobile(traced_script_module)
        # Save the TorchScript model
        traced_script_module._save_for_lite_interpreter('models/##_rir_model_app_3.ptl')
    
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
