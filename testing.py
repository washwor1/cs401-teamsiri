import torch
import torchaudio
import audioModel as am
import importDataset
import os
import soundfile as sf

device = 'cpu'

def set_tranform_function(origin_frequency, new_frequency, device):
    transformFunction = torchaudio.transforms.Resample(orig_freq = origin_frequency, new_freq=new_frequency)
    return transformFunction.to(device)

# Read the WAV file
stereo_file = "wavFiles/recordings/andrew_1_left.wav"
data, sample_rate = sf.read(stereo_file)

#for recording that creates 2 channels
if( data.shape[1] == 2):

    # Convert stereo audio to mono (keep only one channel)
    mono_data = data[:, 0]  # Extract the first channel (left channel)

    # Save the mono audio to a new WAV file
    mono_file = "wavFiles/recordings/andrew_1_left_mono.wav"
    sf.write(mono_file, mono_data, sample_rate)



# exit(0)
waveform, sample_rate = torchaudio.load('wavFiles/recordings/andrew_1_left_mono.wav')
# waveform, sample_rate = torchaudio.load('./SpeechCommands/speech_commands_v0.02/original_backup/up/5fadb538_nohash_0.wav')
transform = set_tranform_function(sample_rate, 8000, device)
# Load your PyTorch model
state_dict = torch.load('test.ptf')
model = am.M5()  # Create an instance of your model
model.load_state_dict(state_dict)  # Load the saved state dict into the model
model.eval()
    
output = am.predict(waveform, model, device, transform, importDataset.index_to_label, perform_transform=True)

print(output)


# directory = 'SpeechCommands/speech_commands_v0.02/original_backup/right/'  # Replace with the path to your directory
# file_names = os.listdir(directory)
# right = 0
# total_right = len(file_names)
# mapping = {}
# for file in file_names:
#     waveform, sample_rate = torchaudio.load(directory + file)
#     if(waveform.size()[1] != 16000):
#         pad_vals = (0, 16000 - waveform.size()[1])
#         padded_waveform = torch.nn.functional.pad(waveform, pad_vals, mode='constant', value=0)
#         output = am.predict(padded_waveform, model, device, transform, importDataset.index_to_label, perform_transform=True)
#     else:
#         output = am.predict(waveform, model, device, transform, importDataset.index_to_label, perform_transform=True)
#     try:
#         mapping[output] += 1
#     except:
#         mapping[output] = 1

# for key in mapping:
#     print(key, mapping[key])

# #print(output)