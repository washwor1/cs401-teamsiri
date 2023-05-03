import sounddevice as sd
import soundfile as sf

# Set the sample rate and duration
sample_rate = 16000
duration = 1.0  # in seconds

# Record audio
audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
sd.wait()

# Save the recorded audio to a WAV file
file_path = 'recorded_audio.wav'
sf.write(file_path, audio, sample_rate)

print("Audio recorded and saved to:", file_path)

# import sounddevice as sd

# sd.default.device = 'WASAPI'  # Replace with the appropriate API for your system
# devices = sd.query_devices()
# print(devices)

# # Specify the desired input device index
# input_device_index = 0  # Replace with the appropriate index for your desired input device

# # Record audio using the specified input device
# audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, input_device=input_device_index)
# sd.wait()