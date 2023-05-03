import wave
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wave
import numpy as np
import matplotlib.pyplot as plt

#Change input file here
file = 'wavFiles/recordings/andrew_1_right.wav'

with wave.open(file, 'rb') as wav_file:
    # Read all frames from the WAV file
    audio_data = wav_file.readframes(-1)

    # Convert the audio data to a NumPy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

# Get the audio file's properties
sample_rate = wav_file.getframerate()
duration = len(audio_array) / sample_rate

# Create the time axis for the waveform plot
time = np.linspace(0, duration, num=len(audio_array))

# Plot the audio waveform
plt.figure(figsize=(10, 4))
plt.plot(time, audio_array, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.grid(True)
plt.show()
