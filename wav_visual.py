import wave
import matplotlib.pyplot as plt
import numpy as np

file_path = "output.wav"
# file_path = "SpeechCommands/speech_commands_v0.02/original_backup/right/0a2b400e_nohash_0.wav"

wave_file = wave.open(file_path, 'rb')

frames = wave_file.readframes(-1)
sample_rate = wave_file.getframerate()

audio_data = np.frombuffer(frames, dtype=np.int16)
wave_file.close()

plt.figure()
plt.plot(audio_data)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Waveform')


plt.show()


