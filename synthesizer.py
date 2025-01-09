import numpy as np
import os
import pyaudio
import soundfile as sf
import threading


class Synthesizer:
    def __init__(self):
        self.paudio = pyaudio.PyAudio()
        self.active_note = {}
        self.lock = threading.Lock()
        self.running = True

        self.sample_rate = 44100
        self.samples = self.load_samples()

        self.stream = self.paudio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=512,
            stream_callback=self.wav_audio_callback
        )

        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def load_samples(self):
        samples = {}
        for octave in range(1, 6):  # Octaves 1 to 5
            for note in ['a', 'as', 'b', 'c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs']:
                file_path = os.path.join('piano-88-notes', f"{octave}-{note}.wav")
                try:
                    data, _ = sf.read(file_path, dtype='float32')

                    # Convert stereo to mono by averaging the two channels
                    if data.ndim > 1 and data.shape[1] == 2:
                        data = data.mean(axis=1)

                    samples[f"{octave}-{note}"] = data
                except FileNotFoundError:
                    print(f"Sample file not found: {file_path}")
        return samples

    def note_on(self, octave, note, velocity):
        file_key = f"{octave}-{note}"
        if file_key in self.samples:
            self.active_note[file_key] = {
                'index': 0,
                'velocity': velocity / 127.0
            }
        else:
            print(f"Sample not loaded for note: {file_key}")

    def note_off(self, octave, note):
        file_key = f"{octave}-{note}"
        if file_key in self.active_note:
            del self.active_note[file_key]

    def get_note_file(self, octave, note):
        note_mapping = {
            'A': 'a', 'A#': 'as', 'Bb': 'as',
            'B': 'b', 'Cb': 'b', 'B#': 'c', 'C': 'c',
            'C#': 'cs', 'Db': 'cs', 'D': 'd',
            'D#': 'ds', 'Eb': 'ds', 'E': 'e', 'Fb': 'e',
            'E#': 'f', 'F': 'f', 'F#': 'fs', 'Gb': 'fs',
            'G': 'g', 'G#': 'gs', 'Ab': 'gs'
        }

        note_file = note_mapping.get(note)
        return f"{octave}-{note_file}.wav"

    def wav_audio_callback(self, in_data, frame_count, time_info, status):
        # pyaudio callback function
        output = np.zeros(frame_count, dtype=np.float32)
        with self.lock:
            active_notes = list(self.active_note.items())

        for note, info in active_notes:
            sample = self.samples[note]
            start_idx = info['index']
            end_idx = start_idx + frame_count

            if end_idx >= len(sample):
                end_idx = len(sample)
                with self.lock:
                    octave, note = note.split('-')
                    self.note_off(octave, note)
                continue

            output[:end_idx - start_idx] += sample[start_idx:end_idx] * info['velocity']
            info['index'] += frame_count

        if np.max(np.abs(output)) > 1.0:
            output /= np.max(np.abs(output))

        return (output.astype(np.float32).tobytes(), pyaudio.paContinue)

    def run(self):
        self.stream.start_stream()

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.paudio.terminate()
        self.thread.join()
