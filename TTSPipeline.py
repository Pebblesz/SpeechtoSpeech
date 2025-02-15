import sounddevice as sd
import numpy as np
import soundfile as sf
import torch
from queue import Queue
from threading import Thread, Event
from transformers import pipeline
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
from huggingface_hub import snapshot_download
import os
from scipy.signal import resample


# CONFIGURATION
DEBUG = True
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # Process audio in 2-second chunks
VIRTUAL_DEVICE_NAME = "CABLE Input (VB-Audio Virtual Cable)"
REFERENCE_PATH = "reference.wav"
MIN_WORDS_FOR_TTS = 2  # Minimum number of words required for TTS output


# GLOBAL STATE
audio_queue = Queue(maxsize=5)
processing_event = Event()
device_tts = torch.device("cpu")
last_spoken_text = ""  # Track the last spoken text to avoid repetition

# MODEL INITIALIZATION
checkpoint_path = snapshot_download(repo_id="coqui/XTTS-v2")
config = XttsConfig()
config.load_json(os.path.join(checkpoint_path, "config.json"))
tts_model = Xtts.init_from_config(config)
tts_model.load_checkpoint(config, checkpoint_dir=checkpoint_path, eval=True)
tts_model.to(device_tts).eval()

stt_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float32
)


# AUDIO FUNCTIONS
def audio_callback(indata, frames, time, status):
    """Continuous audio input callback"""
    if status:
        print(f"Audio input error: {status}")
    audio_queue.put(indata.copy())


def get_virtual_device():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if VIRTUAL_DEVICE_NAME in dev['name'] and dev['max_output_channels'] > 0:
            return i, int(dev['default_samplerate'])
    return None, None


def process_audio():
    """Dedicated processing thread"""
    global last_spoken_text  # Track the last spoken text
    virtual_device_id, target_rate = get_virtual_device()

    while processing_event.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=1)
            audio_chunk = audio_chunk.squeeze().astype(np.float32)
            audio_chunk /= np.max(np.abs(audio_chunk)) + 1e-8

            # STT Processing
            result = stt_pipe(
                {"array": audio_chunk, "sampling_rate": SAMPLE_RATE},
                return_timestamps=False
            )
            text = result["text"].strip()

            if DEBUG and text:
                print(f"[STT] {text}")

            # TTS Processing (only if new text is detected and has enough words)
            if text and text != last_spoken_text:
                word_count = len(text.split())
                if word_count >= MIN_WORDS_FOR_TTS:  # Check word count
                    outputs = tts_model.synthesize(
                        text, config,
                        speaker_wav=REFERENCE_PATH,
                        language="en",
                        gpt_cond_len=3
                    )
                    audio = outputs["wav"].squeeze().astype(np.float32)

                    # Resample and output
                    if target_rate != 24000:
                        new_length = int(len(audio) * (target_rate / 24000))
                        audio = resample(audio, new_length)

                    sd.play(audio, target_rate, device=virtual_device_id, blocking=False)
                    last_spoken_text = text  # Update the last spoken text
                elif DEBUG:
                    print(f"[DEBUG] Ignored single-word input: {text}")

        except Exception as e:
            if DEBUG:
                print(f"[Processing Error] {str(e)}")


# MAIN FLOW
def main():
    # Verify reference audio
    if not os.path.exists(REFERENCE_PATH):
        print("Creating default reference.wav...")
        silence = np.zeros(16000 * 3, dtype=np.float32)
        sf.write(REFERENCE_PATH, silence, 16000)

    # Start processing thread
    processing_event.set()
    processor_thread = Thread(target=process_audio)
    processor_thread.start()

    try:
        # Start continuous recording
        with sd.InputStream(callback=audio_callback,
                            samplerate=SAMPLE_RATE,
                            channels=1,
                            blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
            print("System active - speak naturally...")
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nShutting down...")
        processing_event.clear()
        processor_thread.join()


if __name__ == "__main__":
    main()