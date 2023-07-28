import whisper
from whisper.utils import get_writer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = whisper.load_model("base")
audio = "m-5.mp3"
result = model.transcribe(audio,fp16=False)
output_directory = "."


# Save as an SRT file
srt_writer = get_writer("srt", output_directory)
srt_writer(result, audio)
