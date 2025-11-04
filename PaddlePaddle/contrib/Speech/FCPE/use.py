import numpy as np
from paddlefcpe import spawn_bundled_infer_model, bundled_infer_model_unit_test
import paddle
import librosa

# configure device and target hop_size
device = 'sdaa'
sr = 16000
hop_size = 160

# load audio
audio, sr = librosa.load('test.wav', sr=sr)
audio = librosa.to_mono(audio)
audio_length = len(audio)
f0_target_length=(audio_length // hop_size) + 1
audio = paddle.to_tensor(audio).astype(paddle.float32).unsqueeze(0).unsqueeze(-1).to(device)

# load model
model = spawn_bundled_infer_model(device=device)

# infer
f0 = model.infer(
    audio,
    sr=sr,
    decoder_mode='local_argmax',
    threshold=0.006,
    f0_min=80,
    f0_max=880,
    interp_uv=True,
    output_interp_target_length=f0_target_length,
)
print(f0.shape)
bundled_infer_model_unit_test("test.wav")