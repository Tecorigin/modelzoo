# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import numpy
import os
import librosa
import soundfile as sf
root_dir = r'XXX'
raw_dir = os.path.join(root_dir, 'raw')
music_dir = os.path.join(root_dir, 'music')
f0_mir1k_dir = os.path.join(root_dir, 'f0_mir1k')
f0_ptdb = os.path.join(root_dir, 'f0')
audio_mir1k = os.path.join(root_dir, 'audio_mir1k')
audio_ptdb = os.path.join(root_dir, 'audio')


file_list = os.listdir(raw_dir)

for file_name in file_list:
    print(file_name)
    if file_name[-3:] == '.pv':
        pipe_mode = 'PV'
    else:
        pipe_mode = 'WAV'
    if file_name[:4] == 'mic_':
        dataset_mode = 'ptdb'
    else:
        dataset_mode = 'mir1k'

    dataset_mode = 'ptdb'

    if pipe_mode == 'PV':
        pv_np = numpy.loadtxt(os.path.join(raw_dir, file_name))
        pv_np = numpy.insert(pv_np, 0, 0)
        mask = pv_np == 0
        f0_np = 440 * (2.0 ** ((pv_np - 69.0) / 12.0))
        f0_np[mask] = 0
        if dataset_mode == 'ptdb':
            numpy.save(os.path.join(f0_ptdb, file_name[:-3] + '.wav.npy'), f0_np)
        else:
            numpy.save(os.path.join(f0_mir1k_dir, file_name[:-3] + '.wav.npy'), f0_np)
    else:
        if dataset_mode == 'ptdb':
            audio, sr = librosa.load(os.path.join(raw_dir, file_name), mono=False, sr=None)
            assert sr == 16000
            assert len(audio.shape) == 1
            sf.write(os.path.join(audio_ptdb, file_name), audio, sr)
        else:
            audio, sr = librosa.load(os.path.join(raw_dir, file_name), mono=False, sr=None)
            assert sr == 16000
            assert len(audio.shape) == 2
            sf.write(os.path.join(audio_mir1k, file_name), audio[1], sr)
            sf.write(os.path.join(music_dir, file_name), audio[0], sr)


"""
in_wav, in_sr = librosa.load(os.path.join(raw_dir, "abjones_1_02.wav"), mono=False, sr=None)
sf.write("test_0.wav", in_wav[0], in_sr)  # music
sf.write("test_1.wav", in_wav[1], in_sr)  # voice
print(in_wav.shape)
print(in_sr)
print(os.path.join(raw_dir, "abjones_1_02.wav"))
"""
"""
"""
"""
a = numpy.array([1, 2, 0, 0, 0, 90, 58.03715, 66, 66, 66, 66])
# 给a的前面加一个0
b = numpy.insert(a, 0, 0)
# 为0的地方为False，其他为True
mask = b == 0
# f0 (Hz)= 440 * (2.0 ** ((b - 69.0) / 12.0))
c = 440 * (2.0 ** ((b - 69.0) / 12.0))
# 将c中为0的地方置为0
c[mask] = 0
print(c)
print(mask)
print(b)
"""
