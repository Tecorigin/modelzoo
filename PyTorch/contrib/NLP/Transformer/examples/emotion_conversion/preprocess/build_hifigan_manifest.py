import torchaudio
import torch_sdaa
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="example: python create_hifigan_manifest.py --tsv /checkpoint/felixkreuk/datasets/vctk/splits/vctk_16khz/train.tsv --km /checkpoint/felixkreuk/experiments/hubert/hubert_feats/vctk_16khz_km_100/train.km --km_type hubert_100km > ~/tmp/tmp_mani.txt")
    parser.add_argument("--tsv", required=True, help="path to fairseq tsv file")
    parser.add_argument("--km", required=True, help="path to a km file generated by HuBERT clustering")
    parser.add_argument("--km_type", required=True, help="name of the codes in the output json (for example: 'cpc_100km')")
    args = parser.parse_args()

    km_lines = open(args.km, "r").readlines()
    tsv_lines = open(args.tsv, "r").readlines()
    assert len(km_lines) == len(tsv_lines) - 1, "tsv and km files are not of the same length!"

    wav_root = tsv_lines[0].strip()
    tsv_lines = tsv_lines[1:]

    for tsv_line, km_line in zip(tsv_lines, km_lines):
        tsv_line, km_line = tsv_line.strip(), km_line.strip()
        wav_basename, wav_num_frames = tsv_line.split("\t")
        wav_path = wav_root + "/" + wav_basename
        wav_info = torchaudio.info(wav_path)
        assert int(wav_num_frames) == wav_info.num_frames, "tsv duration and actual duration don't match!"
        wav_duration = wav_info.num_frames / wav_info.sample_rate
        manifest_line = {"audio": wav_path, "duration": wav_duration, args.km_type: km_line}
        print(json.dumps(manifest_line))

if __name__ == "__main__":
    """
    usage:
    python create_hifigan_manifest.py \
            --tsv /checkpoint/felixkreuk/datasets/vctk/manifests/vctk_16khz/valid.tsv \
            --km /checkpoint/felixkreuk/datasets/vctk/manifests/vctk_16khz/hubert_km_100/valid.km \
            --km_type hubert \
            > /checkpoint/felixkreuk/datasets/vctk/manifests/vctk_16khz/hubert_km_100/hifigan_valid_manifest.txt
    """
    main()
