image_size = 32
n_channels = 3
patch_size = 4
n_classes = 10

d_model = 128
d_ffn = 768
n_blocks = 30

configs = {
    "Ti": {
        "image_size": image_size,
        "n_channels": n_channels,
        "patch_size": patch_size,
        "d_model": d_model,
        "d_ffn": d_ffn,
        "n_blocks": n_blocks,
        "n_classes": n_classes,
        "prob_0_L": [1, 0.5],
    },
    "S": {
        "image_size": image_size,
        "n_channels": n_channels,
        "patch_size": patch_size,
        "d_model": d_model * 2,
        "d_ffn": d_ffn * 2,
        "n_blocks": n_blocks,
        "n_classes": n_classes,
        "prob_0_L": [1, 0.5],
    },
    "B": {
        "image_size": image_size,
        "n_channels": n_channels,
        "patch_size": patch_size,
        "d_model": d_model * 4,
        "d_ffn": d_ffn * 4,
        "n_blocks": n_blocks,
        "n_classes": n_classes,
        "prob_0_L": [1, 0.5],
    },
}