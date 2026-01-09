import models.tresnet as tresnet

# --- replace the bottom part with this: ---

def Model(num_classes=1000, variant='M', remove_aa_jit=False):
    """
    Simple factory returning a TResNet variant defined in this file.
    variant: 'M' (medium), 'L' (large), 'XL' (extra-large)
    """
    model_params = {'num_classes': num_classes, 'remove_aa_jit': remove_aa_jit}
    if variant == 'M':
        return tresnet.TResnetM(model_params)
    elif variant == 'L':
        return tresnet.TResnetL(model_params)
    elif variant == 'XL':
        return tresnet.TResnetXL(model_params)
    else:
        raise ValueError(f"Unknown variant: {variant}")

# 测试
if __name__ == "__main__":
    m = Model(num_classes=100, variant='M')
    print(m)