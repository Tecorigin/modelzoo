import model.seresnet as seresnet

def Model(num_classes=1000, variant='18', pretrained=False):
    """
    Simple factory returning a SE-ResNet / SE-ResNeXt variant.
    variant: '18', '34', '50', '101', '152', 'next50', 'next101'
    """
    if variant == '18':
        return seresnet.seresnet18(pretrained=pretrained, num_classes=num_classes)
    elif variant == '34':
        return seresnet.seresnet34(pretrained=pretrained, num_classes=num_classes)
    elif variant == '50':
        return seresnet.seresnet50(pretrained=pretrained, num_classes=num_classes)
    elif variant == '101':
        return seresnet.seresnet101(pretrained=pretrained, num_classes=num_classes)
    elif variant == '152':
        return seresnet.seresnet152(pretrained=pretrained, num_classes=num_classes)
    elif variant.lower() == 'next50':
        return seresnet.seresnext50(pretrained=pretrained, num_classes=num_classes)
    elif variant.lower() == 'next101':
        return seresnet.seresnext101(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")

# 测试
if __name__ == "__main__":
    m = Model(num_classes=100, variant='50', pretrained=False)
    print(m)