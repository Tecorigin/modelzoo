# eca_model_factory.py
import model.eca_resnet as eca_net

def Model(num_classes=1000):
    """
    Minimal ECA model factory, returns a model instance.
    """
    model_type='eca_resnet50'
    k_size=[3,3,3,3]
    pretrained=False
    model_func = getattr(eca_net, model_type)
    return model_func(k_size=k_size, num_classes=num_classes, pretrained=pretrained)


# 测试
if __name__ == "__main__":
    m = Model(model_type='eca_resnet50', k_size=[3,3,3,3], num_classes=100)
    print(m)