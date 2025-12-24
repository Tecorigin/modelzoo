# model_factory.py

# sequencer
from model.vanilla_sequencer import (
    v_sequencer_s,
    v_sequencer_s_h,
    v_sequencer_s_pe,
)

from model.two_dim_sequencer import (
    sequencer2d_m,
    sequencer2d_l,
    sequencer2d_s_392,
    sequencer2d_m_392,
    sequencer2d_l_392,
    sequencer2d_s_unidirectional,
    sequencer2d_s_add,
    sequencer2d_s_h2x,
    sequencer2d_s_without_fc,
    sequencer2d_vertical,
    sequencer2d_s_horizontal,
    gru_sequencer2d_s,
    rnn_sequencer2d_s,
    sequencer2d_l_d4_3x,
)

_MODEL_TABLE = {
    # vanilla sequencer
    "v_sequencer_s": v_sequencer_s,
    "v_sequencer_s_h": v_sequencer_s_h,
    "v_sequencer_s_pe": v_sequencer_s_pe,

    # 2d sequencer
    "sequencer2d_m": sequencer2d_m,
    "sequencer2d_l": sequencer2d_l,
    "sequencer2d_s_392": sequencer2d_s_392,
    "sequencer2d_m_392": sequencer2d_m_392,
    "sequencer2d_l_392": sequencer2d_l_392,
    "sequencer2d_s_unidirectional": sequencer2d_s_unidirectional,
    "sequencer2d_s_add": sequencer2d_s_add,
    "sequencer2d_s_h2x": sequencer2d_s_h2x,
    "sequencer2d_s_without_fc": sequencer2d_s_without_fc,
    "sequencer2d_vertical": sequencer2d_vertical,
    "sequencer2d_s_horizontal": sequencer2d_s_horizontal,
    "gru_sequencer2d_s": gru_sequencer2d_s,
    "rnn_sequencer2d_s": rnn_sequencer2d_s,
    "sequencer2d_l_d4_3x": sequencer2d_l_d4_3x,

}


def Model(num_classes=100, model_name=None, **kwargs):
    """
    Unified model entry (NO timm).

    Args:
        num_classes (int): number of classes (可直接用位置参数传)
        model_name (str, optional): key in _MODEL_TABLE, 默认使用 'sequencer2d_s_392'
        **kwargs: 传给模型构造函数

    Returns:
        torch.nn.Module
    """
    if model_name is None:
        model_name = "sequencer2d_s_392"

    if model_name not in _MODEL_TABLE:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(_MODEL_TABLE.keys())}"
        )

    return _MODEL_TABLE[model_name](
        pretrained=False,
        num_classes=num_classes,
        **kwargs
    )

