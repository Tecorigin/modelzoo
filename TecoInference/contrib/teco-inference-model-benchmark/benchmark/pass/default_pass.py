import tvm
from tvm import relay

def use_passes(ir_module):

    target = "sdaa"
    opt_level = 3

    seq_passes = [
                relay.transform.InferType(),
                relay.transform.SimplifyExpr(),
                relay.transform.CanonicalizeOps(),
                relay.transform.AlterOpLayout(),
                relay.transform.FoldConstant(),
                relay.transform.EliminateCommonSubexpr(),
                relay.transform.DeadCodeElimination(),
    ]

    with tvm.target.Target(target):
        if seq_passes is not None:
            seq = tvm.transform.Sequential(seq_passes)

            with tvm.transform.PassContext(opt_level):
                ir_module = seq(ir_module)
    return ir_module

