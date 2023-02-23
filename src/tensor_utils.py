
from infer import InferModule
from tensor_encoder import TensorEncoder


def build_infer_module(clauses, atoms, lang, rgm, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, rgm=rgm, device=device)
    I = te.encode()
    im = InferModule(I, m=m, infer_step=infer_step, device=device, train=train)
    return im