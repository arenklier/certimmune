"""Same question, detected properly.

PEFT's injected module is peft.tuners.lora.layer.Linear -- its class NAME is
just "Linear", so grepping the type name finds nothing. The reliable marker is
the lora_A ModuleDict the adapter attaches, plus the module's __module__ path.
"""
import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model

CFG = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0)


def n_adapters(module):
    """number of injected adapter sites, and the adapter names on the first one"""
    sites = [(n, m) for n, m in module.named_modules() if hasattr(m, "lora_A")]
    names = list(sites[0][1].lora_A.keys()) if sites else []
    depth = 0
    if sites:  # is an adapter wrapping another adapter?
        inner = getattr(sites[0][1], "base_layer", None)
        while inner is not None:
            depth += 1
            inner = getattr(inner, "base_layer", None)
    return len(sites), names, depth


unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float32)
base_unet = unet

print("before      : sites=%d names=%s nest=%d" % n_adapters(base_unet))
w1 = get_peft_model(unet, CFG)
n1 = sum(p.requires_grad for p in w1.parameters())
print("after  1st  : sites=%d names=%s nest=%d   trainable=%d" % (*n_adapters(base_unet), n1))

# the loop's "reset", verbatim
restored = base_unet
w2 = get_peft_model(restored, CFG)
n2 = sum(p.requires_grad for p in w2.parameters())
print("after  2nd  : sites=%d names=%s nest=%d   trainable=%d" % (*n_adapters(base_unet), n2))

w3 = get_peft_model(base_unet, CFG)
n3 = sum(p.requires_grad for p in w3.parameters())
print("after  3rd  : sites=%d names=%s nest=%d   trainable=%d" % (*n_adapters(base_unet), n3))
