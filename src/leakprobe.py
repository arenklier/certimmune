"""Does lora_calibration.py's per-condition reset actually reset anything?

The loop captures base_unet = pipe.unet once, then for each condition does
  pipe.unet = base_unet ;  pipe = add_lora(pipe, ...)
with a comment saying this avoids relying on peft's unload(). That is only a
reset if get_peft_model leaves the object it was handed untouched. It does not:
PEFT injects adapter modules into the module tree in place. This checks it
directly, on CPU, with no training -- it either finds LoRA modules inside the
supposedly pristine base_unet or it does not.
"""
import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model

CFG = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0)


def n_lora(module):
    return sum(1 for _, m in module.named_modules() if "lora" in type(m).__name__.lower())


unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float32)
base_unet = unet                      # exactly what the script captures
print("before add_lora     : lora modules inside base_unet = %d" % n_lora(base_unet))

wrapped = get_peft_model(unet, CFG)   # exactly what add_lora does
print("after  add_lora     : lora modules inside base_unet = %d" % n_lora(base_unet))
print("                      base_unet is the wrapped model's base: %s"
      % (wrapped.base_model.model is base_unet))

# now do what the next loop iteration does
restored = base_unet
wrapped2 = get_peft_model(restored, CFG)
print("after  2nd add_lora : lora modules inside base_unet = %d" % n_lora(base_unet))
tr = [n for n, p in wrapped2.named_parameters() if p.requires_grad]
print("trainable tensors in 2nd model = %d  (nested adapters if this keeps growing)" % len(tr))
