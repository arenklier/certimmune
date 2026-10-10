"""The decisive question: do the TRAINED weights survive the loop's reset?

PEFT does not nest a second adapter -- it finds the existing one named
"default" and hands it back. So the question is whether that adapter still
carries what the previous condition trained into it. Marking one lora_B tensor
answers it without training anything: lora_B starts at exactly zero, so any
non-zero value that survives is a value the previous condition would have put
there.
"""
import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model

CFG = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0)
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float32)
base_unet = unet

w1 = get_peft_model(unet, CFG)
name, tensor = next((n, p) for n, p in w1.named_parameters() if "lora_B" in n)
print("condition 1: %s starts at %.6f (lora_B is zero-init by design)" % (name.split('.')[-3], tensor.abs().sum()))
with torch.no_grad():
    tensor.fill_(0.1234)            # stand-in for "condition 1 trained this"
print("condition 1: after 'training', abs-sum = %.4f" % tensor.abs().sum())

pipe_unet = base_unet               # the loop's reset, verbatim
w2 = get_peft_model(pipe_unet, CFG)
t2 = dict(w2.named_parameters())[name]
print("condition 2: same tensor now  = %.4f  -> %s"
      % (t2.abs().sum(), "LEAKS (condition 2 starts from condition 1's LoRA)"
         if t2.abs().sum() > 0 else "clean reset"))
