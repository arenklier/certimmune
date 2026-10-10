"""Cheap screen: does ANY published protective perturbation survive IMPRESS?

Runs IMPRESS (official code, official default hyperparameters) on identity_00's
unprotected image and on each of the five protective perturbations this paper
evaluates, and reports post-purification ArcFace similarity to the clean
identity. This is a SCREEN, not a certification: 3 seeds per condition, one
noise level. Its only purpose is to decide whether a full certification sweep
over IMPRESS's noise parameter is worth running.
"""
import json, os, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'Impress'))
from impress import impress

from common import load_sd_pipeline, load_face_app, arcface_embed, cosine_sim
from delta_optimize import pil_to_pixel_tensor, pixel_to_vae_input


def vae_fwd(vae):
    def _f(x):
        rec = vae.decode(vae.encode(x).latent_dist.mean).sample
        class O: pass
        o = O(); o.sample = rec; return o
    return _f


def purify(vae_fp32, x01, seed, noise=0.1):
    torch.manual_seed(seed)
    xv = pixel_to_vae_input(x01).float()
    xp = impress(xv, model=vae_fwd(vae_fp32), eps=0.1, iters=100, lr=0.01,
                 clamp_min=-1, clamp_max=1, pur_alpha=0.1, noise=noise)
    return (xp / 2 + 0.5).clamp(0, 1)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    R = os.path.join(base, '..', 'results')
    pipe = load_sd_pipeline(); app = load_face_app()
    vae32 = pipe.vae.float()

    clean = Image.open(os.path.join(base, '..', 'data', 'celebahq_sample', 'identity_00.png')).convert('RGB')
    emb_clean = arcface_embed(app, clean)

    conditions = {
        'unprotected':      os.path.join(base, '..', 'data', 'celebahq_sample', 'identity_00.png'),
        'ours (EOT)':       os.path.join(R, 'delta_immunized_eot.png'),
        'PhotoGuard':       os.path.join(R, 'photoguard_baseline_immunized.png'),
        'Anti-DreamBooth':  os.path.join(R, 'antidreambooth_baseline_immunized.png'),
        'Glaze':            os.path.join(R, 'glaze_baseline_immunized.png'),
        'AntiPure':         os.path.join(R, 'antipure_baseline_immunized.png'),
    }

    out = {}
    for name, path in conditions.items():
        if not os.path.exists(path):
            print(f'SKIP {name}: missing {path}'); continue
        x01 = pil_to_pixel_tensor(Image.open(path).convert('RGB'))
        sims = []
        for s in range(3):
            xp = purify(vae32, x01, seed=s)
            arr = (xp[0].permute(1,2,0).float().cpu().numpy()*255).round().astype(np.uint8)
            e = arcface_embed(app, Image.fromarray(arr))
            sims.append(float(cosine_sim(emb_clean, e)) if e is not None else 0.0)
            torch.cuda.empty_cache()
        out[name] = {'sims': sims, 'mean': float(np.mean(sims))}
        print(f'{name:20s} post-IMPRESS ArcFace sim: {np.mean(sims):.4f}   {[round(v,3) for v in sims]}')

    with open(os.path.join(R, 'impress_screen_all.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSaved impress_screen_all.json')


if __name__ == '__main__':
    main()
