"""Certify against IMPRESS as a THIRD purifier architecture.

IMPRESS (Cao et al., NeurIPS 2023) fits this framework's structural assumption
P_theta(x') = D_theta(x' + eps), eps ~ N(0, sigma^2 I): its released code
initializes purification at x' + N(0, pur_noise^2) and then runs a
VAE-consistency optimization as the (iterative, deterministic) map D_theta.
The continuous parameter certified over here is therefore sigma = pur_noise,
exactly analogous to DiffPure's noise scale -- but the map D_theta is
mechanistically different (VAE fixed-point optimization, not diffusion
reverse-denoising), which is the point: it tests whether the certification
machinery transfers to an architecture outside the diffusion-reverse family.

Reuses certify.py's Clopper-Pearson + grid-Lipschitz certificate unchanged.
"""
import argparse, json, os, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'Impress'))
from impress import impress

from common import load_sd_pipeline, load_face_app, arcface_embed, cosine_sim
from delta_optimize import pil_to_pixel_tensor, pixel_to_vae_input
from certify import clopper_pearson_upper, certify


def vae_fwd(vae):
    def _f(x):
        rec = vae.decode(vae.encode(x).latent_dist.mean).sample
        class O: pass
        o = O(); o.sample = rec; return o
    return _f


def impress_pass(vae32, x01, sigma, seed, iters=100, lr=0.01, pur_eps=0.1, pur_alpha=0.1):
    torch.manual_seed(seed)
    xv = pixel_to_vae_input(x01).float()
    xp = impress(xv, model=vae_fwd(vae32), eps=pur_eps, iters=iters, lr=lr,
                 clamp_min=-1, clamp_max=1, pur_alpha=pur_alpha, noise=sigma)
    x01p = (xp / 2 + 0.5).clamp(0, 1)
    arr = (x01p[0].permute(1, 2, 0).float().detach().cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def sweep(vae32, face_app, x01, emb_clean, tau, sigma_grid, n_seeds):
    results = {}
    for s in sigma_grid:
        k = 0
        for seed in range(n_seeds):
            pur = impress_pass(vae32, x01, s, seed)
            emb = arcface_embed(face_app, pur)
            sim = cosine_sim(emb_clean, emb) if emb is not None else 0.0
            if sim >= tau:
                k += 1
            torch.cuda.empty_cache()
        results[s] = {'k': k, 'n': n_seeds, 'phat': k / n_seeds}
        print(f'  sigma={s}: k={k}/{n_seeds}')
    return results


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    R = os.path.join(base, '..', 'results')
    ap.add_argument('--clean', default=os.path.join(base, '..', 'data', 'celebahq_sample', 'identity_00.png'))
    ap.add_argument('--immunized', default=os.path.join(R, 'delta_immunized_eot.png'))
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--sigma-grid', type=float, nargs='+', default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument('--n-seeds', type=int, default=20)
    ap.add_argument('--alpha-mc', type=float, default=0.05)
    ap.add_argument('--out', default=os.path.join(R, 'certificate_impress.json'))
    args = ap.parse_args()

    pipe = load_sd_pipeline(); app = load_face_app()
    vae32 = pipe.vae.float()

    clean_img = Image.open(args.clean).convert('RGB')
    emb_clean = arcface_embed(app, clean_img)
    x_clean = pil_to_pixel_tensor(clean_img)
    x_imm = pil_to_pixel_tensor(Image.open(args.immunized).convert('RGB'))

    print(f'=== IMPRESS certification: tau={args.tau}, sigma grid={args.sigma_grid}, n={args.n_seeds} ===')
    print('--- UNPROTECTED ---')
    r_un = sweep(vae32, app, x_clean, emb_clean, args.tau, args.sigma_grid, args.n_seeds)
    print('--- IMMUNIZED ---')
    r_im = sweep(vae32, app, x_imm, emb_clean, args.tau, args.sigma_grid, args.n_seeds)

    out = {'tau': args.tau, 'sigma_grid': args.sigma_grid, 'n_seeds': args.n_seeds,
           'purifier': 'IMPRESS (VAE-consistency, sigma=pur_noise)',
           'unprotected': {'results': r_un,
                           'marginal': certify(r_un, args.sigma_grid, args.alpha_mc, False),
                           'simultaneous': certify(r_un, args.sigma_grid, args.alpha_mc, True)},
           'immunized':   {'results': r_im,
                           'marginal': certify(r_im, args.sigma_grid, args.alpha_mc, False),
                           'simultaneous': certify(r_im, args.sigma_grid, args.alpha_mc, True)}}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print()
    print('unprotected alpha:', out['unprotected']['marginal']['alpha_final'])
    print('immunized   alpha:', out['immunized']['marginal']['alpha_final'])
    print('saved', args.out)


if __name__ == '__main__':
    main()
