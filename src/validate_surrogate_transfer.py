"""Sec 9 item 12: quantify facenet-to-ArcFace transfer as its own claim.

delta is optimized against a differentiable facenet-pytorch/VGGFace2 surrogate
(ArcFace is not differentiable), then certified and evaluated strictly with
ArcFace. That every certificate comes out non-vacuous is *consistent* with the
surrogate transferring, but the paper never isolated it. This script does: it
embeds a heterogeneous pool of this project's own images (clean, protected by
five different mechanisms, and purifier outputs) in BOTH spaces, measures
similarity-to-clean in each, and reports the rank and linear correlation
between them. High correlation means the surrogate ranks perturbations the way
the certification metric does; low correlation would mean our certificates
could be right for the wrong reason.
"""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image
from scipy import stats

from common import load_face_app, arcface_embed, cosine_sim
from delta_optimize import load_facenet, facenet_embed, pil_to_pixel_tensor


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    R = os.path.join(base, '..', 'results')
    D = os.path.join(base, '..', 'data', 'celebahq_sample')

    app = load_face_app()
    net = load_facenet()

    clean_path = os.path.join(D, 'identity_00.png')
    clean_img = Image.open(clean_path).convert('RGB')
    emb_a_clean = arcface_embed(app, clean_img)
    with torch.no_grad():
        emb_f_clean = facenet_embed(net, pil_to_pixel_tensor(clean_img))

    # heterogeneous pool: identity_00 protected by every mechanism, plus purifier outputs
    cands = [
        ('clean', clean_path),
        ('ours-EOT', os.path.join(R, 'delta_immunized_eot.png')),
        ('ours-EOT-full', os.path.join(R, 'delta_immunized_eot_full.png')),
        ('ours-single-theta', os.path.join(R, 'delta_immunized_single.png')),
        ('ours-17step', os.path.join(R, 'delta_computematched17_immunized.png')),
        ('ours-100step', os.path.join(R, 'delta_stepcount100_immunized.png')),
        ('fast-EOT', os.path.join(R, 'delta_immunized_fasteot_wcmatch.png')),
        ('fast-EOT-k1', os.path.join(R, 'delta_immunized_fasteot_k1_155_00.png')),
        ('PhotoGuard', os.path.join(R, 'photoguard_baseline_immunized.png')),
        ('Anti-DreamBooth', os.path.join(R, 'antidreambooth_baseline_immunized.png')),
        ('Glaze', os.path.join(R, 'glaze_baseline_immunized.png')),
        ('AntiPure', os.path.join(R, 'antipure_baseline_immunized.png')),
        ('AntiPure-ETGonly', os.path.join(R, 'antipure_baseline_etgonly_immunized.png')),
        ('AntiPure-PFGonly', os.path.join(R, 'antipure_baseline_pfgonly_immunized.png')),
        ('IMPRESS-purified-unprot', os.path.join(R, 'impress_purified_unprotected.png')),
        ('IMPRESS-purified-ours', os.path.join(R, 'impress_purified_eotdelta.png')),
        ('eps4', os.path.join(R, 'immunized_identity_00_eps4.png')),
        ('eps16', os.path.join(R, 'immunized_identity_00_eps16.png')),
    ]

    rows = []
    for name, p in cands:
        if not os.path.exists(p):
            print(f'skip (missing): {name}'); continue
        img = Image.open(p).convert('RGB')
        ea = arcface_embed(app, img)
        if ea is None:
            print(f'skip (no face): {name}'); continue
        with torch.no_grad():
            ef = facenet_embed(net, pil_to_pixel_tensor(img))
        sa = float(cosine_sim(emb_a_clean, ea))
        sf = float(torch.nn.functional.cosine_similarity(emb_f_clean, ef).mean())
        rows.append({'name': name, 'arcface_sim': sa, 'facenet_sim': sf})
        print(f'{name:26s} ArcFace={sa:+.4f}   facenet={sf:+.4f}')

    def _stats(subset):
        aa = np.array([r['arcface_sim'] for r in subset])
        ff = np.array([r['facenet_sim'] for r in subset])
        pr = stats.pearsonr(ff, aa); sr = stats.spearmanr(ff, aa)
        return {'n': len(subset),
                'pearson_r': float(pr[0]), 'pearson_p': float(pr[1]),
                'spearman_rho': float(sr[0]), 'spearman_p': float(sr[1])}

    # The clean anchor sits at (1.0, 1.0) by construction and is a perfect-
    # agreement point, so including it inflates the correlation. The paper
    # reports the perturbed-only figure; both are stored here.
    all_rows = _stats(rows)
    perturbed = _stats([r for r in rows if r['name'] != 'clean'])
    print(f"\nall rows (incl. clean anchor): n = {all_rows['n']}  "
          f"r = {all_rows['pearson_r']:.4f}  rho = {all_rows['spearman_rho']:.4f}")
    print(f"perturbed only (PAPER-REPORTED): n = {perturbed['n']}  "
          f"r = {perturbed['pearson_r']:.4f}  rho = {perturbed['spearman_rho']:.4f}")

    with open(os.path.join(R, 'surrogate_transfer_validation.json'), 'w') as fh:
        json.dump({'rows': rows,
                   'paper_reported': 'perturbed_only',
                   'perturbed_only': perturbed,
                   'all_rows_incl_clean_anchor': all_rows,
                   'n': perturbed['n'],
                   'pearson_r': perturbed['pearson_r'], 'pearson_p': perturbed['pearson_p'],
                   'spearman_rho': perturbed['spearman_rho'], 'spearman_p': perturbed['spearman_p']},
                  fh, indent=2)
    print('Saved surrogate_transfer_validation.json')


if __name__ == '__main__':
    main()
