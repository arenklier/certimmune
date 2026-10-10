import os, glob, json, collections
from PIL import Image
from common import load_face_app, arcface_embed
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'metacloak_identities')
app = load_face_app()
res = {}
for d in sorted(glob.glob(os.path.join(D, 'identity_*'))):
    name = os.path.basename(d)
    imgs = sorted(glob.glob(os.path.join(d, '*')))
    ok = 0
    for p in imgs:
        try:
            e = arcface_embed(app, Image.open(p).convert('RGB'))
        except Exception:
            e = None
        if e is not None: ok += 1
    res[name] = {'n_images': len(imgs), 'n_detected': ok}
    print('%-14s %d/%d' % (name, ok, len(imgs)), flush=True)
c = collections.Counter(v['n_detected'] for v in res.values())
print()
print('identities total:', len(res))
for k in sorted(c, reverse=True): print('  exactly %d/4 detected: %d identities' % (k, c[k]))
for thr in (4,3,2,1):
    n = sum(1 for v in res.values() if v['n_detected'] >= thr)
    print('  >= %d detected: %d identities' % (thr, n))
json.dump(res, open('../results/metacloak_relaxed_screen.json','w'), indent=2)
print('saved metacloak_relaxed_screen.json')
