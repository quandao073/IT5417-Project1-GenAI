"""B14 diagnosis (CPU): why FT1 lr 3e-5 sits at loss ~ ln 128 and val P@5 below the smoke run.
Same captions, same images (eval transform), base vs adapter epoch 1 / 2 of runs/ft1_lr3e-5."""
import math
from pathlib import Path

import numpy as np, polars as pl, torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn.functional import normalize

from src.common.config import load_config
from src.data import probe_pool
from src.retrieval.templates import training_concepts
from training import captions, data, lora
from training.model import load_base, resolve_config_dir

torch.set_num_threads(16)
torch.manual_seed(0)
cfg, cand = load_config("training"), load_config("models")["image_encoder"]["candidates"]["cxrclip"]
concepts, concepts_cfg = training_concepts(cfg), load_config("concepts")
images, index, _ = data.load_cache(cfg["image"]["cache_dir"])
resolved = pl.read_parquet(Path("data/probe_pool") / probe_pool.POOL_LABELS)
statuses = probe_pool.resolved_status_matrix(resolved, index["study_id"].to_list())
sampler = captions.CaptionSampler(statuses, concepts, concepts_cfg, mix=cfg["captions"]["mix"], seed=7)
ds = data.PoolDataset(cfg["image"]["cache_dir"], sampler, train=True, augment_cfg=None)
rng = np.random.default_rng(0)
B, NB = 128, 4
rows = rng.choice(ds.rows, B * NB, replace=False)
full = np.stack([captions.label_vector(statuses[r], concepts) for r in rows])   # image's own labels
items = [sampler.sample(int(r)) for r in rows]
texts = [t for t, _ in items]
y = torch.from_numpy(np.stack([v for _, v in items])).float()
pix = torch.stack([data.eval_transform()(images[int(r)]) for r in rows])
print("examples:", [(texts[k], y[k].nonzero().flatten().tolist()) for k in range(4)])


def targets(yb):
    K = yb @ yb.T
    K = K - torch.diag(K.diag()) + torch.diag(K.diag().clamp(min=1))
    T = K.clamp(min=0)
    return K, T / T.sum(1, keepdim=True)


# Model-independent: what the caption-only targets call negative although the image matches.
fn_neg, fn_conf, n_neg, n_conf, floor = 0, 0, 0, 0, []
for b in range(NB):
    sl = slice(b * B, (b + 1) * B)
    K, T = targets(y[sl])
    yb, fb = y[sl].numpy(), full[sl]
    # caption j truly describes image i: every affirmed concept PRESENT, every negated one not PRESENT
    rel = np.array([[np.all(fb[i][yb[j] > 0] == 1) and np.all(fb[i][yb[j] < 0] != 1)
                     for j in range(B)] for i in range(B)])
    neg = (T == 0).numpy() & ~(K < 0).numpy()
    conf = (K < 0).numpy()
    fn_neg += (rel & neg).sum(); n_neg += neg.sum()
    fn_conf += (rel & conf).sum(); n_conf += conf.sum()
    Tn = T.numpy()
    floor.append(float(-(np.where(Tn > 0, Tn * np.log(np.where(Tn > 0, Tn, 1)), 0)).sum(1).mean()))
print(f"targets: attract floor (mean entropy of T) {np.mean(floor):.3f}, uniform ln{B} {math.log(B):.3f}")
print(f"targets: pairs treated as plain negatives {n_neg}, of which the image truly matches the caption "
      f"{fn_neg / n_neg:.1%}; conflict pairs {n_conf}, of which truly matching {fn_conf / max(n_conf, 1):.1%}")

lc = cfg["lora"]
tok = None
for name, adapter in [("base", None), ("epoch1", "runs/ft1_lr3e-5/adapter_epoch1"),
                      ("epoch2", "runs/ft1_lr3e-5/adapter_epoch2")]:
    module, tok, _ = load_base(cand["weights"], resolve_config_dir(cand["swin_config"]),
                               resolve_config_dir(cand["text_config"]))
    pm = lora.apply_lora(module, rank=lc["rank"], alpha=lc["alpha"], dropout=lc["dropout"], targets=lc["targets"])
    if adapter:
        set_peft_model_state_dict(pm, load_file(Path(adapter) / "adapter_model.safetensors"))
    net = lora.unwrap(pm).eval()
    scale = float(net.logit_scale.exp().clamp(max=100.0))
    with torch.inference_mode():
        zi = torch.cat([net.encode_image(pix[k:k + 64]) for k in range(0, len(pix), 64)])
        tk = tok(texts, padding="longest", truncation=True, max_length=64, return_tensors="pt")
        zt = net.encode_text(tk["input_ids"], tk["attention_mask"])
    zi, zt = normalize(zi.float(), dim=-1), normalize(zt.float(), dim=-1)
    att, rep, cpos, cneg, cconf, cdiag, rstd = [], [], [], [], [], [], []
    for b in range(NB):
        sl = slice(b * B, (b + 1) * B)
        K, T = targets(y[sl])
        cos = zi[sl] @ zt[sl].T
        logits = scale * cos
        att.append(float(-0.5 * ((T * logits.log_softmax(1)).sum(1).mean() + (T * logits.T.log_softmax(1)).sum(1).mean())))
        conf = K < 0
        rep.append(float(logits[conf].clamp(min=0).mean()) if conf.any() else 0.0)
        pos = (T > 0) & ~torch.eye(B, dtype=bool)
        cpos.append(float(cos[pos].mean())); cneg.append(float(cos[(T == 0) & ~conf].mean()))
        cconf.append(float(cos[conf].mean())); cdiag.append(float(cos.diag().mean()))
        rstd.append(float(logits.std(1).mean()))
    ii = zi @ zi.T; tt = zt @ zt.T
    off = lambda m: float(m[~torch.eye(len(m), dtype=bool)].mean())
    print(f"{name:6s} scale {scale:6.2f} | attract {np.mean(att):.3f} repel {np.mean(rep):.3f} "
          f"total {np.mean(att) + 0.3 * np.mean(rep):.3f} | cos diag {np.mean(cdiag):.3f} pos {np.mean(cpos):.3f} "
          f"neg {np.mean(cneg):.3f} conflict {np.mean(cconf):.3f} | logit std/row {np.mean(rstd):.2f} | "
          f"img-img {off(ii):.3f} txt-txt {off(tt):.3f}", flush=True)
