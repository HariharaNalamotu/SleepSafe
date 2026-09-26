"""Train the Stage-2 BiGRU head on cached Stage-1 features (all data resident on GPU).

    python scripts/train.py --name v1

Outputs runs/<name>/: model.pt (preprocessor + head + post-processing config), metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleepsafe.labels import TARGETS  # noqa: E402
from sleepsafe.model import WINDOW_S, Preprocessor, SleepHead, predict_session  # noqa: E402
from sleepsafe.postprocess import PostConfig, detect_events, runs  # noqa: E402

FEAT = ROOT / "data" / "features"
TRAIN_VARIANTS = [0, 1, 2, 3, 4, 5]
EVAL_VARIANTS = [0, 1, 2, 3]          # clean mic/tracheal + lapel-augmented mic/tracheal
LOSS_W = torch.tensor([1.0, 1.0, 0.5, 0.5])
DEV = "cuda"  # set from --device in main()


def load_raw(s: str, v: int, dev=None):
    d = FEAT / s
    dev = dev or DEV
    return (torch.from_numpy(np.load(d / f"emb_v{v}.npy")).to(dev),
            torch.from_numpy(np.load(d / f"scores_v{v}.npy")).to(dev),
            torch.from_numpy(np.load(d / f"energy_v{v}.npy")).to(dev))


def split_subjects(subjects: list[str], seed: int = 0):
    ahi = {s: float(np.load(FEAT / s / "labels.npz")["ahi"]) for s in subjects}
    order = sorted(subjects, key=lambda s: (np.nan_to_num(ahi[s]), s))
    test = [s for i, s in enumerate(order) if i % 8 == 0]
    val = [s for i, s in enumerate(order) if i % 8 == 4]
    train = [s for s in order if s not in test and s not in val]
    return train, val, test, ahi


class Tracks:
    """Concatenated (subject, variant) tracks on GPU with offsets for crop sampling."""

    def __init__(self, pre: Preprocessor, subjects, variants):
        xs, ys, ms, self.meta = [], [], [], []
        off = 0
        for s in subjects:
            lab = np.load(FEAT / s / "labels.npz")
            y = torch.from_numpy(lab["y"]).to(DEV).half()
            m = torch.from_numpy(lab["mask"]).to(DEV)
            for v in variants:
                x = pre(*load_raw(s, v)).half()
                T = min(len(x), len(y))
                xs.append(x[:T]); ys.append(y[:T]); ms.append(m[:T])
                self.meta.append({"subject": s, "variant": v, "offset": off, "length": T})
                off += T
        self.x, self.y, self.m = torch.cat(xs), torch.cat(ys), torch.cat(ms)
        self.offsets = torch.tensor([t["offset"] for t in self.meta], device=DEV)
        self.lengths = torch.tensor([t["length"] for t in self.meta], device=DEV)

    def track(self, i):
        t = self.meta[i]
        sl = slice(t["offset"], t["offset"] + t["length"])
        return self.x[sl], self.y[sl], self.m[sl]

    def sample(self, B: int, L: int, g: torch.Generator):
        w = (self.lengths - L).clamp_min(0).float()
        ti = torch.multinomial(w, B, replacement=True, generator=g)
        start = (torch.rand(B, device=DEV, generator=g) * (self.lengths[ti] - L + 1)).long()
        idx = (self.offsets[ti] + start).unsqueeze(1) + torch.arange(L, device=DEV)
        return self.x[idx], self.y[idx], self.m[idx]


def average_precision(p: torch.Tensor, y: torch.Tensor) -> float:
    if y.sum() == 0:
        return float("nan")
    o = p.argsort(descending=True)
    y = y[o].float()
    prec = y.cumsum(0) / torch.arange(1, len(y) + 1, device=y.device)
    return (prec * y).sum().item() / y.sum().item()


def match_events(pred: list[tuple[int, int]], true: list[tuple[float, float]]) -> tuple[int, int, int]:
    """One-to-one greedy matching on any overlap. Returns (tp, n_pred, n_true)."""
    used = set()
    tp = 0
    for a, b in pred:
        for j, (s, e) in enumerate(true):
            if j not in used and a < e and s < b:
                used.add(j)
                tp += 1
                break
    return tp, len(pred), len(true)


def f1(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def true_resp_events(s: str, T: int):
    ev = np.load(FEAT / s / "labels.npz")["events"]
    return [(a, a + d) for a, d, k in ev if k in (0, 1) and a < T]


def evaluate(head, tracks: Tracks, cfg: PostConfig | None = None, block_s: int | None = None):
    """Per-second AP per target + (if cfg) event-level F1 and AHI agreement.

    block_s: evaluate on independent sessions of this length (e.g. 1800 for 30-min sessions);
    features are re-preprocessed per block by the caller-provided tracks, so this only affects
    how predictions are windowed.
    """
    all_p, all_y, all_m = [], [], []
    per_track = []
    for i, t in enumerate(tracks.meta):
        x, y, m = tracks.track(i)
        if block_s:
            ps = [predict_session(head, x[a: a + block_s]) for a in range(0, len(x) - block_s + 1, block_s)]
            n = len(ps) * block_s
            if not ps:
                continue
            p = torch.cat(ps)
            x, y, m = x[:n], y[:n], m[:n]
        else:
            p = predict_session(head, x)
        all_p.append(p); all_y.append(y); all_m.append(m)
        per_track.append((t, p.cpu().numpy(), y.cpu().numpy()))
    P, Y, M = torch.cat(all_p), torch.cat(all_y), torch.cat(all_m)
    out = {f"ap_{n}": average_precision(P[M[:, k], k], Y[M[:, k], k]) for k, n in enumerate(TARGETS)}
    # zero-shot baseline for snoring: the AudioSet "snoring" logit column of the model input
    from sleepsafe.postprocess import SCORE_NAMES
    col = tracks.x.shape[1] - 36 + SCORE_NAMES.index("snoring")
    if not block_s:  # full tracks: rows line up with P / Y / M
        X = torch.cat([tracks.track(i)[0][:, col] for i in range(len(tracks.meta))])
        out["ap_snore_zeroshot"] = average_precision(X[M[:, 2]].float(), Y[M[:, 2], 2])
    asleep_pred = P[M[:, 3], 3] >= 0.5
    out["acc_asleep"] = (asleep_pred == (Y[M[:, 3], 3] > 0)).float().mean().item()
    if cfg is None:
        return out, per_track

    tp = npred = ntrue = 0
    ahi_true, ahi_pred = [], []
    for t, p, y in per_track:
        T = len(p)
        if block_s:
            sess = [(a, a + block_s) for a in range(0, T, block_s)]
        else:
            sess = [(0, T)]
        tru_all = true_resp_events(t["subject"], T)
        for a, b in sess:
            ev = detect_events(p[a:b], cfg)
            pred = [(e["start_offset_s"] + a, e["end_offset_s"] + a) for e in ev if e["type"] in ("apnea", "hypopnea")]
            tru = [(s, e) for s, e in tru_all if a <= s < b]
            r = match_events(pred, tru)
            tp += r[0]; npred += r[1]; ntrue += r[2]
            tst_true = y[a:b, 3].sum() / 3600
            tst_pred = (p[a:b, 3] >= cfg.thr_asleep).sum() / 3600
            if tst_true > 0.25 and not block_s:
                ahi_true.append(len(tru) / tst_true)
                ahi_pred.append(len(pred) / max(tst_pred, 0.25))
    out["event_f1"] = f1(tp, npred, ntrue)
    out["event_precision"] = tp / npred if npred else 0.0
    out["event_recall"] = tp / ntrue if ntrue else 0.0
    if ahi_true:
        at, ap_ = np.array(ahi_true), np.array(ahi_pred)
        band = lambda a: np.digitize(a, [5, 15, 30])  # noqa: E731
        out["ahi_pearson"] = float(np.corrcoef(at, ap_)[0, 1]) if len(at) > 2 else float("nan")
        out["ahi_mae"] = float(np.abs(at - ap_).mean())
        out["severity_acc"] = float((band(at) == band(ap_)).mean())
        out["severity_within_one"] = float((np.abs(band(at) - band(ap_)) <= 1).mean())
    return out, per_track


def tune_thresholds(per_track) -> PostConfig:
    """Grid-search apnea/hypopnea/snore thresholds on validation predictions."""
    best = (-1.0, None)
    grid = np.round(np.arange(0.2, 0.85, 0.05), 2)
    for ta in grid:
        for th in grid:
            cfg = PostConfig(thr_apnea=float(ta), thr_hypopnea=float(th))
            tp = npred = ntrue = 0
            for t, p, y in per_track:
                ev = detect_events(p, cfg)
                pred = [(e["start_offset_s"], e["end_offset_s"]) for e in ev if e["type"] in ("apnea", "hypopnea")]
                r = match_events(pred, true_resp_events(t["subject"], len(p)))
                tp += r[0]; npred += r[1]; ntrue += r[2]
            score = f1(tp, npred, ntrue)
            if score > best[0]:
                best = (score, cfg)
    cfg = best[1]
    # snore: per-second F1
    P = np.concatenate([p[:, 2] for _, p, _ in per_track])
    Y = np.concatenate([y[:, 2] for _, _, y in per_track]) > 0
    cfg.thr_snore = float(max(grid, key=lambda t: f1(((P >= t) & Y).sum(), (P >= t).sum(), Y.sum())))
    return cfg


def calibrate_zero_shot(cfg: PostConfig, subjects: list[str], quantile: float = 0.999) -> None:
    """Zero-shot AudioSet flags have no labels: set each class threshold so it fires on only the
    top (1 - quantile) of seconds across clean + lapel-augmented recordings (never below the default)."""
    from sleepsafe.postprocess import SCORE_NAMES
    sc = np.concatenate([np.load(FEAT / s / f"scores_v{v}.npy").astype(np.float32)
                         for s in subjects for v in EVAL_VARIANTS])
    for kind, (cls, thr, min_len) in list(cfg.zero_shot.items()):
        q = float(np.quantile(sc[:, SCORE_NAMES.index(cls)], quantile))
        cfg.zero_shot[kind] = (cls, round(max(thr, q), 3), min_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--patience", type=int, default=4, help="stop after this many evals without improvement")
    ap.add_argument("--sleep-ms", type=float, default=0.0, help="pause after each step to cap GPU duty cycle")
    ap.add_argument("--finalize-from", default=None,
                    help="skip training: load preprocessor + head from this checkpoint, then tune and test")
    args = ap.parse_args()

    global DEV
    DEV = args.device
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(p.name for p in FEAT.iterdir() if (p / "labels.npz").exists())
    train_s, val_s, test_s, ahi = split_subjects(subjects)
    print(f"subjects: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test", flush=True)

    if args.finalize_from:
        ck = torch.load(args.finalize_from, map_location=DEV, weights_only=False)
        pre = Preprocessor(pca_dim=ck["pca_dim"]).to(DEV)
        pre.load_state_dict(ck["preprocessor"])
        head = SleepHead(pre.dim, **ck["head_args"]).to(DEV)
        head.load_state_dict(ck["head"])
        val = Tracks(pre, val_s, EVAL_VARIANTS)
        print(f"finalizing {args.finalize_from} (no training)", flush=True)
        finalize(args, pre, head, val, train_s, val_s, test_s, ahi, [], out_dir, ck["pca_dim"], ck["head_args"])
        return

    # --- fit preprocessing on training data ---
    t0 = time.time()
    pre = Preprocessor(pca_dim=args.pca).to(DEV)
    emb_sample = torch.cat([load_raw(s, v)[0][::10] for s in train_s for v in (0, 1)])
    var = pre.fit_pca(emb_sample)
    raw_sample = torch.cat([pre.raw(*[a[::1] for a in load_raw(s, v)])[::10] for s in train_s for v in (0, 1, 2, 3)])
    pre.fit_norm(raw_sample)
    del emb_sample, raw_sample
    print(f"PCA {args.pca} dims keeps {var:.1%} variance; preprocessing fit in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    train = Tracks(pre, train_s, TRAIN_VARIANTS)
    val = Tracks(pre, val_s, EVAL_VARIANTS)
    print(f"loaded {train.x.shape[0] / 3600:.0f}h train tracks, {val.x.shape[0] / 3600:.0f}h val tracks "
          f"in {time.time() - t0:.0f}s on {DEV}", flush=True)

    # --- class balance ---
    m = train.m.float()
    pos = (train.y.float() * m).sum(0)
    neg = ((1 - train.y.float()) * m).sum(0)
    pos_weight = (neg / pos.clamp_min(1)).sqrt().clamp(1, 10)
    print("positive rate:", {n: round((pos[k] / m[:, k].sum()).item(), 4) for k, n in enumerate(TARGETS)},
          "pos_weight:", pos_weight.tolist(), flush=True)

    head = SleepHead(pre.dim, hidden=args.hidden, dropout=args.dropout).to(DEV)
    print(f"head params: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    g = torch.Generator(device=DEV).manual_seed(args.seed)
    lw = LOSS_W.to(DEV)

    best = (-1.0, None)
    bad_evals = 0
    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        head.train()
        x, y, mk = train.sample(args.batch, WINDOW_S, g)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits = head(x.float())
        loss_el = F.binary_cross_entropy_with_logits(logits.float(), y.float(), pos_weight=pos_weight, reduction="none")
        mk = mk.float()
        loss = ((loss_el * mk).sum((0, 1)) / mk.sum((0, 1)).clamp_min(1) * lw).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if args.sleep_ms:
            if DEV == "cuda":
                torch.cuda.synchronize()
            time.sleep(args.sleep_ms / 1000)
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} ({(time.time() - t0) / step * 1000:.0f} ms/step)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics, _ = evaluate(head, val)
            score = np.nanmean([metrics["ap_apnea"], metrics["ap_hypopnea"]])
            history.append({"step": step, **metrics})
            print(f"  val @ {step}: " + " ".join(f"{k} {v:.3f}" for k, v in metrics.items()), flush=True)
            if score > best[0]:
                best = (score, {k: v.detach().clone() for k, v in head.state_dict().items()})
                bad_evals = 0
                # crash insurance: a usable model (default thresholds) after every improvement
                torch.save({"preprocessor": pre.state_dict(), "pca_dim": args.pca, "head": best[1],
                            "head_args": {"hidden": args.hidden, "dropout": args.dropout},
                            "post_config": PostConfig().__dict__, "targets": TARGETS, "partial": True},
                           out_dir / "model_partial.pt")
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"early stop at step {step}", flush=True)
                    break
    print(f"training done in {(time.time() - t0) / 60:.1f} min; best val resp AP {best[0]:.3f}", flush=True)
    head.load_state_dict(best[1])
    del train
    finalize(args, pre, head, val, train_s, val_s, test_s, ahi, history, out_dir,
             args.pca, {"hidden": args.hidden, "dropout": args.dropout})


def finalize(args, pre, head, val, train_s, val_s, test_s, ahi, history, out_dir, pca_dim, head_args):
    """Tune post-processing on validation, evaluate on test, save model.pt + metrics.json."""
    _, val_tracks = evaluate(head, val)
    cfg = tune_thresholds(val_tracks)
    calibrate_zero_shot(cfg, val_s + train_s[::4])
    print("post-processing config:", cfg.__dict__, flush=True)
    val_metrics, _ = evaluate(head, val, cfg)
    print("val (tuned):", json.dumps({k: round(v, 3) for k, v in val_metrics.items()}), flush=True)
    test = Tracks(pre, test_s, EVAL_VARIANTS)
    test_metrics = {}
    for v, name in [(0, "mic"), (1, "tracheal"), (2, "mic_lapel_aug"), (3, "tracheal_lapel_aug")]:
        sub = Tracks.__new__(Tracks)
        idx = [i for i, t in enumerate(test.meta) if t["variant"] == v]
        sub.x, sub.y, sub.m = test.x, test.y, test.m
        sub.meta = [test.meta[i] for i in idx]
        full, _ = evaluate(head, sub, cfg)
        short, _ = evaluate(head, sub, cfg, block_s=1800)
        test_metrics[name] = {"full_night": full, "30min_sessions": short}
        print(f"test {name:20s} full : " + json.dumps({k: round(x, 3) for k, x in full.items()}), flush=True)
        print(f"test {name:20s} 30min: " + json.dumps({k: round(x, 3) for k, x in short.items()}), flush=True)

    torch.save({
        "preprocessor": pre.state_dict(), "pca_dim": pca_dim,
        "head": head.state_dict(), "head_args": head_args,
        "post_config": cfg.__dict__, "targets": TARGETS,
    }, out_dir / "model.pt")
    json.dump({"args": vars(args), "split": {"train": train_s, "val": val_s, "test": test_s},
               "ahi": ahi, "history": history, "val": val_metrics, "test": test_metrics,
               "post_config": cfg.__dict__}, open(out_dir / "metrics.json", "w"), indent=1, default=float)
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
