"""
train.py — 本学習スクリプト（仕様 v7 §5 学習・§9.2 評価）
================================================================================
使い方:
  python train.py --data dataset_v7 --steps 30000 --batch 32          # 学習（自動再開）
  python train.py --data dataset_v7 --eval_only --split test          # §9.2 評価のみ

実装内容:
  - memmap データパイプライン（manifest の splits に従う）
  - Gumbel 温度アニーリング 1.0 → 0.3（指数、§5）
  - Adam + 勾配クリッピング（ADHSR の 1/a 項対策）
  - チェックポイント保存・自動再開、CSV ログ
  - 検証ループ: 損失内訳 + §9.2 指標
      2. アサイン復元 P/R/F1（マッチング後。FP=余計な配線 を個別報告）
      3. 深さ回帰誤差（真アサイン成分）
      4. ソース形状誤差（信号空間 e_k(n)、使用行のみ）
      5. 静的パラメータ誤差
      6. 音響距離 MRS
      8. 型正解率（低レート LFO 層 f·T<1 を層別）
      9. クラスタ層別（コア8 / クラスタ / 弱効き別の F1、クラスタ内誤帰属数）
注:
  - λI 版 PNP（M̄(θ) 事前計算後は estimator.losses の param 項を二次形式に差し替え）
  - OOM 時は --batch を下げる（§9.4: scan の BPTT がメモリ律速）
"""
import argparse, csv, json, os, time

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import estimator as E
import weights_io as W

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(0)

K, M = ds.K_ENVS, ds.M_DEST
CORE8 = [0, 1, 2, 3, 4, 7, 8, 10]
CLUSTER = [10, 11, 15, 16]          # thresh, ratio, drive_d, mix_d（thresh は二重役割）
WEAK = [5, 6, 9, 12, 13, 14]
LAM = dict(spec=1.0, param=0.2, assign=3.0, sparse=0.0, none_w=0.25)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--temp_hi", type=float, default=1.0)
    ap.add_argument("--temp_lo", type=float, default=0.3)
    ap.add_argument("--ckpt_dir", type=str, default="ckpt_v7")
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--arch", type=str, default="v2", choices=["v1","v2"])
    ap.add_argument("--spec_warm0", type=int, default=5000,
                    help="このステップまで λ_spec=0（教師あり項で離散構造を先に学習）")
    ap.add_argument("--spec_warmlen", type=int, default=5000,
                    help="warm0 以降、この長さで λ_spec を 0→設定値へ線形に上げる")
    ap.add_argument("--mdiag", type=str, default="",
                    help="Mdiag.npz のパス。指定すると param 損失が知覚感度で重み付けされる")
    ap.add_argument("--mdiag_pow", type=float, default=1.0,
                    help="重みの強さ w = d^pow（0=等方, 1=そのまま, 0.5=緩やか）")
    ap.add_argument("--param_w", type=float, default=-1.0,
                    help="λ_param を上書き（-1 で LAM の既定値 0.2 を使用）")
    ap.add_argument("--spec_w", type=float, default=-1.0,
                    help="λ_spec を上書き（-1 で LAM の既定値 1.0 を使用）")
    ap.add_argument("--none_w", type=float, default=0.25,
                    help="ゲート CE における ∅ クラスの重み（1.0=従来）")
    ap.add_argument("--depth_tf", action="store_true",
                    help="深さを真の割り当て位置で教える（ゲートの外れで深さが 0 に引かれるのを防ぐ）")
    ap.add_argument("--depth_none0", action="store_true",
                    help="真が ∅ の列で深さを 0 に教える（--depth_tf と併用）")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------- データ
class Data:
    def __init__(self, root):
        self.man = json.load(open(os.path.join(root, "manifest.json")))
        n, N = self.man["n_total"], self.man["N"]
        self.wav = np.memmap(os.path.join(root, "waveforms.int16.bin"),
                             dtype=np.int16, mode="r", shape=(n, N))
        self.P = dict(np.load(os.path.join(root, "params.npz")))
        self.scale = self.man["int16_scale"]
        self.splits = {k: np.arange(v[0], v[1]) for k, v in self.man["splits"].items()}
        # --- v8.2: 正準ソース順（学習教師の安定化）。ENV 先頭 attack 昇順 → LFO rate 昇順 ---
        styp, sp = self.P["src_type"], self.P["src_params"]
        z0 = self.P["z"].astype(np.int64)
        usedm = np.stack([(z0 == k).any(axis=1) for k in range(styp.shape[1])], 1)
        # 使用中(=音に現れる)ソースを先頭へ。未使用は末尾（番号が音に依存しなくなる）
        key = (np.where(usedm, 0.0, 1e4)
               + np.where(styp == 1.0, sp[..., 0], 100.0 + sp[..., 0]))
        perm = np.argsort(key, axis=1, kind="stable")             # [n,K]
        ar = np.arange(len(key))[:, None]
        for k_ in ("src_params", "src_type", "R"):
            self.P[k_] = self.P[k_][ar, perm]
        inv = np.argsort(perm, axis=1).astype(np.int64)           # 旧→新
        z = self.P["z"].astype(np.int64)
        self.P["z"] = np.where(
            z >= 0, np.take_along_axis(inv, np.maximum(z, 0), axis=1), -1
        ).astype(np.int8)

    def batch(self, idx):
        x = tf.constant(self.wav[idx].astype(np.float32) / self.scale)
        b = {}
        for k in ("theta_stat", "theta_osc", "gmk", "src_params",
                  "src_type", "R", "f0"):
            b[k] = tf.constant(self.P[k][idx].astype(np.float32))
        b["z"] = tf.constant(self.P["z"][idx].astype(np.int32))
        return x, b


# ---------------------------------------------------------------- §9.2 指標
def metrics_np(pred, batch, mapping, sr, t_off, N):
    """マッチング後の §9.2 指標（numpy 集計）"""
    inv = tf.argsort(mapping, axis=1)
    z = batch["z"].numpy()
    z_re = np.where(z >= 0,
                    tf.gather(inv, tf.maximum(batch["z"], 0), axis=1,
                              batch_dims=1).numpy(), K)
    g_pred = tf.argmax(pred["gate_logits"], -1, output_type=tf.int32).numpy()
    B = z.shape[0]

    pred_on = g_pred < K
    true_on = z_re < K
    tp = (pred_on & true_on & (g_pred == z_re))
    fp = pred_on & ~tp                      # 余計/誤配線（誤帰属は FP かつ FN）
    fn = true_on & ~tp

    def prf(sel):
        TP, FP, FN = tp[:, sel].sum(), fp[:, sel].sum(), fn[:, sel].sum()
        p = TP / max(TP + FP, 1)
        r = TP / max(TP + FN, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        return p, r, f

    out = {}
    out["P"], out["R"], out["F1"] = prf(slice(None))
    # --- v8.4: 「検出」と「ソース同定」の分離診断 ---
    # (a) 検出: どの列が変調されているか（ソース識別は問わない二値問題）
    dTP = (pred_on & true_on).sum(); dFP = (pred_on & ~true_on).sum()
    dFN = (~pred_on & true_on).sum()
    dp = dTP / max(dTP + dFP, 1); dr = dTP / max(dTP + dFN, 1)
    out["F1_detect"] = 2 * dp * dr / max(dp + dr, 1e-9)
    out["detect_P"], out["detect_R"] = dp, dr
    # (b) 同定: 検出が当たった列で、ソース番号まで合っている割合
    both = pred_on & true_on
    out["srcid_acc"] = float((g_pred[both] == z_re[both]).mean()) if both.any() else np.nan
    # 参考: 予測アサイン本数 / サンプル
    out["n_pred"] = float(pred_on.sum(1).mean()); out["n_true"] = float(true_on.sum(1).mean())
    out["FP_rate"] = fp.sum() / max(B * M, 1)
    for name, cols in (("core8", CORE8), ("cluster", CLUSTER), ("weak", WEAK)):
        out[f"F1_{name}"] = prf(cols)[2]

    # クラスタ内誤帰属: 真がクラスタ列 m の source k を、pred が別のクラスタ列 m' に配線
    R_hat = E.gather_rows(pred["R"], mapping).numpy()
    mis = 0
    for b in range(B):
        for m in CLUSTER:
            k = z_re[b, m]
            if k < K and g_pred[b, m] != k:
                for m2 in CLUSTER:
                    if m2 != m and g_pred[b, m2] == k:
                        mis += 1
                        break
    out["cluster_misattr"] = mis / max(B, 1)

    # 深さ回帰誤差（真アサイン成分の R̂ 値）
    errs = []
    Rt = batch["R"].numpy()
    for b in range(B):
        for m in range(M):
            k = z_re[b, m]
            if k < K:
                errs.append(abs(R_hat[b, k, m] - Rt[b, :, m].sum()))
    out["depth_mae"] = float(np.mean(errs)) if errs else np.nan

    # ソース形状誤差（信号空間、使用行のみ）
    used = (z_re[:, :, None] == np.arange(K)[None, None, :]).any(axis=1)  # [B,K]
    e_true = ds.sources(batch["src_params"], batch["src_type"],
                        t_off, N, sr).numpy()
    sp = E.gather_rows(pred["src_params"], mapping)
    st = E.gather_rows(pred["type_oh"][..., 0], mapping)
    e_pred = ds.sources(sp, st, t_off, N, sr).numpy()
    diff = np.abs(e_true - e_pred).mean(-1)                               # [B,K]
    out["src_mae"] = float(diff[used].mean()) if used.any() else np.nan

    # 静的パラメータ誤差
    out["stat_mae"] = float(np.abs(pred["theta_stat"].numpy()
                                   - batch["theta_stat"].numpy()).mean())
    out["gmk_mae"] = float(np.abs(pred["gmk"].numpy() - batch["gmk"].numpy()).mean())

    # 型正解率（低レート層別）
    t_lab = (1.0 - batch["src_type"].numpy()).astype(int)
    t_pred = tf.argmax(E.gather_rows(pred["type_logits"], mapping), -1,
                       output_type=tf.int32).numpy()
    out["type_acc"] = float((t_pred == t_lab).mean())

    # ================= v8.6: 厳密な復元指標（主張C を支える数字） =================
    # 「配線が当たった」だけでは生成過程の説明にならない。列単位で
    #   (1) ソース番号  (2) そのソースの型 ENV/LFO  (3) 深さ  (4) ソース形状
    # がすべて揃って初めて「その列の変調を復元した」とみなす。
    R_true_col = Rt.sum(axis=1)                                   # [B,M] 列の真の深さ
    R_hat_col = R_hat.sum(axis=1)                                 # [B,M] 予測（列スパース）
    e_true_all = ds.sources(batch["src_params"], batch["src_type"],
                            t_off, N, sr).numpy()                 # [B,K,N]
    sp_m = E.gather_rows(pred["src_params"], mapping)
    st_m = E.gather_rows(pred["type_oh"][..., 0], mapping)
    e_pred_all = ds.sources(sp_m, st_m, t_off, N, sr).numpy()
    shape_err_k = np.abs(e_true_all - e_pred_all).mean(-1)        # [B,K]

    DEPTH_TOL, SHAPE_TOL = 0.1, 0.1

    # ===== v8.7 水準1: 変調記述の復元（ソース番号を問わない = 主張C の本体） =====
    # ソース番号は正準順序という人為的規約が与えたラベルであり物理的意味を持たない。
    # 意味を持つのは「どの型・どの形状の変調信号が、どの深さで、どの変調先に刺さるか」。
    # → 検出が当たった列について、予測ソース(番号は任意)の型・形状・深さが真と一致するかを見る。
    desc_hit, desc_grp = [], {"core8": [], "cluster": [], "weak": []}
    for b in range(B):
        for m in range(M):
            k_t = z_re[b, m]
            k_p = g_pred[b, m]
            if k_t >= K or k_p >= K:              # 真が∅ or 予測が∅ → 記述の復元対象外
                continue
            ok_type = bool(t_pred[b, k_p] == t_lab[b, k_t])
            ok_depth = bool(abs(R_hat_col[b, m] - R_true_col[b, m]) <= DEPTH_TOL)
            # 形状は「予測ソース k_p の信号」と「真ソース k_t の信号」を直接比較
            ok_shape = bool(np.abs(e_pred_all[b, k_p] - e_true_all[b, k_t]).mean()
                            <= SHAPE_TOL)
            good = ok_type and ok_depth and ok_shape
            desc_hit.append(good)
            for g, cols in (("core8", CORE8), ("cluster", CLUSTER), ("weak", WEAK)):
                if m in cols:
                    desc_grp[g].append(good)
    out["desc_rec_rate"] = float(np.sum(desc_hit)) / max(int(true_on.sum()), 1)
    out["desc_rec_on_det"] = float(np.mean(desc_hit)) if desc_hit else np.nan
    for g in ("core8", "cluster", "weak"):
        ng = int(true_on[:, {"core8": CORE8, "cluster": CLUSTER, "weak": WEAK}[g]].sum())
        out[f"desc_rec_{g}"] = float(np.sum(desc_grp[g])) / max(ng, 1)

    # ===== v8.7 水準2: 共動構造の復元（番号ラベルに依存しない構造の一致） =====
    # 「同じソースを共有する変調先ペア」の集合が真と一致するか（ペア単位 P/R/F1）。
    co_tp = co_fp = co_fn = 0
    for b in range(B):
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                t_share = (z_re[b, m1] < K) and (z_re[b, m1] == z_re[b, m2])
                p_share = (g_pred[b, m1] < K) and (g_pred[b, m1] == g_pred[b, m2])
                co_tp += t_share and p_share
                co_fp += p_share and not t_share
                co_fn += t_share and not p_share
    cp = co_tp / max(co_tp + co_fp, 1); cr = co_tp / max(co_tp + co_fn, 1)
    out["comove_P"], out["comove_R"] = cp, cr
    out["comove_F1"] = 2 * cp * cr / max(cp + cr, 1e-9)
    cols_srcid, cols_type, cols_depth, cols_shape, cols_full = [], [], [], [], []
    grp = {"core8": [], "cluster": [], "weak": []}
    for b in range(B):
        for m in range(M):
            k = z_re[b, m]
            if k >= K or g_pred[b, m] != k:      # 検出＋ソース同定に成功した列のみ
                continue
            ok_type = bool(t_pred[b, k] == t_lab[b, k])
            ok_depth = bool(abs(R_hat_col[b, m] - R_true_col[b, m]) <= DEPTH_TOL)
            ok_shape = bool(shape_err_k[b, k] <= SHAPE_TOL)
            full = ok_type and ok_depth and ok_shape
            cols_srcid.append(1); cols_type.append(ok_type)
            cols_depth.append(ok_depth); cols_shape.append(ok_shape); cols_full.append(full)
            for g, cols in (("core8", CORE8), ("cluster", CLUSTER), ("weak", WEAK)):
                if m in cols:
                    grp[g].append(full)
    nsrc = max(len(cols_srcid), 1)
    out["typeacc_on_hit"] = float(np.mean(cols_type)) if cols_srcid else np.nan
    out["depth_ok_on_hit"] = float(np.mean(cols_depth)) if cols_srcid else np.nan
    out["shape_ok_on_hit"] = float(np.mean(cols_shape)) if cols_srcid else np.nan
    # 条件付き（同定成功列のうち完全復元）と、全真配線に対する絶対率
    out["full_rec_on_hit"] = float(np.mean(cols_full)) if cols_srcid else np.nan
    out["full_rec_rate"] = float(np.sum(cols_full)) / max(int(true_on.sum()), 1)
    for g in ("core8", "cluster", "weak"):
        n_true_g = int(true_on[:, {"core8": CORE8, "cluster": CLUSTER,
                                   "weak": WEAK}[g]].sum())
        out[f"full_rec_{g}"] = (float(np.sum(grp[g])) / max(n_true_g, 1))
    lo = (t_lab == 1) & (batch["src_params"].numpy()[..., 0] * (N / sr) < 1.0)
    if lo.any():
        out["type_acc_lowrate"] = float((t_pred == t_lab)[lo].mean())
    return out


# ---------------------------------------------------------------- メイン
def main():
    args = parse_args()
    np.random.seed(args.seed); tf.random.set_seed(args.seed)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    LAM["none_w"] = args.none_w
    LAM["depth_tf"] = float(args.depth_tf)
    LAM["depth_none0"] = float(args.depth_none0)
    if args.mdiag:
        _d = np.load(args.mdiag)["d_mean"].astype(np.float32)
        _d = np.maximum(_d, 1e-8) ** args.mdiag_pow
        _d = _d / _d.mean()                      # 平均1に正規化（λ_param の意味を保つ）
        LAM["mdiag"] = _d
        print(f"mdiag loaded: {args.mdiag} pow={args.mdiag_pow} "
              f"(min {_d.min():.3e} / max {_d.max():.3e})")
    if args.param_w >= 0: LAM["param"] = args.param_w
    if args.spec_w >= 0:  LAM["spec"] = args.spec_w
    print(f"LAM = {LAM}  spec_warm0={args.spec_warm0} warmlen={args.spec_warmlen}")
    data = Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    fmax = E.F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))

    model = E.Estimator(sr=SR, t_off=T_OFF, arch=args.arch)
    # v8.4: GRU 等の遅延構築レイヤをチェックポイント復元前にビルドしておく
    model(tf.zeros([1, N]), tf.constant([220.0], tf.float32), temp=1.0, hard=True)
    opt = tf.keras.optimizers.Adam(args.lr)
    step_var = tf.Variable(0, dtype=tf.int64)
    # 重みの正本は weights_io（.weights.h5）。tf.train.Checkpoint は opt と step の再開用
    ckpt = tf.train.Checkpoint(opt=opt, step=step_var)
    mgr = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=3)
    best_dir = os.path.join(args.ckpt_dir, "best")
    best_mgr = tf.train.CheckpointManager(ckpt, best_dir, max_to_keep=1)
    best_score = -1.0

    def save_to(m):
        W.save(model, W.ckpt_weights_path(m.save()))
        W.prune(m)

    if args.eval_only:
        W.load(model, W.latest(args.ckpt_dir))
    elif mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint).expect_partial()
        W.load(model, W.ckpt_weights_path(mgr.latest_checkpoint))
        print(f"resumed from {mgr.latest_checkpoint} (step {int(step_var)})")

    def temp_at(step):
        frac = min(step / max(args.steps, 1), 1.0)
        return float(args.temp_hi * (args.temp_lo / args.temp_hi) ** frac)

    def spec_lam_at(step):
        """v8.3: spec は初期にゲート勾配を CE の 7 倍で圧倒するため段階導入"""
        if step < args.spec_warm0:
            return 0.0
        return LAM["spec"] * min((step - args.spec_warm0) / max(args.spec_warmlen, 1), 1.0)

    @tf.function
    def train_step(x, batch, temp, spec_lam):
        lam_now = dict(LAM); lam_now["spec"] = spec_lam
        with tf.GradientTape() as tape:
            pred = model(x, batch["f0"], temp=temp, hard=True)
            total, parts, _ = E.losses(pred, batch, x, SR, T_OFF, NH, lam_now,
                                       matching=False)
        grads = tape.gradient(total, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, args.clip)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return total, parts

    def evaluate(split, n_eval):
        idx_all = data.splits[split][:n_eval]
        agg, losses_sum, nb = None, None, 0
        for s0 in range(0, len(idx_all), args.batch):
            idx = idx_all[s0:s0 + args.batch]
            x, batch = data.batch(idx)
            pred = model(x, batch["f0"], temp=args.temp_lo, hard=True,
                         noise=False)   # 評価は決定的に
            total, parts, mapping = E.losses(pred, batch, x, SR, T_OFF, NH, LAM, matching=False)
            m = metrics_np(pred, batch, mapping, SR, T_OFF, N)
            m["loss"] = float(total)
            m["spec"] = float(parts["spec"])
            agg = {k: agg.get(k, 0) + v for k, v in m.items()} if agg else dict(m)
            nb += 1
        return {k: v / nb for k, v in agg.items()}

    if args.eval_only:
        r = evaluate(args.split, len(data.splits[args.split]))
        print(f"== §9.2 evaluation on {args.split} ==")
        for k, v in sorted(r.items()):
            print(f"  {k:18s} = {v:.4f}")
        return

    log_path = os.path.join(args.ckpt_dir, "log.csv")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["step", "temp", "total", "spec", "param", "gate",
                       "type", "sparse", "val_F1", "val_type_acc", "val_spec"])

    rng = np.random.default_rng(args.seed + int(step_var))
    train_idx = data.splits["train"]
    t0, last = time.time(), int(step_var)
    while int(step_var) < args.steps:
        step = int(step_var)
        idx = rng.choice(train_idx, args.batch, replace=False)
        x, batch = data.batch(idx)
        temp = temp_at(step)
        sl = spec_lam_at(step)
        total, parts = train_step(x, batch, tf.constant(temp, tf.float32),
                                  tf.constant(sl, tf.float32))
        step_var.assign_add(1)

        if step % 50 == 0:
            rate = (int(step_var) - last) / max(time.time() - t0, 1e-9)
            t0, last = time.time(), int(step_var)
            print(f"step {step:6d} T={temp:.2f} Ls={sl:.2f} total={float(total):8.3f} "
                  + " ".join(f"{k}={float(v):.3f}" for k, v in parts.items())
                  + f"  ({rate:.2f} it/s)", flush=True)

        val = {}
        if step > 0 and step % args.val_every == 0:
            val = evaluate("val", args.eval_n)
            print(f"  [val] F1={val['F1']:.3f} desc={val['desc_rec_rate']:.3f} "
                  f"comove={val['comove_F1']:.3f} full={val['full_rec_rate']:.3f} "
                  f"det={val['F1_detect']:.3f} "
                  f"srcid={val['srcid_acc']:.3f} n={val['n_pred']:.1f}/{val['n_true']:.1f} "
                  f"(core8 {val['F1_core8']:.3f} / "
                  f"cluster {val['F1_cluster']:.3f} / weak {val['F1_weak']:.3f}) "
                  f"FPrate={val['FP_rate']:.3f} depthMAE={val['depth_mae']:.3f} "
                  f"srcMAE={val['src_mae']:.3f} typeAcc={val['type_acc']:.3f} "
                  f"spec={val['spec']:.3f}", flush=True)
            score = val["F1"] + val["desc_rec_rate"]
            if score > best_score:
                best_score = score
                save_to(best_mgr)
                print(f"    best updated: F1={val['F1']:.3f} "
                      f"desc={val['desc_rec_rate']:.4f} → {best_dir}", flush=True)
        if step > 0 and step % args.ckpt_every == 0:
            save_to(mgr)
        logw.writerow([step, f"{temp:.3f}", f"{float(total):.4f}"]
                      + [f"{float(parts[k]):.4f}" for k in
                         ("spec", "param", "gate", "type", "sparse")]
                      + [f"{val.get('F1', ''):}", f"{val.get('type_acc', ''):}",
                         f"{val.get('spec', ''):}"])
        logf.flush()
    save_to(mgr)
    print("training complete")


if __name__ == "__main__":
    main()
