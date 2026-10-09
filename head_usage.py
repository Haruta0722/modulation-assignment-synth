r"""head_usage.py — 共有層（trunk）の容量が、出力ヘッドごとにどれだけ使われているかを測る

  python head_usage.py --ckpt_dir ckpt_v18_single_s0 --arch v4 --log run_v18_s0.log

  1. 層ごとのパラメータ数と出力の数
  2. trunk（256 ユニット、ReLU）の実際の使われ方: 一度も働かないユニット、主成分で見た実効的な次元
  3. ヘッドごとの trunk の使い方: ユニット j が head h に送る信号の分散 Var(a_j)·Σ_o W_h[j,o]² を
     寄与とみなし、各ヘッドが受け取る寄与の合計と、各ユニットを主に使っているヘッドを数える
  4. 学習の終盤（最後の 2000 ステップ）の損失の配分（ログから）
"""
import argparse, re

import numpy as np
import tensorflow as tf

import estimator as E
import train as T
import weights_io as W

HEADS = [("h_env", "ENV パラメータ（4 ソース × 5）"), ("h_lfo", "LFO パラメータ（4 × 4）"),
         ("h_type", "ソースの型（4 × 2）"), ("h_gate", "ゲート（17 列 × 5 クラス）"),
         ("h_depth", "深さ（17 列）"), ("h_stat", "静的な値 θ_stat（17 列）"),
         ("h_static", "その他の静的な値（3）")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--ckpt_dir", default="ckpt_v18_single_s0")
    ap.add_argument("--arch", default="v4")
    ap.add_argument("--log", default="")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    model = E.Estimator(sr=SR, t_off=T_OFF, arch=args.arch)
    model(tf.zeros([1, N]), tf.constant([220.0]), temp=1.0, hard=True)
    W.load(model, W.latest(args.ckpt_dir))

    print("[1] 層ごとのパラメータ数")
    groups = {"畳み込み（2D、4 層）": model.convs, "BiGRU": [model.bigru]}
    if hasattr(model, "att"):
        groups["注意プーリング"] = [model.att]
    if hasattr(model, "tconv"):
        groups["時間方向の畳み込み（2 層）"] = model.tconv
    groups["trunk（共有層 256）"] = [model.trunk]
    for name, layers in groups.items():
        print(f"  {name:28s} {sum(l.count_params() for l in layers):>10,}")
    for h, label in HEADS:
        l = getattr(model, h)
        print(f"  {label:28s} {l.count_params():>10,}  （出力 {l.units}）")
    print(f"  {'合計':28s} {model.count_params():>10,}")

    idx = data.splits["test"][:args.n]
    acts = []
    for s in range(0, len(idx), 128):
        x, b = data.batch(idx[s:s + 128])
        acts.append(model.encode(x, b["f0"]).numpy())
    A = np.concatenate(acts)                                           # [n,256]

    print(f"\n[2] trunk の使われ方（test {len(A)} 件）")
    alive = (A > 0).any(0)
    var = A.var(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(A.T)))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    print(f"  一度も働かないユニット: {(~alive).sum()} / {A.shape[1]}")
    print(f"  ほぼ一定のユニット（分散 < 最大の 1%）: {(var < 0.01 * var.max()).sum()} / {A.shape[1]}")
    print(f"  主成分で分散の 90% / 99% を説明する次元: {np.searchsorted(cum, 0.9) + 1} / {np.searchsorted(cum, 0.99) + 1}")

    print("\n[3] ヘッドごとの trunk の使い方")
    contrib = {}
    for h, _ in HEADS:
        Wh = getattr(model, h).kernel.numpy()                          # [256,out]
        contrib[h] = var * (Wh ** 2).sum(1)                             # [256]
    C = np.stack([contrib[h] for h, _ in HEADS])                       # [H,256]
    share_unit = C / np.maximum(C.sum(0, keepdims=True), 1e-12)
    main_head = share_unit.argmax(0)
    total = C.sum(1) / C.sum()
    print(f"  {'ヘッド':28s}{'出力':>6s}{'寄与の割合':>12s}{'主に使うユニット':>18s}{'出力 1 つあたり':>16s}")
    for i, (h, label) in enumerate(HEADS):
        n_out = getattr(model, h).units
        n_main = int(((main_head == i) & alive).sum())
        print(f"  {label:28s}{n_out:6d}{100 * total[i]:11.1f}%{n_main:18d}{100 * total[i] / n_out:15.2f}%")

    # 学習済みモデルで学習データの損失を項ごとに計算する（ソースの損失 ×30 を含む v18 の設定）
    print("\n[4] 学習データ 512 件での損失の内訳（v18 の重みで換算）")
    lam = dict(spec=0.0, param=0.2, assign=3.0, sparse=0.0, none_w=0.25, depth_tf=1.0, src_w=30.0)
    fmax = E.F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    tr = data.splits["train"][:512]
    acc = np.zeros(4)
    for s in range(0, len(tr), 128):
        x, b = data.batch(tr[s:s + 128])
        p = model(x, b["f0"], temp=0.3, hard=True, noise=False)
        _, full, _ = E.losses(p, b, x, SR, T_OFF, NH, lam, matching=False)
        _, nosrc, _ = E.losses(p, b, x, SR, T_OFF, NH, dict(lam, src_w=0.0), matching=False)
        acc += [lam["param"] * (float(full["param"]) - float(nosrc["param"])),     # ソース（×30 込み）
                lam["param"] * float(nosrc["param"]),                             # 静的な値・深さ
                lam["assign"] * float(full["gate"]), lam["assign"] * float(full["type"])]
    acc /= len(range(0, len(tr), 128))
    names = ["ソースのパラメータ（ENV・LFO、×30 込み）", "静的な値・深さ", "ゲート CE", "型 CE"]
    tot = acc.sum()
    for k, v in zip(names, acc):
        print(f"  {k:36s} {v:7.3f}（{100 * v / tot:5.1f}%）")

    if args.log:
        print(f"\n[4] 学習の終盤の損失（{args.log} の最後の 2000 ステップ平均）")
        rows = []
        for line in open(args.log, encoding="utf-8", errors="ignore"):
            m = re.match(r"step\s+(\d+).*total=\s*([\d.]+).*param=([\d.]+) gate=([\d.]+) type=([\d.]+)", line)
            if m:
                rows.append([float(v) for v in m.groups()])
        R = np.array(rows)
        R = R[R[:, 0] >= R[-1, 0] - 2000].mean(0)
        lam = dict(param=0.2, assign=3.0)
        parts = {"param（静的な値・深さ・ソース × 重み）": lam["param"] * R[2],
                 "ゲート CE": lam["assign"] * R[3], "型 CE": lam["assign"] * R[4]}
        tot = sum(parts.values())
        for k, v in parts.items():
            print(f"  {k:36s} {v:7.3f}（{100 * v / tot:5.1f}%）")


if __name__ == "__main__":
    main()
