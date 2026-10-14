r"""order_analysis.py — ソース番号のずれ（順序の推定）の原因を切り分ける

  python order_analysis.py dump_v13_single_s{0,1,2}_test.npz

正解のソース番号は「使用中が先 → ENV はアタック昇順 → LFO はレート昇順」で並べてある
（train.Data）。モデルはこの並びを暗黙に推定して枠 0〜3 に出力しており、出力を規則で並べ替えてはいない。

[1] 番号がずれたサンプルの原因の内訳
    対応 π は「一致する配線の数が最大」のもの（同点なら恒等）。恒等でないサンプルを次の順に分類する。
      (a) 使用中のソースの数が違う（ゲートがあるソースの配線を 1 本も出さなかった、または余計に出した）
      (b) 数は同じだが、使用中の ENV / LFO の数が違う（型の取り違え）
      (c) 数も型の構成も同じだが、同じ型どうしの順が入れ替わっている
    (c) については、隣り合う同じ型のソースの真の差（ENV: アタック秒、LFO: レート比）ごとに逆転率を出す。
[2] モデル自身の推定（ゲートによる使用中の判断・予測の型・予測のアタック / レート）で
    出力を正解と同じ規則で並べ直したときの F1 と、番号がずれたサンプルの割合
"""
import argparse
from itertools import permutations

import numpy as np

import analyze_dump as A

K, TOL = A.K, A.TOL
PERMS = np.array(list(permutations(range(K))))
ENV_BINS = [0, 0.02, 0.05, 0.1, 0.2, 1.01]           # 真のアタックの差（秒）
LFO_BINS = [1, 1.25, 1.5, 2, 4, 1e9]                  # 真のレートの比


def load(path):
    D = A.Dump(path)
    d = np.load(path)
    D.sp_true, D.sp_pred = d["sp_true"], d["sp_pred"]
    return D


def wiring_pi(z_pred, z_true, pred_on):
    """一致する配線の数が最大の π（予測 k → 真 π[k]）。同点なら恒等（PERMS の先頭）"""
    if z_true.max() >= K:
        raise ValueError("枠が 4 より多い dump（typed、実験 41）には未対応")
    zp = np.minimum(z_pred, K - 1)
    match = (PERMS[:, zp] == z_true[None]) & pred_on[None]            # [24,B,M]
    return PERMS[np.argmax(match.sum(-1), axis=0)]                     # [B,K]


def f1(z_pred, z_true):
    on, ton = z_pred < K, z_true >= 0
    tp = (on & (z_pred == z_true)).sum()
    p, r = tp / on.sum(), tp / ton.sum()
    return 2 * p * r / (p + r)


def resort(D):
    """予測をモデル自身の推定で正解と同じ規則に並べ直した z_pred と shape_ok を返す"""
    used_p = np.stack([(D.z_pred == k).any(1) for k in range(K)], 1)
    key = (np.where(used_p, 0.0, 1e4)
           + np.where(D.type_pred == 1, D.sp_pred[..., 0], 100.0 + D.sp_pred[..., 0]))
    perm = np.argsort(key, axis=1, kind="stable")                      # 新しい枠 j ← 旧枠 perm[j]
    inv = np.argsort(perm, axis=1)                                     # 旧枠 → 新しい枠
    z_new = np.where(D.pred_on, np.take_along_axis(inv, np.minimum(D.z_pred, K - 1), 1), K)
    e_new = np.take_along_axis(D.e_param, perm[:, :, None], 1)
    hit = D.true_on & (z_new == D.z_true)
    b, m = np.nonzero(hit)
    sh = np.abs(e_new[b, z_new[b, m]] - D.e_true[b, D.z_true[b, m]]).mean(-1)
    return z_new, (sh <= TOL).mean()


def classify(D, pi):
    used_t = D.used
    used_p = np.stack([(D.z_pred == k).any(1) for k in range(K)], 1)
    ident = np.all(pi == np.arange(K), 1)
    n_t, n_p = used_t.sum(1), used_p.sum(1)
    env_t = (used_t & (D.type_true == 1)).sum(1)
    env_p = (used_p & (D.type_pred == 1)).sum(1)
    a = ~ident & (n_t != n_p)
    b = ~ident & ~a & (env_t != env_p)
    c = ~ident & ~a & ~b
    return ident, a, b, c


def swap_by_gap(D, pi):
    """隣り合う同じ型の使用中の真のソース (t, t+1) が、予測で逆順の枠に置かれた割合を、真の差ごとに"""
    inv_pi = np.argsort(pi, axis=1)                                    # 真 t → 予測の枠
    rows = {"ENV": [], "LFO": []}
    for b in range(D.B):
        for t in range(K - 1):
            if not (D.used[b, t] and D.used[b, t + 1]) or D.type_true[b, t] != D.type_true[b, t + 1]:
                continue
            x0, x1 = D.sp_true[b, t, 0], D.sp_true[b, t + 1, 0]
            swapped = inv_pi[b, t] > inv_pi[b, t + 1]
            if D.type_true[b, t] == 1:
                rows["ENV"].append((x1 - x0, swapped))
            else:
                rows["LFO"].append((x1 / max(x0, 1e-6), swapped))
    return {k: np.array(v) for k, v in rows.items()}


def param_err(D, pi):
    """番号を π で揃えた使用中ソースの、アタック（秒）とレート（比）の推定誤差"""
    inv_pi = np.argsort(pi, axis=1)
    env, lfo = [], []
    for b in range(D.B):
        for t in np.nonzero(D.used[b])[0]:
            k = inv_pi[b, t]
            if D.type_true[b, t] != D.type_pred[b, k]:
                continue
            xt, xp = D.sp_true[b, t, 0], D.sp_pred[b, k, 0]
            (env if D.type_true[b, t] == 1 else lfo).append(
                abs(xp - xt) if D.type_true[b, t] == 1 else max(xp, xt) / max(min(xp, xt), 1e-6))
    return np.array(env), np.array(lfo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    Ds = [load(p) for p in args.dumps]

    print("[1] 番号がずれたサンプルの原因の内訳（全サンプルに対する割合）")
    rows = []
    for D in Ds:
        pi = wiring_pi(D.z_pred, D.z_true, D.pred_on)
        ident, a, b, c = classify(D, pi)
        rows.append([~ident, a, b, c])
    r = np.array([[x.mean() for x in row] for row in rows])
    for i, name in enumerate(["ずれあり（合計）", "(a) 使用中の数が違う", "(b) 型の構成が違う",
                              "(c) 同じ型どうしの順の逆転"]):
        print(f"  {name:22s} {100 * r[:, i].mean():5.1f}% ± {100 * r[:, i].std(ddof=1) if len(Ds) > 1 else 0:.1f}")

    D = Ds[0]
    pi = wiring_pi(D.z_pred, D.z_true, D.pred_on)
    g = swap_by_gap(D, pi)
    env_err, lfo_err = param_err(D, pi)
    print(f"\n  (c) の詳細（{args.dumps[0]}）")
    print(f"    ENV: 隣り合う ENV の真のアタックの差 → 予測で逆順になった割合（アタックの推定誤差の中央値 {np.median(env_err):.3f} 秒）")
    for lo, hi in zip(ENV_BINS[:-1], ENV_BINS[1:]):
        s = (g["ENV"][:, 0] >= lo) & (g["ENV"][:, 0] < hi)
        if s.any():
            print(f"      差 {lo:.2f}〜{hi:.2f} 秒  n={s.sum():5d}  逆転 {100 * g['ENV'][s, 1].mean():5.1f}%")
    print(f"    LFO: 隣り合う LFO の真のレートの比 → 予測で逆順になった割合（レートの推定誤差の中央値 ×{np.median(lfo_err):.2f}）")
    for lo, hi in zip(LFO_BINS[:-1], LFO_BINS[1:]):
        s = (g["LFO"][:, 0] >= lo) & (g["LFO"][:, 0] < hi)
        if s.any():
            print(f"      比 ×{lo:.2f}〜{hi if hi < 1e8 else float('inf'):.2f}  n={s.sum():5d}  逆転 {100 * g['LFO'][s, 1].mean():5.1f}%")

    print("\n[2] モデル自身の推定で出力を並べ直した場合")
    res = []
    for D in Ds:
        z_new, sh_new = resort(D)
        pi0 = wiring_pi(D.z_pred, D.z_true, D.pred_on)
        pi1 = wiring_pi(z_new, D.z_true, D.pred_on)
        b, m = np.nonzero(D.hit)
        sh0 = (np.abs(D.e_param[b, D.z_pred[b, m]] - D.e_true[b, D.z_true[b, m]]).mean(-1) <= TOL).mean()
        res.append([f1(D.z_pred, D.z_true), f1(z_new, D.z_true),
                    1 - np.all(pi0 == np.arange(K), 1).mean(), 1 - np.all(pi1 == np.arange(K), 1).mean(),
                    sh0, sh_new])
    r = np.array(res)
    sd = r.std(0, ddof=1) if len(Ds) > 1 else np.zeros(r.shape[1])
    for i, name in enumerate(["F1（そのまま）", "F1（並べ直し）", "番号がずれたサンプル（そのまま）",
                              "番号がずれたサンプル（並べ直し）", "shape_ok（そのまま）", "shape_ok（並べ直し）"]):
        print(f"  {name:26s} {r[:, i].mean():.4f} ± {sd[i]:.4f}")


if __name__ == "__main__":
    main()
