import sys, os
OUT = sys.argv[1]
FONT = "IPAPGothic, 'IPA Pゴシック', 'Hiragino Sans', 'Yu Gothic', 'Noto Sans JP', sans-serif"
DEFS = '''<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>'''
STYLE = '''<style>
text{fill:#000}
.n{font-size:24px}.b{font-size:23px;font-weight:bold}.s{font-size:19px;fill:#333}.d{font-size:19px;fill:#333}
.bx{fill:#fff;stroke:#000;stroke-width:2}.g{fill:#e6e6e6;stroke:#000;stroke-width:2}
.fx{fill:#fff;stroke:#000;stroke-width:2;stroke-dasharray:8 5}
.w{fill:none;stroke:#000;stroke-width:2}.by{fill:none;stroke:#555;stroke-width:1.6;stroke-dasharray:6 4}
.grp{fill:none;stroke:#000;stroke-width:1.4}
</style>'''

def t(x, y, s, cls="n", anchor="middle"):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>'
def box(x, y, w, h, cls="bx"):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
def arr(d, cls="w"):
    return f'<path class="{cls}" d="{d}" marker-end="url(#ar)"/>'

def synth(oy=0, label=None):
    e = []
    if label: e.append(t(0, oy + 20, label, "b", "start")); oy += 34
    # ソースと割り当て
    e += [box(10, oy+10, 180, 80), t(100, oy+43, "ENV 1〜4"), t(100, oy+75, "LFO 1〜3")]
    e += [arr(f"M190 {oy+50} H236")]
    e += [box(240, oy+10, 330, 80, "g"), t(405, oy+43, "割り当て R", "b"), t(405, oy+74, 'ソース <tspan font-style="italic">z</tspan><tspan font-size="14" dy="5" font-style="italic">m</tspan><tspan dy="-5"> と深さ </tspan><tspan font-style="italic">d</tspan><tspan font-size="14" dy="5" font-style="italic">m</tspan>', "s")]
    e += [arr(f"M570 {oy+50} H616"), t(624, oy+45, "18 の変調先へ", "n", "start"), t(624, oy+74, "（各段の下の項目）", "s", "start")]
    # 1 段目
    y = oy + 140; c = y + 35
    for x, w, a, b in [(10,180,"オシレータ ×2",None),(230,150,"ドライブ","tanh"),(420,150,"フィルタ",None),(610,150,"VCA",None)]:
        e.append(box(x, y, w, 70))
        if b: e += [t(x+w/2, y+30, a), t(x+w/2, y+57, b, "s")]
        else: e.append(t(x+w/2, y+43, a))
    e += [arr(f"M190 {c} H226"), arr(f"M380 {c} H416"), arr(f"M570 {c} H606")]
    for x, s in [(100,"level ×2, Fin ×2"),(305,"drive"),(495,"cutoff, res"),(685,"amp")]:
        e.append(t(x, y+100, s, "d"))
    # 2 段目（エフェクト）
    y2 = oy + 330; c2 = y2 + 35
    e.append(arr(f"M760 {c} H805 V{oy+282} H22 V{c2} H56"))
    fx = [(60,"コーラス","rate, depth, fb"),(240,"コンプ","thresh, ratio"),(420,"EQ","low, mid, high"),(600,"ディスト","drive, mix")]
    for x, n, dst in fx:
        e.append(f'<path class="by" d="M{x-15} {c2} V{y2-20} H{x+155} V{c2}"/>')
        e += [box(x, y2, 140, 70, "fx"), t(x+70, y2+43, n), t(x+70, y2+100, dst, "d")]
    e += [arr(f"M200 {c2} H236"), arr(f"M380 {c2} H416"), arr(f"M560 {c2} H596"), arr(f"M740 {c2} H766")]
    e += [box(770, y2, 62, 70), t(801, y2+43, "音")]
    e.append(t(10, oy+478, "各段の下の項目が変調先（計 18）。破線の段は確率 1/2 で素通し", "s", "start"))
    return "\n".join(e), oy + 490

def estimator(oy=0, label=None):
    e = []
    if label: e.append(t(0, oy + 20, label, "b", "start")); oy += 34
    e += [f'<rect class="bx" x="10" y="{oy+6}" width="180" height="44" rx="22"/>', t(100, oy+36, "音（1.5 s）")]
    e += [f'<rect class="bx" x="662" y="{oy+6}" width="120" height="44" rx="22"/>', t(722, oy+36, "f0")]
    e += [arr(f"M100 {oy+50} V{oy+76}"), arr(f"M722 {oy+50} V{oy+76}")]
    y = oy + 80; c = y + 40
    for x, w, a, b, cls in [(10,180,"対数メル","128 帯域","bx"),(225,160,"2 次元","畳み込み ×4","bx"),
                            (420,160,"双方向","GRU","bx"),(615,215,"共有の特徴","256 次元","bx")]:
        e += [box(x, y, w, 80, cls), t(x+w/2, y+34, a), t(x+w/2, y+63, b, "s")]
    e += [arr(f"M190 {c} H221"), arr(f"M385 {c} H416"), arr(f"M580 {c} H611")]
    gy = oy + 200
    e.append(arr(f"M722 {y+80} V{gy-4}"))
    e.append(f'<rect class="grp" x="2" y="{gy}" width="836" height="250" rx="8"/>')
    e.append(t(16, gy+30, "出力ヘッド 8 個（それぞれ全結合層 1 枚）", "b", "start"))
    heads = [("割り当て","18 ×（7 ＋ なし）"),("深さ","18（分類）"),("ENV の形","4 × 5"),("LFO のレート","3（分類）"),
             ("LFO の位相・波形","3 × 2"),("静的な値 <tspan font-style=\"italic\">θ</tspan>","18"),("その他の静的値","波形 2・makeup 1"),("エフェクトの有無","4（2 値）")]
    for i, (n, s) in enumerate(heads):
        x = 12 + (i % 4) * 206; yy = gy + 48 + (i // 4) * 98
        e += [box(x, yy, 196, 84, "g"), t(x+98, yy+35, n, "b"), t(x+98, yy+66, s, "s")]
    oy2 = gy + 250
    e.append(arr(f"M420 {oy2} V{oy2+30}"))
    e += [box(10, oy2+34, 820, 84), t(420, oy2+68, "推定結果：割り当て R̂（18 × 7）、ソースの形、静的な値、エフェクトの有無"),
          t(420, oy2+100, "ソースの型は枠で固定（ENV 4 枠・LFO 3 枠）。(a) のシンセに入れると音に戻せる", "s")]
    return "\n".join(e), oy2 + 124

def svg(body, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 {h}" width="84mm" height="{h/10}mm" '
            f'font-family="{FONT}">{DEFS}{STYLE}<rect width="840" height="{h}" fill="#fff"/>{body}</svg>')

a, ha = synth()
b, hb = estimator()
open(os.path.join(OUT, "fig_synth.svg"), "w").write(svg(a, ha))
open(os.path.join(OUT, "fig_estimator.svg"), "w").write(svg(b, hb))
a2, h1 = synth(0, "(a) シンセサイザ（構成 C）")
b2, h2 = estimator(h1 + 30, "(b) 推定器")
open(os.path.join(OUT, "fig_overview.svg"), "w").write(svg(a2 + b2, h2))
print(ha, hb, h2)
