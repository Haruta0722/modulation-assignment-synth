// 梗概の図 1 を PowerPoint の図形（編集可能）で作る。node make_pptx.js
const pptxgen = require("pptxgenjs");
const JA = "游明朝", LA = "Times New Roman";
const PT = 7, PT_S = 6;              // 本文 7 pt、変調先の名前 6 pt
const LW = 0.75;                     // 線の太さ [pt]
const mm = (v) => v / 25.4;

function deck(wmm, hmm) {
  const p = new pptxgen();
  p.defineLayout({ name: "FIG", width: mm(wmm), height: mm(hmm) });
  p.layout = "FIG";
  p.theme = { headFontFace: JA, bodyFontFace: JA };
  return p;
}
// 文字列 → ラン。$x$ は Times の斜体、_m は下付き
function runs(s, size) {
  const out = [];
  for (const part of s.split(/(\$[^$]+\$)/)) {
    if (!part) continue;
    if (part.startsWith("$")) {
      const [base, sub] = part.slice(1, -1).split("_");
      out.push({ text: base, options: { fontFace: LA, italic: true, fontSize: size } });
      if (sub) out.push({ text: sub, options: { fontFace: LA, italic: true, fontSize: size, subscript: true } });
    } else out.push({ text: part, options: { fontFace: JA, fontSize: size } });
  }
  return out;
}
function lines(arr, size) {           // 複数行
  const r = [];
  arr.forEach((s, i) => {
    const rr = runs(s, size);
    if (i < arr.length - 1) rr[rr.length - 1].options.breakLine = true;
    r.push(...rr);
  });
  return r;
}
function box(sl, x, y, w, h, text, o = {}) {
  sl.addText(lines([].concat(text), o.size || PT), {
    shape: "rect", x: mm(x), y: mm(y), w: mm(w), h: mm(h), margin: 0,
    align: "center", valign: "middle", color: "000000", fill: { color: "FFFFFF" },
    line: { color: "000000", width: LW, dashType: o.dash ? "dash" : "solid" },
    lineSpacingMultiple: 0.95, objectName: o.name,
  });
}
function label(sl, x, y, w, text, o = {}) {
  sl.addText(lines([].concat(text), o.size || PT_S), {
    x: mm(x), y: mm(y), w: mm(w), h: mm(o.h || 3), margin: 0, isTextBox: true,
    align: o.align || "center", valign: "middle", color: "000000", objectName: o.name,
  });
}
// 折れ線（最後の区間だけ矢印）。pts = [[x,y],...]
function wire(sl, pts, o = {}) {
  for (let i = 0; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
    const last = i === pts.length - 2;
    sl.addShape("line", {
      x: mm(Math.min(x1, x2)), y: mm(Math.min(y1, y2)), w: mm(Math.abs(x2 - x1)), h: mm(Math.abs(y2 - y1)),
      flipH: x2 < x1, flipV: y2 < y1,
      line: { color: o.color || "000000", width: o.w || LW, dashType: o.dash ? "dash" : "solid",
              endArrowType: last && !o.noArrow ? "triangle" : undefined },
    });
  }
}

// ---------------- (a) シンセサイザ（構成 C） ----------------
async function synth() {
  const p = deck(84, 46), s = p.addSlide();
  // ソースと割り当て
  box(s, 1, 1, 17, 8, ["ENV 1～4", "LFO 1～3"], { name: "ソース" });
  wire(s, [[18, 5], [24, 5]]);
  box(s, 24, 1, 24, 8, ["割り当て $R$", "$z_m$, $d_m$"], { name: "割り当て" });
  wire(s, [[48, 5], [54, 5]]);
  label(s, 55, 3.5, 28, "18 の変調先へ", { size: PT, align: "left" });
  // 1 段目: オシレータ → ドライブ → フィルタ → VCA
  const r1 = 13, c1 = r1 + 3;
  const row1 = [[1, 19, "オシレータ×2", "level×2, Fin×2"], [24, 13, "ドライブ", "drive"],
                [41, 13, "フィルタ", "cutoff, res"], [58, 11, "VCA", "amp"]];
  row1.forEach(([x, w, n, d]) => { box(s, x, r1, w, 6, n, { name: n }); label(s, x - 3, r1 + 6.6, w + 6, d); });
  wire(s, [[20, c1], [24, c1]]); wire(s, [[37, c1], [41, c1]]); wire(s, [[54, c1], [58, c1]]);
  // VCA → コーラスへ折り返す
  const r2 = 30, c2 = r2 + 3;
  wire(s, [[69, c1], [81, c1], [81, 25.5], [2, 25.5], [2, c2], [6, c2]]);
  // 2 段目: エフェクト（破線の枠 = 確率 1/2 で素通し）
  const row2 = [[6, 13, "コーラス", "rate, depth, fb"], [26, 11, "コンプ", "thresh, ratio"],
                [44, 9, "EQ", "low, mid, high"], [60, 12, "ディスト", "drive, mix"]];
  row2.forEach(([x, w, n, d]) => {
    box(s, x, r2, w, 6, n, { dash: true, name: n });
    label(s, x - 4, r2 + 6.6, w + 8, d);
    wire(s, [[x - 2, c2], [x - 2, r2 - 2.5], [x + w + 2, r2 - 2.5], [x + w + 2, c2]], { noArrow: true, w: 0.5 });
  });
  wire(s, [[19, c2], [26, c2]]); wire(s, [[37, c2], [44, c2]]); wire(s, [[53, c2], [60, c2]]); wire(s, [[72, c2], [78, c2]]);
  box(s, 78, r2, 5, 6, "音", { name: "音" });
  label(s, 1, 42, 82, "破線の段はそれぞれ確率 1/2 で素通し", { align: "left" });
  await p.writeFile({ fileName: "fig_synth.pptx" });
}

// ---------------- (b) 推定器 ----------------
// 左: エンコーダ（全出力で共有）、中: 出力ヘッド 8 個、右: 各ヘッドの出力。枠の名前は下側
function group(sl, x, y, w, h, title, sub, name) {
  sl.addShape("rect", { x: mm(x), y: mm(y), w: mm(w), h: mm(h), fill: { type: "none" },
                        line: { color: "000000", width: 0.5 }, objectName: name });
  label(sl, x, y + h - 7, w, title, { size: PT, h: 3 });          // 枠の名前は枠の下側に置く
  if (sub) label(sl, x, y + h - 3.8, w, sub, { size: PT_S, h: 2.6 });
}
async function estimator() {
  const p = deck(84, 58), s = p.addSlide();
  const top = 4, H = 53;
  group(s, 0.5, top, 29, H, "エンコーダ", "全出力で共有", "枠: エンコーダ");
  group(s, 35.5, top, 26, H, "出力ヘッド（8 個）", "各 全結合 1 層", "枠: 出力ヘッド");
  // 入力
  label(s, 4, 0.2, 22, "音（1.5 s）", { size: PT });
  wire(s, [[18, 3.2], [18, 7]]);
  // エンコーダの中: 対数メル → 畳み込み → GRU → 全結合
  const ex = 8, ew = 20, eh = 7;
  const enc = [[7, ["対数メル", "128 帯域"], "対数メル"], [17, ["2 次元", "畳み込み×4"], "畳み込み"],
               [27, ["双方向 GRU"], "GRU"], [37, ["全結合", "256 次元"], "共有の全結合層"]];
  enc.forEach(([y, t, n], i) => {
    box(s, ex, y, ew, eh, t, { name: n });
    if (i < enc.length - 1) wire(s, [[ex + ew / 2, y + eh], [ex + ew / 2, enc[i + 1][0]]]);
  });
  label(s, 1, 38.5, 5, "$f_0$", { size: PT });
  wire(s, [[5.2, 40.5], [ex, 40.5]]);
  // 共有の特徴 → 8 個のヘッドへ分岐
  const heads = [
    ["割り当て", "$z_m$：8 択 × 18"],
    ["深さ", "$d_m$ × 18"],
    ["ENV の形", "AHDSR × 4 枠"],
    ["LFO のレート", "レート × 3 枠"],
    ["LFO の位相・波形", "位相, 波形 × 3 枠"],
    ["静的な値", "$θ_m$ × 18"],
    ["その他の静的な値", "波形 × 2, makeup"],
    ["エフェクトの有無", "有無 × 4 段"],
  ];
  const hx = 37.5, hw = 22, hh = 4.2, y0 = 7, pitch = 5.1, bus = 32.5;
  const cy = (i) => y0 + i * pitch + hh / 2;
  wire(s, [[ex + ew, 40.5], [bus, 40.5]], { noArrow: true });
  wire(s, [[bus, cy(0)], [bus, cy(7)]], { noArrow: true });
  heads.forEach(([n, o], i) => {
    wire(s, [[bus, cy(i)], [hx, cy(i)]]);
    box(s, hx, y0 + i * pitch, hw, hh, n, { name: "ヘッド: " + n });
    wire(s, [[hx + hw, cy(i)], [63, cy(i)]]);
    label(s, 63.6, cy(i) - 1.5, 20.4, o, { align: "left" });
  });
  await p.writeFile({ fileName: "fig_estimator.pptx" });
}

// 矢印の先を小さく（pptxgenjs では指定できないので XML を書き換える）
const JSZip = require("jszip"), fs = require("fs");
async function smallArrows(f) {
  const z = await JSZip.loadAsync(fs.readFileSync(f));
  const n = "ppt/slides/slide1.xml";
  const x = (await z.file(n).async("string")).replace(/<a:tailEnd type="triangle"\/>/g, '<a:tailEnd type="triangle" w="sm" len="sm"/>');
  z.file(n, x);
  fs.writeFileSync(f, await z.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}
(async () => { await synth(); await estimator(); await smallArrows("fig_synth.pptx"); await smallArrows("fig_estimator.pptx"); })();
