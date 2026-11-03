const { chromium } = require('playwright'); const fs = require('fs');
(async () => {
  const b = await chromium.launch();
  for (const n of ['fig_synth', 'fig_estimator', 'fig_overview']) {
    const svg = fs.readFileSync(n + '.svg', 'utf8');
    const h = parseFloat(svg.match(/height="([\d.]+)mm"/)[1]);
    const p = await b.newPage({ viewport: { width: 840, height: Math.ceil(h * 10) }, deviceScaleFactor: 3 });
    await p.setContent(`<html><body style="margin:0">${svg}</body></html>`);
    await p.pdf({ path: n + '.pdf', width: '84mm', height: h + 'mm', printBackground: true, pageRanges: '1' });
    await p.setContent(`<html><body style="margin:0">${svg.replace(/width="84mm" height="[\d.]+mm"/, 'width="840" height="' + Math.ceil(h*10) + '"')}</body></html>`);
    await p.screenshot({ path: n + '.png', fullPage: true });
    await p.close();
  }
  await b.close();
})();
