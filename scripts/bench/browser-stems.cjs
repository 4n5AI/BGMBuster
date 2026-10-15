// Runs the bench page on given input files in headless Chromium and saves the separated stems as .f32 files,
// for scripts/parity/browser_parity.py. Requires a server for the built site that also serves the input files.
// usage: node browser-stems.cjs <bench url with inputL/inputR params> <out_dir> [playwright module path]
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.argv[4] || 'playwright');

(async () => {
  const [url, outDir] = process.argv.slice(2);
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.on('console', (m) => console.log('[page]', m.text()));
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__benchResults && window.__benchResults.length > 0, null, { timeout: 1800000 });
  const result = await page.evaluate(() => window.__benchResults[0]);
  console.log(JSON.stringify(result));
  if (result.error) process.exit(1);
  fs.mkdirSync(outDir, { recursive: true });
  for (const src of ['music', 'speech', 'sfx']) {
    const n = await page.evaluate((s) => window.__benchStems[s].length, src);
    for (let c = 0; c < n; c++) {
      // transfer as base64 to keep the round trip cheap
      const b64 = await page.evaluate(([s, i]) => {
        const bytes = new Uint8Array(window.__benchStems[s][i].buffer);
        let bin = '';
        for (let k = 0; k < bytes.length; k += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(k, k + 0x8000));
        return btoa(bin);
      }, [src, c]);
      fs.writeFileSync(path.join(outDir, `browser_${src}_${'LR'[c]}.f32`), Buffer.from(b64, 'base64'));
    }
  }
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
