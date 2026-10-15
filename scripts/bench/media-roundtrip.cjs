// Runs dev/media-test.html on a fixture in headless Chromium, saves the output file, prints the result JSON.
// usage: node media-roundtrip.cjs <base url> <fixture url path> <out file> [playwright module path]
const fs = require('node:fs');
const { chromium } = require(process.argv[5] || 'playwright');

(async () => {
  const [base, src, outFile] = process.argv.slice(2);
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.text()); });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(`${base}/dev/media-test.html?src=${encodeURIComponent(src)}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__roundTripResult !== undefined, null, { timeout: 600000 });
  const result = await page.evaluate(() => window.__roundTripResult);
  console.log('RESULT ' + JSON.stringify(result));
  const which = result.output ? '__outputBytes' : '__wavBytes';
  const has = await page.evaluate((w) => !!window[w], which);
  if (has && outFile) {
    const b64 = await page.evaluate((w) => {
      const bytes = window[w];
      let bin = '';
      for (let k = 0; k < bytes.length; k += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(k, k + 0x8000));
      return btoa(bin);
    }, which);
    fs.writeFileSync(outFile, Buffer.from(b64, 'base64'));
    console.log(`saved ${outFile}`);
  }
  await browser.close();
  process.exit(result.error ? 2 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
