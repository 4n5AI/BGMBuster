// End-to-end run of the app in headless Chromium: choose a fixture, remove the music, save the video and the
// WAV, switch language, and exercise an error path. Prints a JSON summary; saves downloads to <out_dir>.
// usage: node app-e2e.cjs <base url> <fixture path on disk> <out_dir> [playwright module path] [error fixture]
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.argv[5] || 'playwright');

(async () => {
  const [base, fixture, outDir, , errorFixture] = process.argv.slice(2);
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ acceptDownloads: true, locale: 'en-US' });
  const page = await context.newPage();
  // Headless Chromium has showSaveFilePicker but its dialog is auto-cancelled; exercise the download fallback.
  await page.addInitScript(() => { delete window.showSaveFilePicker; });
  const logs = [];
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  const summary = { steps: [] };
  const step = (name, ok, extra) => { summary.steps.push({ name, ok, ...extra }); console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' ' + JSON.stringify(extra) : ''}`); };

  await page.goto(`${base}/`, { waitUntil: 'load' });
  step('page loads in English', (await page.locator('h1').textContent()) === 'BGM Buster');
  await page.getByRole('button', { name: '日本語' }).click();
  step('switch to Japanese', (await page.locator('h1').textContent()) === 'BGM排除くん' && (await page.evaluate(() => document.documentElement.lang)) === 'ja');
  await page.getByRole('button', { name: 'EN' }).click();

  // error path: a file without audio
  if (errorFixture) {
    await page.locator('input[type=file]').setInputFiles(errorFixture);
    await page.getByRole('alert').waitFor({ timeout: 30000 });
    step('no-audio error shown', (await page.getByRole('alert').textContent()).includes('no audio'));
  }

  // happy path
  await page.locator('input[type=file]').setInputFiles(fixture);
  await page.getByRole('button', { name: 'Remove music' }).waitFor({ timeout: 30000 });
  const summaryText = await page.locator('.file-summary').textContent();
  step('file summary', summaryText.includes(path.basename(fixture)), { summaryText });
  const t0 = Date.now();
  await page.getByRole('button', { name: 'Remove music' }).click();
  await page.getByRole('button', { name: 'Cancel' }).waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Save audio (WAV)' }).waitFor({ timeout: 600000 });
  const seconds = (Date.now() - t0) / 1000;
  step('separation finished', true, { seconds });
  const tech = await page.evaluate(() => Array.from(document.querySelectorAll('dl.tech dd')).map((d) => d.textContent));
  summary.tech = tech;

  // listen controls
  await page.getByRole('button', { name: 'Music only' }).click();
  const sliderDisabled = await page.locator('#music-level').isDisabled();
  await page.getByRole('button', { name: 'Music removed' }).click();
  await page.locator('#music-level').fill('25');
  step('slider disabled for music-only and re-enabled', sliderDisabled && !(await page.locator('#music-level').isDisabled()));

  // play a bit (muted video drives Web Audio); headless has no audio output but the graph must not throw
  await page.evaluate(() => { const v = document.querySelector('video, audio'); return v && v.play().catch(() => undefined); });
  await page.waitForTimeout(1500);
  const playing = await page.evaluate(() => { const v = document.querySelector('video, audio'); return v ? !v.paused && v.currentTime > 0.5 : false; });
  await page.evaluate(() => { const v = document.querySelector('video, audio'); if (v) v.pause(); });
  step('preview plays', playing);

  // downloads
  const saveVideo = page.getByRole('button', { name: 'Save video' });
  if (await saveVideo.count()) {
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 300000 }), saveVideo.click()]);
    const file = path.join(outDir, download.suggestedFilename());
    await download.saveAs(file);
    summary.video = file;
    step('video saved', fs.statSync(file).size > 1000, { file, bytes: fs.statSync(file).size });
  }
  const [wav] = await Promise.all([page.waitForEvent('download', { timeout: 300000 }), page.getByRole('button', { name: 'Save audio (WAV)' }).click()]);
  const wavFile = path.join(outDir, wav.suggestedFilename());
  await wav.saveAs(wavFile);
  summary.wav = wavFile;
  step('wav saved', fs.statSync(wavFile).size > 1000, { file: wavFile });

  // a stem from the details section
  await page.locator('details summary').click();
  const [stem] = await Promise.all([page.waitForEvent('download', { timeout: 300000 }), page.getByRole('button', { name: 'Music', exact: true }).click()]);
  const stemFile = path.join(outDir, stem.suggestedFilename());
  await stem.saveAs(stemFile);
  step('music stem saved', fs.statSync(stemFile).size > 1000, { file: stemFile });

  // reset
  await page.getByRole('button', { name: 'Process another file' }).click();
  step('reset to idle', (await page.getByRole('button', { name: 'Remove music' }).count()) === 0);

  // cancel path
  await page.locator('input[type=file]').setInputFiles(fixture);
  await page.getByRole('button', { name: 'Remove music' }).click();
  await page.getByRole('button', { name: 'Cancel' }).waitFor({ timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.getByRole('button', { name: 'Cancel' }).click();
  await page.getByRole('button', { name: 'Remove music' }).waitFor({ timeout: 30000 });
  step('cancel returns to loaded', (await page.locator('text=Canceled').count()) > 0);

  summary.logs = logs;
  console.log('SUMMARY ' + JSON.stringify(summary));
  await browser.close();
  process.exit(summary.steps.every((s) => s.ok) ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(1); });
