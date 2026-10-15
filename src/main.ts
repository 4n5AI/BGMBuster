// Entry point. The UI (M3) is built on top of the separation engine in src/separation/.
const app = document.getElementById('app');
if (app) {
  const p = document.createElement('p');
  p.textContent = 'BGM排除くん（BGM Buster）: UI is under construction. See bench.html for the engine benchmark.';
  app.append(p);
}
