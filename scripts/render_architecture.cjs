// Render the canonical Mermaid source locally. Requires `playwright` and a browser.
// MERMAID_JS points at the local mermaid@10.9.3/dist/mermaid.min.js distribution.
// BROWSER_EXECUTABLE may select a local Chromium/Edge executable.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const source = fs.readFileSync(path.join(root, 'docs/system_architecture.mmd'), 'utf8');
  const library = process.env.MERMAID_JS || path.join(root, 'output/tools/mermaid-10.9.3.min.js');
  const options = { headless: true };
  if (process.env.BROWSER_EXECUTABLE) options.executablePath = process.env.BROWSER_EXECUTABLE;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 2400 }, deviceScaleFactor: 2 });
    const font = fs.readFileSync(path.join(root,'docs/fonts/MPLUS1p-Regular.ttf')).toString('base64');
    await page.setContent('<!doctype html><html lang="ja"><head><style>@font-face{font-family:ReportJP;src:url(data:font/ttf;base64,'+font+')} body{margin:0;background:white;font-family:ReportJP,sans-serif}</style></head><body><div id="diagram"></div></body></html>');
    await page.evaluate(() => document.fonts.load('18px ReportJP'));
    await page.addScriptTag({ path: library });
    await page.evaluate(async (text) => {
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'base',
        themeVariables: { fontFamily: 'ReportJP, sans-serif', fontSize: '18px', primaryColor: '#e3edf5',
          primaryTextColor: '#153552', primaryBorderColor: '#52677d', lineColor: '#52677d' },
        flowchart: { htmlLabels: true, curve: 'linear', nodeSpacing: 24, rankSpacing: 36 } });
      await mermaid.parse(text);
      const { svg } = await mermaid.render('system_architecture', text);
      document.getElementById('diagram').innerHTML = svg;
      const el = document.querySelector('svg');
      el.style.maxWidth = 'none';
      // Mermaid 10 sizes cluster headers too tightly for Japanese font metrics.
      // Keep the font size, expand the text viewport and lift the title clear of nodes.
      for (const label of el.querySelectorAll('g.cluster-label')) {
        const object=label.querySelector('foreignObject');
        if (object) object.setAttribute('height', Number(object.getAttribute('height'))+14);
        const transform=label.getAttribute('transform') || '';
        label.setAttribute('transform', transform+' translate(0,-8)');
      }
      const box = el.viewBox.baseVal;
      el.setAttribute('viewBox', `${box.x} ${box.y-10} ${box.width} ${box.height+10}`);
      el.setAttribute('width', box.width);
      el.setAttribute('height', box.height);
    }, source);
    const svg = page.locator('#diagram svg');
    const dest = path.join(root, 'docs/images');
    fs.mkdirSync(dest, { recursive: true });
    await svg.screenshot({ path: path.join(dest, 'system_architecture.png') });
    fs.writeFileSync(path.join(dest, 'system_architecture.svg'), await svg.evaluate(el => el.outerHTML));
    console.log('Mermaid syntax valid; architecture SVG and PNG generated.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
