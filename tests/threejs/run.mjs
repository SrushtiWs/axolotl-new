// Draw every case with the real 3D renderer; save each canvas as PNG.
//   node run.mjs <page url> <cases.json> <out dir>
import { chromium } from 'playwright-core'
import { readFileSync, writeFileSync } from 'node:fs'

const [url, casesPath, outDir] = process.argv.slice(2)
const cases = JSON.parse(readFileSync(casesPath, 'utf8'))
// Software WebGL, so the run needs no GPU and is repeatable.
const browser = await chromium.launch({ channel: 'chrome', args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage()
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
await page.goto(url)
await page.waitForFunction(() => typeof window.render3d === 'function')
const results = {}
for (const c of cases) {
  const { png, problems } = await page.evaluate(([b, s]) => window.render3d(b, s), [c.jobBase, c.surface])
  writeFileSync(`${outDir}/${c.name}.png`, Buffer.from(png.split(',')[1], 'base64'))
  results[c.name] = { problems }
}
writeFileSync(`${outDir}/results.json`, JSON.stringify({ results, errors }, null, 2))
await browser.close()
