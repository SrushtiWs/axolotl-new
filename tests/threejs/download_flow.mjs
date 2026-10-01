// On-image Download: saves exactly what Before / Mask / After shows.
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'
const [APP, OUT, PHOTO] = process.argv.slice(2)
const b = await chromium.launch({ channel: 'chrome' })
const ctx = await b.newContext({ acceptDownloads: true, viewport: { width: 1500, height: 950 } })
const p = await ctx.newPage()
const errors = []; p.on('pageerror', (e) => errors.push(e.message))
const idle = async () => { await p.waitForTimeout(1500); await p.waitForFunction(() => !document.querySelector('.marker[aria-busy="true"]'), null, { timeout: 300000 }); await p.waitForTimeout(1500) }
await p.goto(APP)
await p.locator('input[type=file]').first().setInputFiles(PHOTO)
await p.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 600000 })
await p.locator('.marker[aria-label="Floor"]').click()
await p.locator('button.tool.primary', { hasText: 'Apply' }).click(); await idle()
const out = { errors }
for (const step of ['After', 'Mask', 'Before']) {
  await p.locator('.step-switch button', { hasText: step }).click(); await p.waitForTimeout(2000)
  const src = await p.locator('img.stage-image').getAttribute('src')
  const [dl] = await Promise.all([p.waitForEvent('download'), p.locator('.stage-download').click()])
  const path = `${OUT}/dl_${step.toLowerCase()}.png`
  await dl.saveAs(path)
  // What the view shows, fetched in the page, for comparison.
  const shownB64 = await p.evaluate(async (u) => { const r = await fetch(u); const a = new Uint8Array(await r.arrayBuffer()); let s = ''; for (const x of a) s += String.fromCharCode(x); return btoa(s) }, src)
  writeFileSync(`${OUT}/shown_${step.toLowerCase()}`, Buffer.from(shownB64, 'base64'))
  out[step] = { filename: dl.suggestedFilename(), path }
}
await p.screenshot({ path: `${OUT}/download_button.png`, clip: { x: 330, y: 60, width: 900, height: 120 } })
writeFileSync(`${OUT}/download.json`, JSON.stringify(out, null, 2)); console.log(JSON.stringify(out))
await b.close()
