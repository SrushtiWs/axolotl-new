// Diagnose: per selection, every /generate call (status, error), failed dots, page errors.
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'
const [APP, OUT, PHOTO, MODE] = process.argv.slice(2)   // MODE "rerun": press Re-run first
const b = await chromium.launch({ channel: 'chrome' })
const p = await (await b.newContext({ viewport: { width: 1500, height: 950 } })).newPage()
const errors = []; p.on('pageerror', (e) => errors.push(e.message))
let calls = []
p.on('response', async (r) => {
  if (!r.url().includes('/generate')) return
  const buf = r.request().postDataBuffer()
  const body = buf ? buf.toString('latin1') : ''
  const field = (n) => (body.match(new RegExp(`name="${n}"\\r\\n\\r\\n([^\\r]*)`)) || [])[1]
  let detail = null
  if (r.status() !== 200) { try { detail = (await r.json()).detail } catch { detail = await r.text().catch(() => null) } }
  calls.push({ status: r.status(), surface: field('surface'), wall_id: field('wall_id') || null, detail })
})
const idle = async () => {
  await p.waitForTimeout(2500)
  await p.waitForFunction(() => !document.querySelector('.marker[aria-busy="true"]'), null, { timeout: 300000 })
  await p.waitForTimeout(2500)
}
await p.goto(APP)
await p.locator('input[type=file]').first().setInputFiles(PHOTO)
await p.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 600000 })
await p.waitForTimeout(1500)
if (MODE === 'rerun') {
  const t = Date.now()
  await p.locator('button', { hasText: 'Re-run' }).first().click()
  await p.waitForFunction(() => [...document.querySelectorAll('button')].some((b) => /Cleaning/.test(b.textContent || '')), null, { timeout: 60000 })
  await p.waitForFunction(() => [...document.querySelectorAll('button')].some((b) => (b.textContent || '').trim() === 'Re-run'), null, { timeout: 900000 })
  await p.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 120000 })
  await p.waitForTimeout(3000)
  console.log(`re-run took ${((Date.now() - t) / 1000).toFixed(1)} s`)
}
const labels = await p.locator('.marker').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')))
const walls = labels.filter((l) => l !== 'Floor')
const steps = [['Floor'], ...walls.map((w) => [w]), labels]
const out = { labels, steps: [], errors }
for (const want of steps) {
  calls = []
  for (const l of labels) {
    const m = p.locator(`.marker[aria-label="${l}"]`)
    const on = (await m.getAttribute('aria-checked')) === 'true'
    if (on !== want.includes(l)) await m.click()
  }
  await p.locator('button.tool.primary', { hasText: 'Apply' }).click()
  await idle()
  const failed = await p.locator('.marker.failed, .marker[title*="could not be tiled"]').evaluateAll((els) => els.map((e) => e.getAttribute('title')))
  const shot = `${OUT}/step_${want.length > 2 ? 'all' : want.join('_').replace(/ /g, '')}.png`
  await p.screenshot({ path: shot })
  out.steps.push({ selected: want, generate: calls, failed_markers: failed, screenshot: shot })
  console.log(JSON.stringify({ selected: want, generate: calls.map((c) => `${c.status} ${c.surface}${c.wall_id ? ':' + c.wall_id : ''}${c.detail ? ' -> ' + String(c.detail).slice(0, 160) : ''}`), failed }))
}
writeFileSync(`${OUT}/diagnose.json`, JSON.stringify(out, null, 2))
console.log('page errors:', JSON.stringify(errors))
await b.close()
