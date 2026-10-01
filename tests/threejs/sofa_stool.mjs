// Living room: select Floor + every wall, Apply, save the result and the job ids.
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'
const [APP, OUT, PHOTO] = process.argv.slice(2)
const b = await chromium.launch({ channel: 'chrome' })
const p = await b.newPage({ viewport: { width: 1500, height: 950 } })
const errors = []; p.on('pageerror', (e) => errors.push(e.message))
const idle = async () => { await p.waitForTimeout(1500); await p.waitForFunction(() => !document.querySelector('.marker[aria-busy="true"]'), null, { timeout: 300000 }); await p.waitForTimeout(2000) }
await p.goto(APP)
await p.locator('input[type=file]').first().setInputFiles(PHOTO)
await p.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 600000 })
const labels = await p.evaluate(() => [...document.querySelectorAll('.marker')].map((m) => m.getAttribute('aria-label')))
for (const l of labels) await p.locator(`.marker[aria-label="${l}"]`).click()
await p.locator('button.tool.primary', { hasText: 'Apply' }).click(); await idle()
const composed = await p.locator('img.stage-image').getAttribute('src')
const clean = await p.evaluate(() => performance.getEntriesByType('resource').map((r) => r.name).find((u) => /\/jobs\/[0-9a-f]+\/segments\//.test(u)))
await p.screenshot({ path: `${OUT}/living_after.png`, clip: { x: 330, y: 60, width: 900, height: 700 } })
const r = { labels, composed, clean_job: clean?.match(/jobs\/([0-9a-f]+)\//)?.[1], errors }
writeFileSync(`${OUT}/sofa_stool.json`, JSON.stringify(r, null, 2)); console.log(JSON.stringify(r))
await b.close()
