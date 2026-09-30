// Before -> Mask -> After in the Studio: upload, select Floor + a wall, switch views, save each.
//   node mask_view.mjs <app url> <out dir> <photo>
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'
const [APP, OUT, PHOTO] = process.argv.slice(2)
const browser = await chromium.launch({ channel: 'chrome' })
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } })
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
const idle = async () => { await page.waitForTimeout(1500); await page.waitForFunction(() => !document.querySelector('.marker[aria-busy="true"]'), null, { timeout: 300000 }); await page.waitForTimeout(1500) }
await page.goto(APP)
await page.locator('input[type=file]').first().setInputFiles(PHOTO)
await page.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 600000 })
const labels = await page.evaluate(() => [...document.querySelectorAll('.marker')].map((m) => m.getAttribute('aria-label')))
const wall = labels.find((l) => l !== 'Floor')
await page.locator('.marker[aria-label="Floor"]').click(); await idle()
await page.locator(`.marker[aria-label="${wall}"]`).click(); await idle()
const jobs = await page.evaluate(() => performance.getEntriesByType('resource').map((r) => r.name).filter((u) => /\/jobs\/[0-9a-f]+\//.test(u)))
const result = { selected: ['Floor', wall] }
for (const step of ['Before', 'Mask', 'After']) {
  await page.locator('.step-switch button', { hasText: step }).click()
  await page.waitForTimeout(step === 'Mask' ? 2500 : 1200)
  const src = await page.locator('img.stage-image').getAttribute('src')
  result[step] = src.startsWith('data:') ? 'data:image/png (mask)' : src
  if (step === 'Mask') writeFileSync(`${OUT}/mask_data_full.png`, Buffer.from(src.split(',')[1], 'base64'))
  await page.screenshot({ path: `${OUT}/mask_view_${step.toLowerCase()}.png`, clip: { x: 330, y: 60, width: 900, height: 700 } })
}
result.clean_job = (jobs.find((u) => /segments|surfaces/.test(u)) || '').match(/jobs\/([0-9a-f]+)\//)?.[1] ?? null
result.composed = (await page.locator('img.stage-image').getAttribute('src'))
result.errors = errors
writeFileSync(`${OUT}/mask_view.json`, JSON.stringify(result, null, 2))
console.log(JSON.stringify(result))
await browser.close()
