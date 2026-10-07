// Pink-wall living room, Desktop build: upload, wait for the surface dots, record them,
// select Floor + the wall whose dot is nearest the image centre, Apply, then the Mask tab.
//   node ../tests/threejs/pink_room.mjs http://localhost:5173 <out_dir> <photo>
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'
const [APP, OUT, PHOTO] = process.argv.slice(2)
const b = await chromium.launch({ channel: 'chrome' })
const p = await (await b.newContext({ viewport: { width: 1440, height: 1000 } })).newPage()
const errors = []; p.on('pageerror', (e) => errors.push(e.message))
await p.goto(APP)
await p.locator('input[type=file]').first().setInputFiles(PHOTO)
await p.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 1500000 })
await p.waitForTimeout(1500)
const dots = await p.evaluate(() => {
  const img = document.querySelector('img.stage-image').getBoundingClientRect()
  return [...document.querySelectorAll('.marker')].map((m) => {
    const r = m.querySelector('.marker-dot').getBoundingClientRect()
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2
    return { label: m.getAttribute('aria-label'), x_in_image_pct: +(100 * (cx - img.left) / img.width).toFixed(1),
             y_in_image_pct: +(100 * (cy - img.top) / img.height).toFixed(1), inside: cx >= img.left && cx <= img.right && cy >= img.top && cy <= img.bottom }
  })
})
await p.screenshot({ path: `${OUT}/studio_dots.png` })
const walls = dots.filter((d) => d.label !== 'Floor')
const back = walls.sort((a, c) => Math.abs(a.x_in_image_pct - 50) - Math.abs(c.x_in_image_pct - 50))[0]
await p.locator('.marker[aria-label="Floor"]').click()
if (back) await p.locator(`.marker[aria-label="${back.label}"]`).click()
await p.locator('button.tool.primary', { hasText: 'Apply' }).click()
await p.locator('.stage-result-actions button', { hasText: 'Download Image' }).waitFor({ timeout: 900000 })
await p.waitForTimeout(2000)
await p.screenshot({ path: `${OUT}/studio_after_floor_backwall.png` })
await p.locator('.step-switch button', { hasText: 'Mask' }).click()
await p.waitForTimeout(2500)
await p.screenshot({ path: `${OUT}/studio_mask_tab.png` })
const out = { errors, dots, selected_wall: back?.label }
writeFileSync(`${OUT}/pink_room_ui.json`, JSON.stringify(out, null, 1)); console.log(JSON.stringify(out, null, 1))
await b.close()
