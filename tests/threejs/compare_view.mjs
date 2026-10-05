// Compare view: both images drawn at the same size, divider + handle, Left / ✕ / Right bar.
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'
const [APP, OUT, PHOTO] = process.argv.slice(2)
const b = await chromium.launch({ channel: 'chrome' })
const p = await (await b.newContext({ viewport: { width: 1500, height: 950 } })).newPage()
const errors = []; p.on('pageerror', (e) => errors.push(e.message))
await p.goto(APP)
await p.locator('input[type=file]').first().setInputFiles(PHOTO)
await p.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 900000 })
await p.locator('.marker[aria-label="Floor"]').click()
await p.locator('button.tool.primary', { hasText: 'Apply' }).click()
await p.locator('.stage-result-actions button', { hasText: 'Download Image' }).waitFor({ timeout: 600000 })
await p.locator('button.tool', { hasText: 'Compare' }).click()
await p.locator('.compare-top img').waitFor()
await p.waitForTimeout(1500)
const boxes = async () => p.evaluate(() => {
  const r = (el) => { const x = el.getBoundingClientRect(); return [Math.round(x.left), Math.round(x.top), Math.round(x.width), Math.round(x.height)] }
  // the drawn picture inside each <img> (object-fit: contain)
  const drawn = (i) => { const x = i.getBoundingClientRect(), s = Math.min(x.width / i.naturalWidth, x.height / i.naturalHeight)
    const w = i.naturalWidth * s, h = i.naturalHeight * s; return [Math.round(x.left + (x.width - w) / 2), Math.round(x.top + (x.height - h) / 2), Math.round(w), Math.round(h)] }
  const imgs = [...document.querySelectorAll('.compare img')]
  return { box: r(document.querySelector('.compare')), drawn: imgs.map(drawn),
           natural: imgs.map((i) => [i.naturalWidth, i.naturalHeight]), handle: r(document.querySelector('.compare-handle')),
           clip: document.querySelector('.compare-top').style.clipPath,
           tags: [...document.querySelectorAll('.compare-tag')].map((t) => t.className + ':' + t.textContent) }
})
const out = { errors, middle: await boxes() }
await p.screenshot({ path: `${OUT}/compare_middle.png` })
const bx = out.middle.drawn[0]
await p.mouse.move(bx[0] + bx[2] * 0.5, bx[1] + bx[3] / 2); await p.mouse.down()
await p.mouse.move(bx[0] + bx[2] * 0.25, bx[1] + bx[3] / 2, { steps: 5 }); await p.mouse.up()
out.dragged_to_25 = await boxes(); await p.screenshot({ path: `${OUT}/compare_drag25.png` })
await p.locator('.compare-bar button', { hasText: 'Left' }).click(); out.left = await boxes()
await p.screenshot({ path: `${OUT}/compare_left.png` })
await p.locator('.compare-bar button', { hasText: 'Right' }).click(); out.right = await boxes()
await p.screenshot({ path: `${OUT}/compare_right.png` })
await p.locator('.compare-bar button', { hasText: '✕' }).click(); await p.waitForTimeout(300)
out.after_close = { compare_present: await p.locator('.compare').count(), stage_image: await p.locator('img.stage-image').count() }
await p.screenshot({ path: `${OUT}/compare_closed.png` })
writeFileSync(`${OUT}/compare_view.json`, JSON.stringify(out, null, 1)); console.log(JSON.stringify(out, null, 1))
await b.close()
