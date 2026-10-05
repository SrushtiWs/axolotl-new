// Open a room in the real app; count dots, collect page errors, screenshot.
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'
const [APP, OUT, PHOTO] = process.argv.slice(2)
const b = await chromium.launch({ channel: 'chrome' })
const p = await (await b.newContext({ viewport: { width: 1500, height: 950 } })).newPage()
const errors = []; p.on('pageerror', (e) => errors.push(e.message)); p.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
const t = Date.now()
await p.goto(APP)
await p.locator('input[type=file]').first().setInputFiles(PHOTO)
await p.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 600000 })
const out = { seconds: (Date.now() - t) / 1000, markers: await p.locator('.marker').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label'))), errors }
await p.screenshot({ path: `${OUT}/open_room.png` })
writeFileSync(`${OUT}/open_room.json`, JSON.stringify(out, null, 2)); console.log(JSON.stringify(out))
await b.close()
