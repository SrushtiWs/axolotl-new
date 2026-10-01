import { chromium } from 'playwright-core'
const [APP, PHOTO, OUT] = process.argv.slice(2)
const b = await chromium.launch({ channel: 'chrome' }); const p = await b.newPage({ viewport: { width: 1500, height: 950 } })
await p.goto(APP); await p.locator('input[type=file]').first().setInputFiles(PHOTO)
await p.waitForSelector('.stage-download', { timeout: 60000 }); await p.waitForTimeout(1500)
console.log(await p.locator('.stage-download').evaluate((e) => { const s = getComputedStyle(e); return JSON.stringify({ text: e.innerText, color: s.color, bg: s.backgroundColor, font: s.fontSize, opacity: s.opacity, disabled: e.disabled, width: e.getBoundingClientRect().width }) }))
await p.mouse.move(5, 5); await p.waitForTimeout(300)
await p.locator('.stage-download').screenshot({ path: `${OUT}/dl_btn.png` })
await b.close()
