// UI audit: screenshots of every reachable screen and state, desktop (1440) and mobile (390).
//   node ../tests/threejs/ui_audit.mjs http://localhost:5173 <out_dir> <room_photo>
// Read only: clicks controls the way a user would; nothing is saved outside <out_dir>.
import { chromium } from 'playwright-core'
import { writeFileSync } from 'node:fs'

const [APP, OUT, PHOTO] = process.argv.slice(2)
const browser = await chromium.launch({ channel: 'chrome' })
const log = []

async function run(label, viewport) {
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: 1 })
  const p = await ctx.newPage()
  const errors = []
  p.on('pageerror', (e) => errors.push(e.message))
  p.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
  const shot = async (name, full = false) => {
    await p.waitForTimeout(400)
    await p.screenshot({ path: `${OUT}/${label}_${name}.png`, fullPage: full })
    log.push(`${label} ${name}`)
  }
  const tryStep = async (name, fn) => {
    try { await fn() } catch (e) { log.push(`${label} ${name} FAILED: ${String(e.message).split('\n')[0]}`) }
  }

  // ---- home / room picker
  // loading state: hold the demo catalogue for a moment
  await p.route('**/demo/catalogue.json', async (route) => { await new Promise((r) => setTimeout(r, 1500)); await route.continue() })
  await p.goto(APP)
  await shot('01_picker_loading')
  await p.waitForSelector('.room-card, .picker-empty:not(:has-text("Loading"))', { timeout: 20000 })
  await shot('02_picker', false)
  await shot('02_picker_full', true)
  await p.unroute('**/demo/catalogue.json')

  // ---- studio: upload the room (Clean Room reuses the cached result for this photo)
  await p.locator('input[type=file]').first().setInputFiles(PHOTO)
  await p.waitForSelector('.studio', { timeout: 60000 })
  await shot('03_studio_preparing')
  await p.waitForFunction(() => document.querySelectorAll('.marker').length >= 2, null, { timeout: 900000 })
  await shot('04_studio_before_apply')
  await shot('04_studio_before_apply_full', true)

  await p.locator('.marker[aria-label="Floor"]').click()
  await shot('05_floor_selected')
  await p.locator('button.tool.primary', { hasText: 'Apply' }).click()
  await tryStep('06_laying_tiles', async () => { await p.waitForSelector('.stage-busy', { timeout: 5000 }); await shot('06_laying_tiles') })
  await p.locator('.stage-result-actions button', { hasText: 'Download Image' }).waitFor({ timeout: 600000 })
  await shot('07_after_apply')
  await shot('07_after_apply_full', true)

  await tryStep('08_mask', async () => { await p.locator('.step-switch button', { hasText: 'Mask' }).click(); await p.waitForTimeout(1500); await shot('08_mask') })
  await tryStep('09_before_tab', async () => { await p.locator('.step-switch button', { hasText: 'Before' }).click(); await shot('09_before_tab') })
  await tryStep('10_compare', async () => { await p.locator('button.tool', { hasText: 'Compare' }).click(); await p.locator('.compare').waitFor(); await shot('10_compare') })
  await tryStep('10b_compare_close', async () => { await p.locator('.compare-bar button', { hasText: '✕' }).click() })
  await tryStep('11_3d', async () => { await p.locator('button.tool', { hasText: '3D View' }).click(); await p.waitForTimeout(2500); await shot('11_3d_view') })
  await tryStep('11b_3d_off', async () => { await p.locator('button.tool', { hasText: '3D View' }).click() })
  await tryStep('12_grout', async () => { await p.locator('button.tool', { hasText: 'Grout' }).click(); await shot('12_grout_popover') })
  await tryStep('13_layout', async () => { await p.locator('button.tool', { hasText: 'Layout' }).click(); await shot('13_layout_popover') })
  await tryStep('14_size', async () => { await p.locator('button.tool', { hasText: 'Room Size' }).click(); await shot('14_room_size_popover') })
  await tryStep('14b_close', async () => { await p.locator('button.tool', { hasText: 'Room Size' }).click() })
  await tryStep('15_saved', async () => {
    await p.locator('.stage-top button', { hasText: 'Add to Selection' }).click()
    await p.locator('.shelf-card button', { hasText: 'Save this room' }).click()
    await p.waitForTimeout(1500)
    await shot('15_saved_designs_rooms_full', true)
  })
  await tryStep('16_rail_grid', async () => { await p.locator('.view-toggle button[title="Grid"]').click(); await shot('16_rail_grid') })
  await tryStep('17_rail_empty', async () => { await p.locator('.rail-search input').fill('zzzz'); await shot('17_rail_empty_search'); await p.locator('.rail-search input').fill('') })
  await tryStep('18_hover_focus', async () => {
    await p.keyboard.press('Tab'); await p.keyboard.press('Tab'); await p.keyboard.press('Tab')
    await shot('18_keyboard_focus')
  })
  // error state: make the next compose fail
  await tryStep('19_error', async () => {
    await p.route('**/compose**', (route) => route.fulfill({ status: 500, body: 'audit: forced error' }))
    await p.locator('button.tool', { hasText: 'Grout' }).click()
    await p.locator('.popover-row button', { hasText: '8 mm' }).click()
    await p.waitForTimeout(4000)
    await shot('19_error_state')
    await p.unroute('**/compose**')
  })
  log.push(`${label} console/page errors: ${JSON.stringify(errors)}`)
  await ctx.close()
}

await run('desktop1440', { width: 1440, height: 900 })
await run('mobile390', { width: 390, height: 844 })
writeFileSync(`${OUT}/ui_audit_log.txt`, log.join('\n'))
console.log(log.join('\n'))
await browser.close()
