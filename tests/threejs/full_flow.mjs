// 11c: the full user flow in Chrome, per room photo:
//   upload -> Clean Room -> select Floor + walls -> tiles render -> compose ->
//   change the tile (no new detection) -> 3D View on.
//   node full_flow.mjs <app url> <api port> <shots dir> <photo> [<photo> ...]
import { chromium } from 'playwright-core'
const [APP, API_PORT, SHOTS, ...PHOTOS] = process.argv.slice(2)
const browser = await chromium.launch({ channel: 'chrome', args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const summary = []
for (const photo of PHOTOS) {
  const name = photo.split('/').pop()
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } })
  const errors = [], posts = [], failed = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('response', (r) => {
    if (!r.url().includes(`:${API_PORT}/`) || r.request().method() !== 'POST') return
    const path = new URL(r.url()).pathname
    posts.push(path)
    if (!r.ok()) failed.push(`${path} ${r.status()}`)
  })
  const idle = async () => {
    await page.waitForTimeout(1500)
    await page.waitForFunction(() => !document.querySelector('.marker[aria-busy="true"]'), null, { timeout: 300000 })
    await page.waitForTimeout(1500)
  }
  const row = { photo: name }
  try {
    const t0 = Date.now()
    await page.goto(APP)
    await page.locator('input[type=file]').first().setInputFiles(photo)
    await page.waitForFunction(() => document.querySelectorAll('.marker').length >= 1, null, { timeout: 600000 })
    row.clean_room_s = Math.round((Date.now() - t0) / 1000)
    const labels = await page.evaluate(() => [...document.querySelectorAll('.marker')].map((m) => m.getAttribute('aria-label')))
    row.surfaces = labels.length
    const pick = ['Floor', ...labels.filter((l) => l !== 'Floor').slice(0, 2)].filter((l) => labels.includes(l))
    for (const l of pick) { await page.locator(`.marker[aria-label="${l}"]`).click(); await idle() }
    const tiled = await page.evaluate(() => [...document.querySelectorAll('.marker')].map((m) => [m.getAttribute('aria-label'), m.getAttribute('aria-checked'), m.getAttribute('title') || '']))
    row.selected = tiled.filter((t) => t[1] === 'true').map((t) => t[0])
    row.refused = tiled.filter((t) => /could not be tiled/.test(t[2])).map((t) => `${t[0]}: ${t[2].replace(/^.*could not be tiled: /, '').slice(0, 70)}`)
    row.composed = await page.evaluate(() => [...document.querySelectorAll('img')].some((i) => /composed_/.test(i.src)))
    await page.screenshot({ path: `${SHOTS}/flow_${name}_2d.png` })
    // Tile change: another catalogue tile -> re-render only, no detection.
    const before = posts.length
    const activeBefore = await page.locator('.rail-list button.product.active').innerText()
    await page.locator('.rail-list button.product:not(.active)').first().click()   // throws if absent
    row.tile_changed_to = (await page.locator('.rail-list button.product.active').innerText()).split('\n')[0]
    if (row.tile_changed_to === activeBefore.split('\n')[0]) throw new Error('tile did not change')
    await page.waitForTimeout(1000); await idle()
    const after = posts.slice(before)
    row.tile_change_posts = [...new Set(after)]
    row.tile_change_detection = after.filter((p) => /segment|surfaces|floor-wall/.test(p)).length
    // 3D View.
    await page.locator('button', { hasText: '3D View' }).click()
    await page.waitForTimeout(5000)
    row.view3d = await page.evaluate(() => {
      const layer = document.querySelector('.three-layer')
      const note = [...document.querySelectorAll('*')].map((e) => e.textContent).find((t) => t && t.startsWith('3D view unavailable') && t.length < 400)
      return note ? `unavailable: ${note}` : (layer && getComputedStyle(layer).visibility === 'visible' ? 'shown' : 'not shown')
    })
    await page.screenshot({ path: `${SHOTS}/flow_${name}_3d.png` })
  } catch (e) {
    row.error = String(e.message || e).split('\n')[0]
  }
  row.http_errors = failed
  row.page_errors = errors
  summary.push(row)
  console.log(JSON.stringify(row))
  await page.close()
}
await browser.close()
