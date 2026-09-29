import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'
// Loaded second so the showroom's light theme overrides the older dark tokens.
import './showroom.css'

const container = document.getElementById('root')

if (!container) {
  throw new Error('Root element #root not found in index.html')
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
