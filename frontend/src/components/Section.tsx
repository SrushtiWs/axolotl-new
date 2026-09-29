import type { ReactNode } from 'react'

interface SectionProps {
  title: string
  hint?: string
  required?: boolean
  children: ReactNode
}

export function Section({ title, hint, required, children }: SectionProps) {
  return (
    <section className="section">
      <header className="section-head">
        <h2 className="section-title">
          {title}
          {required && <span className="required" aria-label="required"> *</span>}
        </h2>
        {hint && <p className="section-hint">{hint}</p>}
      </header>
      {children}
    </section>
  )
}
