import type { ReactNode } from "react"

import { POSTER_BASE, type FilmBrief, type Recommended } from "../api/client"

function minutes(runtime: number | null): string {
  return runtime === null ? "" : String(runtime)
}

function predicted(film: FilmBrief | Recommended): string {
  return "score" in film ? film.score.toFixed(2) : ""
}

export function Poster({ film }: { film: FilmBrief }) {
  if (film.poster_path === null) return <span className="row__art" aria-hidden="true" />
  return <img className="row__art" src={`${POSTER_BASE}${film.poster_path}`} alt="" loading="lazy" />
}

export function FilmRow({
  film,
  open,
  onToggle,
  children,
}: {
  film: FilmBrief | Recommended
  open: boolean
  onToggle: () => void
  children?: ReactNode
}) {
  return (
    <div className="row" data-open={open}>
      <button className="row__line" onClick={onToggle} aria-expanded={open}>
        <Poster film={film} />
        <span className="row__body">
          <span className="row__title narrow">{film.title}</span>
          <span className="row__meta num">
            <span>{film.year ?? ""}</span>
            <span>{film.directors[0] ?? ""}</span>
            <span>{minutes(film.runtime)}</span>
            {film.in_watchlist ? <span>on your watchlist</span> : null}
          </span>
          {film.hook === null ? null : <span className="row__hook">{film.hook}</span>}
        </span>
        <span className="row__score num">{predicted(film)}</span>
      </button>
      {open ? children : null}
    </div>
  )
}
