import { Fragment, useEffect, useState } from "react"
import { api, type EvidenceRow, type FilmRecord, type Recommended } from "../api/client"

function Facts({ film }: { film: FilmRecord }) {
  const rows: [string, string][] = [
    ["Directed by", film.directors.join(", ")],
    ["Written by", film.writers.join(", ")],
    ["With", film.cast.slice(0, 5).join(", ")],
    ["Country", film.countries.join(", ")],
    ["Language", film.original_language ?? ""],
    ["Runtime", film.runtime === null ? "" : `${film.runtime} minutes`],
    ["Genres", film.genres.join(", ")],
    ["Keywords", film.keywords.slice(0, 10).join(", ")],
  ]
  return (
    <dl className="detail__facts">
      {rows
        .filter(([, value]) => value !== "")
        .map(([label, value]) => (
          <Fragment key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </Fragment>
        ))}
    </dl>
  )
}

function Contributions({ shares }: { shares: Record<string, number> }) {
  const rows = Object.entries(shares)
    .filter(([, value]) => value !== 0)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 6)
  if (rows.length === 0) return null
  const widest = Math.max(...rows.map(([, value]) => Math.abs(value)))
  return (
    <ul className="bars">
      {rows.map(([name, value]) => {
        // Signed, so the bar grows either side of the midline rather than needing a legend.
        const share = (Math.abs(value) / widest) * 50
        return (
          <li key={name}>
            <span>{name}</span>
            <span className="bar" data-sign={value < 0 ? "down" : "up"}>
              <span style={{ left: `${value < 0 ? 50 - share : 50}%`, width: `${share}%` }} />
            </span>
            <span className="num v">{value.toFixed(2)}</span>
          </li>
        )
      })}
    </ul>
  )
}

function Evidence({ rows }: { rows: EvidenceRow[] }) {
  if (rows.length === 0) return null
  return (
    <ul className="pairs">
      {rows.map((row, index) => (
        <li key={`${row.source_table}-${row.source_id}-${index}`}>
          <span>{row.text}</span>
          <span className="n">{row.kind}</span>
          <span className="v">{row.source_table}</span>
        </li>
      ))}
    </ul>
  )
}

export function FilmDetail({ filmId, ranked }: { filmId: number; ranked?: Recommended }) {
  const [film, setFilm] = useState<FilmRecord | null>(null)
  const [problem, setProblem] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    setFilm(null)
    setProblem(null)
    api
      .film(filmId)
      .then((found) => {
        if (live) setFilm(found)
      })
      .catch((error: Error) => {
        if (live) setProblem(error.message)
      })
    return () => {
      live = false
    }
  }, [filmId])

  if (problem !== null) return <div className="detail">{problem}</div>
  if (film === null) return <div className="detail">Reading the record.</div>

  return (
    <div className="detail">
      {film.overview === "" ? null : <p>{film.overview}</p>}
      {film.your_rating === null ? null : (
        <p className="mine num">You rated this {film.your_rating.toFixed(1)}</p>
      )}
      <Facts film={film} />
      {ranked === undefined ? null : (
        <>
          <Contributions shares={ranked.feature_contributions} />
          <Evidence rows={ranked.evidence} />
        </>
      )}
    </div>
  )
}
