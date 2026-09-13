import { Fragment, useEffect, useState } from "react"
import { api, type EntitySummary, type ModeSummary, type TasteProfile } from "../api/client"

const TIERS: Record<string, string> = {
  cold: "too few ratings to rank without a query",
  thin: "enough ratings for a direction, not for the ridge",
  full: "every part of the model is fitted",
}

function Modes({ title, rows }: { title: string; rows: ModeSummary[] }) {
  if (rows.length === 0) return null
  return (
    <section className="section">
      <h2>{title}</h2>
      <ul className="pairs">
        {rows.map((mode) => (
          <li key={`${mode.polarity}-${mode.mode_id}`}>
            <span>{mode.label ?? `mode ${mode.mode_id}`}</span>
            <span className="n num">{mode.n_members}</span>
            <span className="v num">{mode.coherence.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Entities({ title, rows }: { title: string; rows: EntitySummary[] }) {
  if (rows.length === 0) return null
  return (
    <section className="section">
      <h2>{title}</h2>
      <ul className="pairs">
        {rows.map((row) => (
          <li key={`${row.kind}-${row.entity_id}`}>
            <span>{row.name}</span>
            <span className="n num">{row.n}</span>
            <span className="v num">{row.affinity.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

export function ProfilePanel() {
  const [profile, setProfile] = useState<TasteProfile | null>(null)
  const [problem, setProblem] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    api
      .profile()
      .then((found) => {
        if (live) setProfile(found)
      })
      .catch((error: Error) => {
        if (live) setProblem(error.message)
      })
    return () => {
      live = false
    }
  }, [])

  if (problem !== null) return <p className="notice">{problem}</p>
  if (profile === null) return <p className="empty">Reading the profile.</p>

  return (
    <div>
      <section className="section">
        <h2>Fit</h2>
        <dl className="detail__facts">
          {(
            [
              ["Tier", profile.tier],
              ["Ratings", String(profile.n_rated)],
              ["Dated ratings", String(profile.n_reliable_dated)],
              ["Mean rating", profile.mean_rating.toFixed(2)],
            ] as [string, string][]
          ).map(([label, value]) => (
            <Fragment key={label}>
              <dt>{label}</dt>
              <dd className="num">{value}</dd>
            </Fragment>
          ))}
        </dl>
        <p className="empty">
          {TIERS[profile.tier] ?? ""}
          {profile.stale ? " This profile was fitted in another vector space." : ""}
        </p>
      </section>
      <Modes
        title="Modes you return to"
        rows={profile.modes.filter((mode) => mode.polarity === "like")}
      />
      <Modes
        title="Modes you avoid"
        rows={profile.modes.filter((mode) => mode.polarity === "dislike")}
      />
      <Entities title="Directors you return to" rows={profile.top_directors} />
      <Entities title="Directors you do not" rows={profile.bottom_directors} />
      <Entities title="Keywords you return to" rows={profile.top_keywords} />
      <Entities title="Keywords you do not" rows={profile.bottom_keywords} />
    </div>
  )
}
