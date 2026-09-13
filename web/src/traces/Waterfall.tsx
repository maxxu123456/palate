import { useEffect, useState } from "react"
import { api, type TraceDetail, type TraceRun, type TraceSpan } from "../api/client"

function ms(value: number | null): string {
  return value === null ? "" : value.toFixed(0)
}

function Spans({ spans }: { spans: TraceSpan[] }) {
  const longest = Math.max(...spans.map((span) => span.latency_ms ?? 0), 1)
  return (
    <ul className="waterfall">
      {spans.map((span) => (
        <li key={span.span_id} data-status={span.status}>
          <span className="waterfall__kind">{span.kind}</span>
          <span style={{ paddingLeft: `${span.depth * 10}px` }}>
            {span.name}
            {span.error_type === null ? "" : ` ${span.error_type}`}
          </span>
          <span className="waterfall__bar">
            <span style={{ width: `${((span.latency_ms ?? 0) / longest) * 100}%` }} />
          </span>
          <span className="waterfall__ms num">{ms(span.latency_ms)}</span>
        </li>
      ))}
    </ul>
  )
}

function Claims({ rows }: { rows: Record<string, unknown>[] }) {
  if (rows.length === 0) return null
  return (
    <section className="section">
      <h2>Claims checked</h2>
      <ul className="pairs">
        {rows.map((row) => (
          <li key={String(row.claim_id)}>
            <span>{String(row.sentence)}</span>
            <span className="n">{String(row.claim_kind)}</span>
            <span className="v">{row.supported === 1 ? String(row.method) : "unsupported"}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

export function Waterfall() {
  const [runs, setRuns] = useState<TraceRun[]>([])
  const [open, setOpen] = useState<TraceDetail | null>(null)
  const [problem, setProblem] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    api
      .traces()
      .then((found) => {
        if (live) setRuns(found)
      })
      .catch((error: Error) => {
        if (live) setProblem(error.message)
      })
    return () => {
      live = false
    }
  }, [])

  function show(runId: string) {
    if (open?.run_id === runId) {
      setOpen(null)
      return
    }
    api.trace(runId).then(setOpen).catch((error: Error) => setProblem(error.message))
  }

  if (problem !== null) return <p className="notice">{problem}</p>
  if (runs.length === 0) return <p className="empty">No runs traced in the last day.</p>

  return (
    <div>
      <ol className="listing">
        {runs.map((run) => (
          <li key={run.run_id}>
            <div className="row" data-open={open?.run_id === run.run_id}>
              <button className="row__line" onClick={() => show(run.run_id)}>
                <span />
                <span className="row__body">
                  <span className="row__title narrow">{run.kind}</span>
                  <span className="row__meta num">
                    <span>{run.started_at.slice(11, 19)}</span>
                    <span>{run.turns} turns</span>
                    <span>
                      {run.tokens_in} in {run.tokens_out} out
                    </span>
                    <span>{run.status}</span>
                  </span>
                </span>
                <span className="row__score num">
                  {run.cost_usd.toFixed(4)}
                  {run.cost_complete ? "" : "?"}
                </span>
              </button>
              {open?.run_id === run.run_id ? (
                <div>
                  <Spans spans={open.spans} />
                  <Claims rows={open.claims} />
                </div>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
