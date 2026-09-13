// The dev server proxies /api to the loopback server. Point VITE_PALATE_API at the
// server's own origin to skip the proxy.
const BASE = import.meta.env.VITE_PALATE_API ?? "/api"

export const POSTER_BASE = "https://image.tmdb.org/t/p/w92"

export interface FilmBrief {
  film_id: number
  title: string
  year: number | null
  directors: string[]
  countries: string[]
  original_language: string | null
  runtime: number | null
  in_watchlist: boolean
  hook: string | null
  poster_path: string | null
}

export interface EvidenceRow {
  kind: string
  text: string
  source_table: string
  source_id: string
  value: number | string | null
  span: [number, number] | null
}

export interface Recommended extends FilmBrief {
  score: number
  confidence: number
  mode_label: string | null
  feature_contributions: Record<string, number>
  evidence: EvidenceRow[]
}

export interface RecommendOut {
  films: Recommended[]
  pool_size: number
  condition: string
  profile_id: string
  degraded: string[]
  diagnostics: Record<string, unknown>
}

export interface FilmRecord {
  film_id: number
  title: string
  original_title: string | null
  year: number | null
  runtime: number | null
  original_language: string | null
  overview: string
  overview_offset: number
  overview_length: number
  tagline: string
  genres: string[]
  keywords: string[]
  countries: string[]
  directors: string[]
  writers: string[]
  cast: string[]
  collection: string | null
  vote_average: number | null
  vote_count: number
  your_rating: number | null
  watched_date: string | null
  in_watchlist: boolean
}

export interface ModeSummary {
  mode_id: number
  polarity: string
  label: string | null
  n_members: number
  mean_signal: number
  coherence: number
  confidence: number
  exemplar_film_ids: number[]
}

export interface EntitySummary {
  kind: string
  entity_id: string
  name: string
  n: number
  affinity: number
}

export interface TasteProfile {
  tier: string
  n_rated: number
  n_reliable_dated: number
  mean_rating: number
  rating_histogram: Record<string, number>
  modes: ModeSummary[]
  top_directors: EntitySummary[]
  bottom_directors: EntitySummary[]
  top_keywords: EntitySummary[]
  bottom_keywords: EntitySummary[]
  stale: boolean
}

export interface Health {
  ok: boolean
  chat: { provider: string; model: string; ok: boolean; detail: string }
  embeddings: { provider: string; ok: boolean; detail: string }
  index: { index_id: string; model: string; dim: number; vectors: number }
  profile: { profile_id?: string; tier?: string; n_rated?: number; stale?: boolean }
  corpus: { films: number; eligible: number; rated: number }
}

export interface AnsweredFilm {
  film_id: number
  title: string
  year: number | null
  runtime: number | null
  directors: string[]
  why: string
  evidence_refs: string[]
}

/** An answered film as a listing row. The reason takes the hook's place, nothing is invented. */
export function asBrief(film: AnsweredFilm): FilmBrief {
  return {
    film_id: film.film_id,
    title: film.title,
    year: film.year,
    directors: film.directors,
    countries: [],
    original_language: null,
    runtime: film.runtime,
    in_watchlist: false,
    hook: film.why,
    poster_path: null,
  }
}

export interface SseFrame {
  event: string
  data: Record<string, unknown>
}

export interface ChatAsk {
  message: string
  session_id?: string | null
}

async function failure(answer: Response): Promise<Error> {
  const body = (await answer.json().catch(() => null)) as { detail?: string } | null
  return new Error(body?.detail ?? `${answer.status} ${answer.statusText}`)
}

async function get<T>(path: string, params: Record<string, string | number> = {}): Promise<T> {
  const query = new URLSearchParams(
    Object.entries(params).map(([key, value]) => [key, String(value)]),
  )
  const suffix = query.toString() ? `?${query}` : ""
  const answer = await fetch(`${BASE}${path}${suffix}`)
  if (!answer.ok) throw await failure(answer)
  return (await answer.json()) as T
}

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const answer = await fetch(`${BASE}${path}`, {
    method,
    headers: body === undefined ? {} : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!answer.ok) throw await failure(answer)
  return (await answer.json()) as T
}

export const api = {
  health: () => get<Health>("/health"),
  profile: () => get<TasteProfile>("/profile"),
  film: (filmId: number) => get<FilmRecord>(`/films/${filmId}`),
  recommend: (query: string, limit = 12) =>
    send<RecommendOut>("/recommend", "POST", { search: { query, limit } }),
  cancel: (runId: string) => fetch(`${BASE}/chat/${runId}/cancel`, { method: "POST" }),
}

function parseBlock(block: string): SseFrame | null {
  let event = ""
  let data = ""
  for (const line of block.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7)
    else if (line.startsWith("data: ")) data += line.slice(6)
  }
  if (!event || !data) return null
  return { event, data: JSON.parse(data) as Record<string, unknown> }
}

/** Read one run as it happens. Server sent events over POST, so not EventSource. */
export async function streamChat(
  ask: ChatAsk,
  onFrame: (frame: SseFrame) => void,
  signal?: AbortSignal,
): Promise<void> {
  const answer = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(ask),
    signal,
  })
  if (!answer.ok || answer.body === null) throw await failure(answer)
  const reader = answer.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ""
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += value.replace(/\r\n/g, "\n")
    let cut = buffer.indexOf("\n\n")
    while (cut >= 0) {
      const frame = parseBlock(buffer.slice(0, cut))
      if (frame !== null) onFrame(frame)
      buffer = buffer.slice(cut + 2)
      cut = buffer.indexOf("\n\n")
    }
  }
}
