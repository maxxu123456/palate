// The server's event union, mirrored. A parity test compares the member names on both
// sides, so adding an event on the server without handling it here fails CI.

export interface RunStarted {
  type: "run.started"
  run_id: string
  session_id: string
  model: string
  budget: Record<string, number>
}

export interface TurnStarted {
  type: "turn.started"
  turn: number
  tools_offered: string[]
}

export interface TextDelta {
  type: "text.delta"
  text: string
  channel: "thinking" | "answer"
}

export interface ToolCallProposed {
  type: "tool.proposed"
  call_id: string
  name: string
  arguments: Record<string, unknown>
  source: "field" | "content"
}

export interface ToolCallStarted {
  type: "tool.started"
  call_id: string
  name: string
}

export interface ToolCallFinished {
  type: "tool.finished"
  call_id: string
  name: string
  ok: boolean
  summary: string
  meta: Record<string, unknown>
  latency_ms: number
  undo_token: string | null
}

export interface PreferenceRecorded {
  type: "preference.recorded"
  pref_id: number
  label: string
  polarity: string
  hardness: string
  affected_films: number
  undo_token: string
}

export interface GroundingChecked {
  type: "grounding.checked"
  grounded_ratio: number
  nli_available: boolean
  unsupported: string[]
  dropped_film_ids: number[]
}

export interface Recommendations {
  type: "recommendations"
  films: AnsweredFilm[]
  prose: string
}

export interface RunFinished {
  type: "run.finished"
  stop_reason: string
  turns: number
  tool_calls: number
  input_tokens: number
  output_tokens: number
  cost_usd: number
  wall_ms: number
}

export interface RunFailed {
  type: "run.failed"
  error_code: string
  message: string
  stop_reason: string
}

export type AgentEvent =
  | RunStarted
  | TurnStarted
  | TextDelta
  | ToolCallProposed
  | ToolCallStarted
  | ToolCallFinished
  | PreferenceRecorded
  | GroundingChecked
  | Recommendations
  | RunFinished
  | RunFailed

export interface AnsweredFilm {
  film_id: number
  title: string
  year: number | null
  runtime: number | null
  directors: string[]
  why: string
  evidence_refs: string[]
}

const TAGS: ReadonlySet<string> = new Set<AgentEvent["type"]>([
  "run.started",
  "turn.started",
  "text.delta",
  "tool.proposed",
  "tool.started",
  "tool.finished",
  "preference.recorded",
  "grounding.checked",
  "recommendations",
  "run.finished",
  "run.failed",
])

/** One stream frame as a typed event, or null for a tag this build does not know. */
export function asEvent(tag: string, data: Record<string, unknown>): AgentEvent | null {
  if (!TAGS.has(tag) || data.type !== tag) return null
  return data as unknown as AgentEvent
}
