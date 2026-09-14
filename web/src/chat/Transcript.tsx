import { ToolTrace, type ToolLine } from "./ToolTrace"

export interface Turn {
  id: number
  who: "you" | "palate"
  text: string
  tools: ToolLine[]
}

/** The run as it reads: who spoke, what they said, what the tools did under it. */
export function Transcript({ turns }: { turns: Turn[] }) {
  return (
    <div className="transcript">
      {turns.length === 0 ? <p className="empty">Ask for what you are in the mood for.</p> : null}
      {turns.map((turn) => (
        <div className={`turn turn--${turn.who}`} key={turn.id}>
          <div className="turn__who">{turn.who}</div>
          <div className="turn__body">{turn.text}</div>
          <ToolTrace lines={turn.tools} />
        </div>
      ))}
    </div>
  )
}
