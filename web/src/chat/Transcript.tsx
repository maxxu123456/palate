import { useEffect, useRef } from "react"

import { ToolTrace, type ToolLine } from "./ToolTrace"

export interface Turn {
  id: number
  who: "you" | "palate"
  text: string
  tools: ToolLine[]
}

const NEAR = 80

/** The run as it reads: who spoke, what they said, what the tools did under it. */
export function Transcript({ turns, streaming }: { turns: Turn[]; streaming: boolean }) {
  const box = useRef<HTMLDivElement>(null)
  const tail = turns[turns.length - 1]

  useEffect(() => {
    const node = box.current
    if (node === null) return
    // Follow the tokens only while the reader is already at the foot of the transcript.
    if (node.scrollHeight - node.scrollTop - node.clientHeight < NEAR) {
      node.scrollTop = node.scrollHeight
    }
  }, [tail?.id, tail?.text])

  return (
    <div className="transcript" ref={box} aria-live="polite" aria-busy={streaming}>
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
