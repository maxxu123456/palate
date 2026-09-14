export interface ToolLine {
  callId: string
  text: string
}

// TODO: fold repeated calls into one line that expands on click. One row per call is
// why the panel is still behind the flag.
export function ToolTrace({ lines }: { lines: ToolLine[] }) {
  if (lines.length === 0) return null
  return (
    <ul className="traces">
      {lines.map((line) => (
        <li className="trace" key={line.callId}>
          {line.text}
        </li>
      ))}
    </ul>
  )
}
