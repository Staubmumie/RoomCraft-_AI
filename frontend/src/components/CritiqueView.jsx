// A critique of the Interior Critic: summary, scores, strengths, issues and suggestions.
// Used for a standalone room critique (Design tab) and for every critique inside a run (Overview).

export const SCORE_LABELS = {
  functionality: 'Function', aesthetics: 'Aesthetics', space_utilization: 'Use of space', preference_fit: 'Fits the request'
};
const SEVERITY_ORDER = { high: 0, medium: 1, low: 2 };

function ScoreBar({ label, score }) {
  return (
    <div className={`score score-${score}`} title={`${label}: ${score} of 5`}>
      <span className="score-label">{label}</span>
      <span className="score-track" aria-hidden="true">
        {[1, 2, 3, 4, 5].map((n) => <span key={n} className={n <= score ? 'on' : ''} />)}
      </span>
      <span className="score-value mono">{score}/5</span>
    </div>
  );
}

export default function CritiqueView({ critique, compact = false }) {
  const issues = [...(critique.issues ?? [])].sort(
    (a, b) => (SEVERITY_ORDER[a.severity] ?? 3) - (SEVERITY_ORDER[b.severity] ?? 3));
  const strengths = critique.strengths ?? [];
  const suggestions = critique.suggestions ?? [];
  return (
    <div className={`critique ${compact ? 'compact' : ''}`}>
      {critique.summary && <p className="critique-summary">{critique.summary}</p>}
      <div className="scores">
        {Object.entries(critique.scores ?? {}).map(([name, score]) => (
          <ScoreBar key={name} label={SCORE_LABELS[name] ?? name} score={score} />
        ))}
      </div>

      {strengths.length > 0 && (
        <div className="critique-block">
          <h4>Strengths</h4>
          <ul className="critique-list strengths">
            {strengths.map((s, n) => <li key={n}>{s}</li>)}
          </ul>
        </div>
      )}

      {issues.length > 0 && (
        <div className="critique-block">
          <h4>Issues</h4>
          <ul className="critique-list issues">
            {issues.map((issue, n) => (
              <li key={n} className={`severity-${issue.severity}`}>
                <span className="issue-tags">
                  <span className={`badge severity ${issue.severity}`}>{issue.severity}</span>
                  {issue.aspect && <span className="badge">{SCORE_LABELS[issue.aspect] ?? issue.aspect}</span>}
                </span>
                <span>{issue.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {suggestions.length > 0 && (
        <div className="critique-block">
          <h4>Suggestions</h4>
          <ol className="critique-list suggestions">
            {suggestions.map((s, n) => <li key={n}>{s}</li>)}
          </ol>
        </div>
      )}
    </div>
  );
}
