"""Report rendering for bisection results."""

from __future__ import annotations

from .core import BisectResult


def render_text(result: BisectResult) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("nanbisect report")
    lines.append("=" * 64)
    lines.append(f"chunks            : {result.n_chunks} (good prefix: {result.good_prefix_len})")
    lines.append(f"metric(good)      : {result.m_good:.4f}")
    lines.append(f"metric(bad, full) : {result.m_bad:.4f}")
    lines.append(f"decision boundary : {result.boundary:.4f} (midpoint)")
    lines.append(f"training runs     : {result.train_runs}")
    lines.append("")
    lines.append("bisection log:")
    lines.append(f"  {'step':>4}  {'range':>11}  {'tested':>6}  {'metric':>8}  side   decision")
    for s in result.steps:
        lines.append(
            f"  {s.step:>4}  [{s.lo:>3},{s.hi:>3})   {s.mid:>6}  {s.metric:>8.4f}  "
            f"{s.side:<6} {s.decision}"
        )
    lines.append("")
    if result.verified and result.culprit_chunk is not None:
        lines.append(f"CULPRIT: chunk #{result.culprit_index} ({result.culprit_chunk})")
        lines.append("verified: prefix without it is good-side, with it is bad-side.")
    else:
        lines.append("INCONCLUSIVE: " + (result.inconclusive_reason or "unknown"))
    lines.append("=" * 64)
    return "\n".join(lines)


def render_markdown(result: BisectResult) -> str:
    lines = []
    lines.append("# nanbisect report")
    lines.append("")
    lines.append(f"- chunks: {result.n_chunks} (good prefix: {result.good_prefix_len})")
    lines.append(f"- metric(good prefix): `{result.m_good:.4f}`")
    lines.append(f"- metric(full data): `{result.m_bad:.4f}`")
    lines.append(f"- decision boundary: `{result.boundary:.4f}` (midpoint)")
    lines.append(f"- training runs used: {result.train_runs}")
    lines.append("")
    lines.append("## Bisection log")
    lines.append("")
    lines.append("| step | range | prefix tested | metric | side | decision |")
    lines.append("| ---: | :---: | ---: | ---: | :--- | :--- |")
    for s in result.steps:
        lines.append(
            f"| {s.step} | [{s.lo},{s.hi}) | {s.mid} | {s.metric:.4f} | "
            f"{s.side} | {s.decision} |"
        )
    lines.append("")
    if result.verified and result.culprit_chunk is not None:
        lines.append(f"## Culprit: chunk #{result.culprit_index} (`{result.culprit_chunk}`)")
        lines.append("")
        lines.append("Verification bracket passed: the prefix without this chunk scores "
                     "good-side, the prefix with it scores bad-side.")
    else:
        lines.append("## Inconclusive")
        lines.append("")
        lines.append(result.inconclusive_reason or "unknown reason")
    return "\n".join(lines)
