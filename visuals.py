import plotly.graph_objects as go

def get_hangman_svg(lives: int) -> str:
    """Return an SVG drawing of the gallows and hangman"""
    used = 6 - lives

    head = '<circle cx="140" cy="70" r="20" stroke="black" stroke-width="4" fill="none" />' if used >= 1 else ''
    body = '<line x1="140" y1="90" x2="140" y2="150" stroke="black" stroke-width="4" />' if used >= 2 else ''
    left_arm = '<line x1="140" y1="110" x2="110" y2="130" stroke="black" stroke-width="4" />' if used >= 3 else ''
    right_arm = '<line x1="140" y1="110" x2="170" y2="130" stroke="black" stroke-width="4" />' if used >= 4 else ''
    left_leg = '<line x1="140" y1="150" x2="115" y2="190" stroke="black" stroke-width="4" />' if used >= 5 else ''
    right_leg = '<line x1="140" y1="150" x2="165" y2="190" stroke="black" stroke-width="4" />' if used >= 6 else ''

    svg = f"""
    <svg width="260" height="240" xmlns="http://www.w3.org/2000/svg">
        <line x1="30" y1="220" x2="180" y2="220" stroke="black" stroke-width="4" />
        <line x1="60" y1="220" x2="60" y2="20" stroke="black" stroke-width="4" />
        <line x1="60" y1="20" x2="140" y2="20" stroke="black" stroke-width="4" />
        <line x1="140" y1="20" x2="140" y2="50" stroke="black" stroke-width="4" />
        {head}
        {body}
        {left_arm}
        {right_arm}
        {left_leg}
        {right_leg}
    </svg>
    """
    return svg

def build_pmf_figure(pmf_data: list[dict], letter: str):
    """Build a Plotly bar chart for the PMF of a guessed letter"""
    x = list(range(len(pmf_data)))
    y = [item["probability"] for item in pmf_data]
    hover_text = []
    for item in pmf_data:
        words_preview = ", ".join(item["words"][:10])
        if len(item["words"]) > 10:
            words_preview += ", ..."
        hover_text.append(
            f"Pattern: {item['pattern_spaced']}<br>"
            f"Probability: {item['probability']:.4f}<br>"
            f"Words in bucket: {item['count']}<br>"
            f"Examples: {words_preview}"
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        hovertext=hover_text,
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f'Post-guess PMF for "{letter}"',
        xaxis_title="Resulting clue pattern",
        yaxis_title="Probability",
        dragmode=False,
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )
    fig.update_xaxes(showticklabels=False, fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig
