"""One look for every figure: journal-sized type, thin lines, and each lab's own colour
(Claude models darker for newer generations, open-weight models grey). Checked for
colour-blind separation; every model is also named on the chart."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

INK = "#1D2433"
MUTED = "#6B7385"
GRID = "#ECEEF2"
GREY = "#A3AAB8"
LIGHT_GREY = "#D3D7DF"
ACCENT = "#D97757"  # Anthropic
ACCENT_TEXT = "#B4532F"  # the same hue, dark enough for text
OPENAI = "#4B4B55"
GPT = {"GPT-6.2 Sol": "#18181C", "GPT-6.2 Astra": "#33333B", "GPT-6.1 Sol": OPENAI, "GPT-6 Astra": "#63636E", "GPT-5.5": "#7C7C87"}  # newer darker
GOOGLE = "#4285F4"
PROTEIN_LM = "#9B59B6"  # protein language models, drawn as diamonds
CLAUDE = {  # older generations lighter
    "Claude Sonnet 4.5": "#F6DFD4", "Claude Opus 4.5": "#F3D3C4", "Claude Opus 4.6": "#EFC1AB",
    "Claude Opus 4.7": "#EAAD92", "Claude Opus 4.8": "#E59C7D", "Claude Opus 5": "#DF8A68", "Claude Opus 5.5": ACCENT, "Claude Fable 5.1": "#C0694C", "Claude Fable 5.2": "#9C3A1C",
}

for path in Path("/usr/share/fonts").rglob("LiberationSans-*.ttf"):
    fm.fontManager.addfont(str(path))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK, "xtick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def color(model: str) -> str:
    """By display name: Claude by generation, GPT charcoal, Gemini blue, open-weight grey."""
    if model.startswith("GPT-"):
        return GPT.get(model, OPENAI)
    if model.startswith("Gemini"):
        return GOOGLE
    if model.startswith(("ProtGPT", "ProGen", "ESM")):
        return PROTEIN_LM
    return CLAUDE.get(model, GREY)


def header(fig, ax, title: str, subtitle: str):
    """Bold one-line finding above the axes; the grey note saying exactly what is plotted goes
    below the axes, right-aligned, like a caption."""
    fig.canvas.draw()
    h = fig.get_figheight() * 72  # points
    fig.text(0.0, ax.get_position().y1 + 10 / h, title, ha="left", va="bottom", fontsize=8.5, fontweight="bold", color=INK)
    box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(box.x1, box.y0 - 6 / h, subtitle, ha="right", va="top", fontsize=6.2, color=MUTED, linespacing=1.35,
             multialignment="right")


# Web figures follow the page's light and dark themes: every fixed colour below becomes a CSS
# variable (the light value is the fallback). Dark values reverse the Claude ramp, so newer models
# are brighter on a dark page, and lighten the GPT greys, which would vanish on it.
THEME = {  # light hex: (variable, dark value)
    "#1d2433": ("ink", "#e6e8ee"), "#6b7385": ("muted", "#a0a8b8"), "#eceef2": ("grid", "#30353f"),
    "#a3aab8": ("grey", "#7d8596"), "#d3d7df": ("light", "#4a505c"), "#ffffff": ("bg", "var(--global-bg-color, #1c1c1d)"),
    "#b4532f": ("accent-text", "#eda48a"),
    "#18181c": ("gpt-62s", "#f2f2f6"), "#33333b": ("gpt-62a", "#d4d4dc"), "#4b4b55": ("gpt-61", "#c2c2cc"),
    "#63636e": ("gpt-6", "#a9a9b4"), "#7c7c87": ("gpt-55", "#8e8e9a"),
    "#f6dfd4": ("c-s45", "#5e3e31"), "#f3d3c4": ("c-o45", "#6a4535"), "#efc1ab": ("c-o46", "#7d4f3b"),
    "#eaad92": ("c-o47", "#925a41"), "#e59c7d": ("c-o48", "#a86548"), "#df8a68": ("c-o5", "#bf6f4e"),
    "#c0694c": ("c-f51", "#e8906c"), "#9c3a1c": ("c-f52", "#ffb08a"),
    "#e6e9ef": ("bar-50", "#353a45"), "#c5cbd6": ("bar-100", "#555c6a"), "#8b93a3": ("bar-200", "#8b93a3"),
}
WEB_FONT = "Arial, 'Liberation Sans', Helvetica, sans-serif"  # Liberation Sans is metric-compatible with Arial


def theme_css() -> str:
    """The variables, for the page that shows the web figures."""
    light = "; ".join(f"--fig-{v}: {k}" for k, (v, _) in THEME.items())
    dark = "; ".join(f"--fig-{v}: {d}" for _, (v, d) in THEME.items())
    return f"d-article {{ {light}; }}\nhtml[data-theme='dark'] d-article {{ {dark}; }}"


def web_svg(text: str, stem: str) -> str:
    """matplotlib SVG -> an inline-ready SVG: themed colours, unique ids, no global styles."""
    import re

    text = text[text.index("<svg"):]
    text = re.sub(r'(<svg[^>]*?) width="[^"]*" height="[^"]*"', rf'\1 id="{stem}" width="100%"', text, count=1)
    text = re.sub(r'id="([^"]+)"', lambda m: m.group(0) if m.group(1) == stem else f'id="{stem}-{m.group(1)}"', text)
    text = re.sub(r'(url\(#|href="#)([^)"]+)', rf"\1{stem}-\2", text)
    text = text.replace("*{", f"#{stem} *{{")  # matplotlib's global stroke style, scoped to this figure
    text = re.sub(r"font-family: ?[^;\"]+", f"font-family: {WEB_FONT}", text)
    text = re.sub(r"(fill|stroke): ?(#[0-9a-fA-F]{6})",
                  lambda m: f"{m.group(1)}: var(--fig-{THEME[m.group(2).lower()][0]}, {m.group(2)})" if m.group(2).lower() in THEME else m.group(0), text)
    return text


def save(fig, stem: str):
    Path("figures/web").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/{stem}.{ext}", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    raw = Path(f"figures/web/{stem}.raw.svg")
    plt.rcParams["svg.hashsalt"] = stem  # stable ids
    fig.savefig(raw, bbox_inches="tight", pad_inches=0.04, facecolor="none")
    Path(f"figures/web/{stem}.svg").write_text(web_svg(raw.read_text(), stem))
    raw.unlink()
