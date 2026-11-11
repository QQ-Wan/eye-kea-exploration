"""
Xinger style — a matplotlib reproduction of the R/cowplot `theme_half_open()` +
`theme_custom()` look (from the dp3r4 OSF figures), for consistent, clean
publication figures.

--------------------------------------------------------------------------
FONT / SIZE CONTRACT  (all sizes are RATIOS keyed to the panel, never fixed pt)
--------------------------------------------------------------------------
The one fixed thing is the ratio of the axis-label font size to the PANEL
(subplot) height, so a panel looks identical whether it is 3.45in or 5.48in
tall and no matter how many columns/rows the figure has.

  ANCHOR:  axis-label = 20 pt on a 3.45 in panel  ->  ratio ~= 5.8 pt / inch.
           A single panel's minimum height is 3.45 in (=> 20 pt label).
           scale = panel_height_in / 3.45   (pass this to apply_xinger / helpers)

Everything else is a ratio of the axis-label size L (= 20*scale):

  * plot title      = 1.00 * L,  NOT italic, NOT bold          (rule 1, 2)
  * panel letter    = 1.00 * L,  bold        (a./b./c. default) (rule 2)
  * axis-tick labels= 0.80 * L,  (smaller than the axis label)  (rule 3)
  * tick->label pad = 0.35 * L pt (narrow, never overlapping)   (rule 5)
  * legend / stats  = 0.80 * L (fixed; = tick size), placed     (rule 6,7,8)
                      INSIDE the panel, never overlapping data.

Use the helpers below rather than raw matplotlib calls so the contract holds:
    xinger_axes(ax, scale)          # spines/ticks + tick-label size + labelpad
    add_title(ax, text, scale)      # non-italic title at L
    add_panel_label(ax, "a", scale) # bold letter at L (lowercase default)
    align_panel_labels(pairs, scale)# multi-panel: align letters by row/col (rule 10)
    place_legend(ax, scale, ...)    # inside, fixed 0.80*L, non-overlapping
    place_stats(ax, text, scale)    # inside, fixed 0.80*L, non-overlapping

Usage:
    from plot_style import (apply_xinger, PALETTE, CYCLE, xinger_axes,
                              add_title, add_panel_label, place_legend,
                              place_stats, save_xinger, scale_for_panel)
    scale = scale_for_panel(panel_height_in)   # or panel_height_in/3.45
    apply_xinger(scale=scale)
    fig, ax = plt.subplots(...)
    ax.plot(x, y, color=PALETTE["blue"])       # NO ax.set_title
    xinger_axes(ax, scale)
    add_title(ax, "Feature-based main effects", scale)
    add_panel_label(ax, "a", scale)
    place_legend(ax, scale)
    save_xinger(fig, "/path/name")             # .png + .pdf, transparent
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---- muted Xinger palette ----
PALETTE = {
    "blue":  "#7887A4",
    "orange": "#F6955E",
    "mauve": "#9B6981",
    "tan":   "#A89F8E",
    "green": "#6E8B6B",   # muted green, palette-consistent extension
    "gold":  "#C9A24B",   # muted gold, palette-consistent extension
    "grey":  "#737373",
    "lightgrey": "#C8C8C8",
    "black": "#000000",
}
# ordered categorical cycle for multiple series (distinct hues, NOT a gradient)
CYCLE = [PALETTE["blue"], PALETTE["orange"], PALETTE["mauve"],
         PALETTE["green"], PALETTE["gold"], PALETTE["tan"]]

# ---- THE FIXED RATIO SYSTEM (see module docstring) ----
REF_PANEL_IN = 3.45          # reference panel height (single-panel minimum)
AXIS_LABEL_PT = 20.0         # axis-label size AT the reference panel
# ratios of every other element to the axis-label size L:
R_TITLE = 1.00               # plot title   (not italic, not bold)
R_PANEL = 1.00               # panel letter (bold)
R_TICK = 0.80                # axis-tick (number) labels  -> smaller than label
R_LABELPAD = 0.35            # tick-label -> axis-label gap, in pt, as frac of L
# In-plot text (legend + stats) is capped at 80% of the axis-label size, which
# equals the tick-label size. It is used at exactly this size (no shrink
# ladder): the max IS the floor.
R_LEGEND_CAP = 0.80          # legend/stats size = 0.80 * L (= tick size)


def scale_for_panel(panel_height_in: float) -> float:
    """The one scale that fixes font-to-plot ratio: scale = panel_h / 3.45.
    Enforces the single-panel minimum of 3.45 in. Pass the result to
    apply_xinger() and the helpers so every figure shares one text-to-plot
    ratio regardless of panel count or panel size."""
    return max(panel_height_in, REF_PANEL_IN) / REF_PANEL_IN


def axis_label_size(scale: float) -> float:
    """L = the axis-label font size at this scale. All other sizes key off it."""
    return AXIS_LABEL_PT * scale


def _sizes(scale):
    """Resolve the full ratio system to concrete point sizes."""
    L = axis_label_size(scale)
    return {
        "label": L,
        "title": R_TITLE * L,
        "panel": R_PANEL * L,
        "tick": R_TICK * L,
        "labelpad": R_LABELPAD * L,
        "legend_cap": R_LEGEND_CAP * L,   # 0.80*L = tick size; used as-is
    }


def apply_xinger(scale: float = 1.0):
    """Set global rcParams to the Xinger look. `scale` (= panel_height_in/3.45)
    multiplies font sizes and line widths so the text-to-plot ratio is fixed on
    every panel. Font hierarchy follows the ratio contract in the docstring:
    title = axis label; ticks = 0.8*axis label."""
    s = _sizes(scale)
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.transparent": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "text.color": "#000000",
        "axes.edgecolor": "#000000", "axes.labelcolor": "#000000",
        "xtick.color": "#000000", "ytick.color": "#000000",
        "axes.linewidth": 1.0 * scale,
        # title == axis label (rule 2); ticks 0.8x (rule 3)
        "axes.titlesize": s["title"], "axes.labelsize": s["label"],
        "xtick.labelsize": s["tick"], "ytick.labelsize": s["tick"],
        "legend.fontsize": s["legend_cap"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.titlelocation": "center",
        # narrow, fixed tick-label -> axis-label gap (rule 5)
        "axes.labelpad": s["labelpad"],
        "xtick.major.pad": 0.20 * s["label"], "ytick.major.pad": 0.20 * s["label"],
        "xtick.major.width": 1.0 * scale, "ytick.major.width": 1.0 * scale,
        "xtick.major.size": 7.1 * scale, "ytick.major.size": 7.1 * scale,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "none",
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "lines.linewidth": 2.2 * scale, "lines.markersize": 7 * scale,
    })


def _wrap_to_two_lines(text):
    """Split a one-line label into two balanced lines at a word boundary near
    the middle. Leaves already-multiline text untouched."""
    if "\n" in text:
        return text
    words = text.split()
    if len(words) < 2:
        return text                       # single token: cannot wrap
    # choose the split point that most evenly balances character counts
    total = len(text)
    best_i, best_diff = 1, 1e9
    for i in range(1, len(words)):
        left = len(" ".join(words[:i]))
        diff = abs(left - (total - left))
        if diff < best_diff:
            best_i, best_diff = i, diff
    return " ".join(words[:best_i]) + "\n" + " ".join(words[best_i:])


def _fit_axis_label_width(ax, which):
    """If an axis label is wider than the axes span, rewrap it to two lines
    (rule 9). `which` is 'x' or 'y'; the y-label is measured against the axes
    HEIGHT (it is rotated 90 deg)."""
    lbl = ax.xaxis.label if which == "x" else ax.yaxis.label
    text = lbl.get_text()
    if not text:
        return
    fig = ax.figure
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    bb = lbl.get_window_extent(renderer=renderer)
    label_extent = bb.width if which == "x" else bb.height  # y-label is rotated
    ax_bb = ax.get_window_extent(renderer=renderer)
    axis_span = ax_bb.width if which == "x" else ax_bb.height
    if label_extent > axis_span:
        wrapped = _wrap_to_two_lines(text)
        if wrapped != text:
            lbl.set_text(wrapped)


def xinger_axes(ax, scale: float = 1.0, wrap_labels: bool = True):
    """Per-axes: half-open spines (left+bottom only), black, thick; long ticks;
    tick-label size = 0.8*axis label; narrow fixed labelpad; no title.
    If wrap_labels (default), an axis label wider than the panel is auto-wrapped
    to two lines (rule 9). Call after plotting (labels must already be set)."""
    s = _sizes(scale)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#000000")
        ax.spines[side].set_linewidth(1.0 * scale)
    ax.tick_params(colors="#000000", width=1.0 * scale, length=7.1 * scale,
                   labelsize=s["tick"], pad=0.20 * s["label"])
    # narrow, fixed gap between tick numbers and the axis label (rule 5)
    ax.xaxis.labelpad = s["labelpad"]
    ax.yaxis.labelpad = s["labelpad"]
    ax.xaxis.label.set_size(s["label"])
    ax.yaxis.label.set_size(s["label"])
    ax.set_title("")
    if wrap_labels:
        _fit_axis_label_width(ax, "x")
        _fit_axis_label_width(ax, "y")


def xinger_map_axes(ax, scale: float = 1.0, unit="m", nticks=5, sep=" "):
    """Special case for FLOOR-PLAN / MAP panels where the axes are spatial
    coordinates (e.g. world x and z in metres).

    Unlike a data plot, a map does not get axis-title labels: the coordinate
    name ("x (m)" / "z (m)") is redundant. Instead this:
      * REMOVES both axis labels,
      * appends the spatial UNIT to the tick numbers themselves (e.g. "20 m"),
      * uses SPARSER ticks than a normal plot (nticks, default 5, vs matplotlib's
        default ~8-10),
    while keeping the rest of the Xinger look (half-open black spines, tick
    size = 0.8*axis label, thick spines/ticks). Call after plotting; set the
    aspect (usually `ax.set_aspect('equal')`) yourself.

    `unit`   : unit string appended to each tick label ("m", "cm", ...); "" to
               suppress the suffix entirely.
    `nticks` : target number of ticks per axis (sparser = smaller).
    `sep`    : separator between number and unit ("20 m"); use "" for "20m".
    """
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    s = _sizes(scale)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#000000")
        ax.spines[side].set_linewidth(1.0 * scale)
    ax.tick_params(colors="#000000", width=1.0 * scale, length=7.1 * scale,
                   labelsize=s["tick"], pad=0.20 * s["label"])
    # no coordinate-name axis labels on a map
    ax.set_xlabel(""); ax.set_ylabel("")
    # sparse ticks; integer-preferred, symmetric endpoints
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nticks, integer=False,
                                           prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nticks, integer=False,
                                           prune=None))

    def _fmt(v, _pos):
        # drop a trailing .0 so "20.0" -> "20", then append the unit
        num = f"{v:g}"
        return f"{num}{sep}{unit}" if unit else num
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.set_title("")


def add_title(ax, text, scale: float = 1.0, **kw):
    """Non-italic, non-bold plot title at the axis-label size (rules 1, 2).
    Rendered as a strip-style text just above the axes (keeps Xinger's
    no-native-title convention but at the correct size and upright)."""
    s = _sizes(scale)
    return ax.text(kw.pop("x", 0.5), kw.pop("y", 1.02), text,
                   transform=ax.transAxes, ha=kw.pop("ha", "center"),
                   va=kw.pop("va", "bottom"), fontstyle="normal",
                   fontweight="normal", fontsize=s["title"],
                   color="#000000", **kw)


def add_panel_label(ax, letter="a", scale: float = 1.0, dot=True, **kw):
    """Bold panel letter at the axis-label size (rule 2). Lowercase by default
    (e.g. 'a.'); pass letter='A' for uppercase. Anchored just outside the
    axes' top-left.

    For MULTI-PANEL figures prefer align_panel_labels() (rule 10): it places all
    letters at a uniform figure-space offset so panels in the same row share the
    letter's HEIGHT and panels in the same column share the letter's X."""
    s = _sizes(scale)
    txt = f"{letter}." if dot and not letter.endswith(".") else letter
    return ax.text(kw.pop("x", -0.16), kw.pop("y", 1.04), txt,
                   transform=ax.transAxes, fontsize=s["panel"],
                   fontweight="bold", va=kw.pop("va", "baseline"),
                   ha=kw.pop("ha", "left"), color="#000000", **kw)


def align_panel_labels(pairs, scale: float = 1.0, dot=True,
                       dx_pt=None, dy_pt=None, tol=0.02, title_y=None):
    """Place panel letters for a MULTI-PANEL figure so they line up (rule 10).

    `pairs` is a list of (ax, letter). Each letter is drawn in FIGURE
    coordinates at a uniform offset from its axes' top-left corner, so:
      * panels sharing a ROW (same axes top) get letters at the SAME height, and
      * panels sharing a COLUMN (same axes left) get letters at the SAME x.
    Rows/columns are detected by clustering axes edges within `tol` (figure
    fraction); each cluster is snapped to a common edge so minor GridSpec
    rounding doesn't misalign the letters.

    `title_y`: if given (the axes-fraction y passed to add_title, e.g. 1.04),
    the letters are placed LEVEL WITH THE TITLES at that height instead of above
    them, and pushed further left so the centered title text never collides with
    the corner letter. If None, letters sit just above the title band.

    Returns the list of Text artists.
    """
    s = _sizes(scale)
    fig = pairs[0][0].figure
    fig.canvas.draw()  # positions must be final
    fig_w, fig_h = fig.get_size_inches()
    lp = s["panel"]
    # leftward gap: wider when level with titles so centered titles clear the
    # corner letter; modest when the letter sits above the title band.
    default_dx = 1.4 * lp if title_y is not None else 0.6 * lp
    dx = (dx_pt if dx_pt is not None else default_dx) / 72.0 / fig_w
    dy = (dy_pt if dy_pt is not None else 1.6 * lp) / 72.0 / fig_h

    # gather each axes' top and left edges (figure fraction). When aligning to
    # the title, the y anchor is the title's axes-fraction height mapped through
    # each axes' own transform (so the letters sit exactly at title height).
    tops, lefts = [], []
    for ax, _ in pairs:
        p = ax.get_position()
        lefts.append(p.x0)
        if title_y is not None:
            # axes-fraction title_y -> figure y for this axes
            _, fy = fig.transFigure.inverted().transform(
                ax.transAxes.transform((0, title_y)))
            tops.append(fy)
        else:
            tops.append(p.y1)

    def snap(vals):
        """Cluster near-equal positions and map each to the cluster mean, so
        row/column mates share an exact edge."""
        out = list(vals)
        for i, v in enumerate(vals):
            grp = [w for w in vals if abs(w - v) <= tol]
            out[i] = sum(grp) / len(grp)
        return out

    tops = snap(tops)
    lefts = snap(lefts)

    # level-with-title: sit ON the title baseline (va='bottom', no extra rise),
    # and RIGHT-align the letter so it extends leftward from the anchor -- this
    # keeps the letter's right edge clear of a centered title that reaches the
    # panel's left. above-title: rise by dy, left-aligned.
    y_rise = 0.0 if title_y is not None else dy
    ha = "right" if title_y is not None else "left"
    arts = []
    for (ax, letter), top, left in zip(pairs, tops, lefts):
        txt = f"{letter}." if dot and not str(letter).endswith(".") else str(letter)
        t = fig.text(left - dx, top + y_rise, txt, fontsize=s["panel"],
                     fontweight="bold", va="bottom", ha=ha, color="#000000")
        arts.append(t)
    return arts


# ---------------------------------------------------------------------------
# Legend / stats placement (rules 6, 7, 8): inside the panel, capped at L-1 pt,
# as large as fits, and NOT overlapping the plotted data (lines/markers/bands).
# ---------------------------------------------------------------------------

# candidate in-axes anchors (loc, x, y) tried in order; first non-overlapping wins
_INSIDE_ANCHORS = [
    ("upper left", 0.03, 0.97), ("upper right", 0.97, 0.97),
    ("lower left", 0.03, 0.03), ("lower right", 0.97, 0.03),
    ("center left", 0.03, 0.5), ("center right", 0.97, 0.5),
    ("upper center", 0.5, 0.97), ("lower center", 0.5, 0.03),
]


def _data_occupancy(ax, grid=24):
    """Coarse boolean grid (axes-fraction) of where plotted data sits, so
    legend/stats can be placed in an empty region. Covers line vertices,
    scatter offsets, and filled-band (PolyCollection) vertices."""
    occ = np.zeros((grid, grid), dtype=bool)
    inv = ax.transAxes.inverted()

    def mark(disp_xy):
        if disp_xy is None or len(disp_xy) == 0:
            return
        frac = inv.transform(disp_xy)
        for fx, fy in frac:
            if 0 <= fx <= 1 and 0 <= fy <= 1:
                occ[min(grid - 1, int(fy * grid)),
                    min(grid - 1, int(fx * grid))] = True

    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy):
            mark(ax.transData.transform(xy))
    for coll in ax.collections:
        try:
            offs = coll.get_offsets()
            if offs is not None and len(offs):
                mark(ax.transData.transform(np.asarray(offs)))
        except Exception:
            pass
        # filled bands (fill_between -> PolyCollection): use path vertices
        for path in getattr(coll, "get_paths", lambda: [])():
            v = path.vertices
            if len(v):
                mark(ax.transData.transform(v))
    return occ


def _box_fill(occ, x, y, w, h):
    """Fraction of the axes-fraction box near (x,y) that is occupied by data."""
    grid = occ.shape[0]
    x0 = min(max(x - w / 2, 0), 1 - w); y0 = min(max(y - h / 2, 0), 1 - h)
    i0, i1 = int(y0 * grid), int((y0 + h) * grid) + 1
    j0, j1 = int(x0 * grid), int((x0 + w) * grid) + 1
    sub = occ[i0:i1, j0:j1]
    return sub.mean() if sub.size else 1.0


def _region_free(occ, x, y, w=0.34, h=0.20):
    """Is the axes-fraction box centered near (x,y) free of data?"""
    return _box_fill(occ, x, y, w, h) == 0.0


def _best_inside_anchor(ax, box_w=0.34, box_h=0.20, occ=None, avoid=None):
    """Pick the in-panel anchor with the LEAST data underneath (0 = fully
    clear). `avoid` is a list of (x,y) already-used anchors to skip so a legend
    and a stats box never stack in the same corner. Ties break by anchor order
    (upper-left first)."""
    if occ is None:
        occ = _data_occupancy(ax)
    avoid = avoid or []
    best, best_fill = None, 2.0
    for loc, x, y in _INSIDE_ANCHORS:
        if any(abs(x - ax0) < 1e-6 and abs(y - ay0) < 1e-6
               for ax0, ay0 in avoid):
            continue
        fill = _box_fill(occ, x, y, box_w, box_h)
        if fill < best_fill:
            best, best_fill = (loc, x, y), fill
    if best is None:                       # everything avoided; use first anchor
        loc, x, y = _INSIDE_ANCHORS[0]
        best = (loc, x, y)
    return best


class LegendFitError(RuntimeError):
    """Raised when a legend cannot be placed inside the panel even at the
    minimum font (tick size) and after wrapping labels to two rows. The caller
    (you) must then shorten category names, enlarge the panel, or decide to
    place the legend outside — per the 'always check' rule, this module
    will NOT silently move it outside, shrink past the floor, or rename."""


def _legend_bbox_frac(leg, ax):
    """The legend's bounding box in AXES-FRACTION coords (after a draw)."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    bb = leg.get_window_extent(renderer=r)
    inv = ax.transAxes.inverted()
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x0, y0, x1, y1


def _legend_fits_inside(leg, ax, occ, margin=0.01):
    """True if the legend is fully inside the panel AND its box does not overlap
    the data-occupancy grid."""
    x0, y0, x1, y1 = _legend_bbox_frac(leg, ax)
    if x0 < -margin or y0 < -margin or x1 > 1 + margin or y1 > 1 + margin:
        return False
    grid = occ.shape[0]
    j0, j1 = max(0, int(x0 * grid)), min(grid, int(x1 * grid) + 1)
    i0, i1 = max(0, int(y0 * grid)), min(grid, int(y1 * grid) + 1)
    sub = occ[i0:i1, j0:j1]
    return not (sub.size and sub.any())


def _wrap_label_two_rows(text):
    """Balance a legend label onto two rows at a word boundary (reuses the
    axis-label wrapper); single-word labels are returned unchanged."""
    return _wrap_to_two_lines(text)


def place_legend(ax, scale: float = 1.0, handles=None, labels=None,
                 italic=True, **kw):
    """Legend placed INSIDE the panel (rules 6-8). In-plot text is a FIXED size
    = 0.80*L (80% of the axis label, = the tick size); there is no shrink ladder.

      1. try each in-panel anchor at 0.80*L; the first anchor whose box fits
         inside without overlapping data wins;
      2. if none fit, WRAP each label to two rows and retry the anchors;
      3. if it STILL won't fit, raise LegendFitError so you can decide
         (shorten names / resize panel / outside) -- this module never silently
         moves it outside, shrinks the text, or renames.

    Draw the data BEFORE calling this. italic/frameless/no-title (Xinger)."""
    s = _sizes(scale)
    # In-plot text is a FIXED size = 0.80*L (= tick size); no shrink ladder.
    fixed_fs = min(s["legend_cap"], kw.pop("fontsize", s["legend_cap"]))
    ax.figure.canvas.draw()
    occ = _data_occupancy(ax)
    used = getattr(ax, "_xinger_used_anchors", [])

    # Resolve the handles/labels once so we can relabel (wrap) them later.
    if handles is not None:
        base_handles = handles
        base_labels = labels
    else:
        base_handles, auto_labels = ax.get_legend_handles_labels()
        base_labels = labels if labels is not None else auto_labels

    def make_legend(lab, fs, loc, x, y):
        lg_kw = dict(frameon=False, title=None, fontsize=fs,
                     loc=loc, bbox_to_anchor=(x, y),
                     bbox_transform=ax.transAxes,
                     handletextpad=kw.get("handletextpad", 0.5),
                     labelspacing=kw.get("labelspacing", 0.4))
        lg_kw.update({k: v for k, v in kw.items()
                      if k not in ("handletextpad", "labelspacing")})
        # always pass explicit handles+labels so wrapping/relabeling applies
        leg = ax.legend(base_handles, lab, **lg_kw)
        if italic:
            for t in leg.get_texts():
                t.set_fontstyle("italic")
        return leg

    def try_ladder(lab):
        """Try every anchor at the FIXED in-plot size (0.80*L); return the first
        anchor whose legend fits inside without overlapping data."""
        for loc, x, y in _INSIDE_ANCHORS:
            if any(abs(x - ax0) < 1e-6 and abs(y - ay0) < 1e-6
                   for ax0, ay0 in used):
                continue
            leg = make_legend(lab, fixed_fs, loc, x, y)
            ax.figure.canvas.draw()
            if _legend_fits_inside(leg, ax, occ):
                return leg, (x, y)
            leg.remove()
        return None, None

    # stage 1-2: font ladder with original labels
    leg, anchor = try_ladder(base_labels)
    # stage 2: wrap labels to two rows, retry the anchors
    if leg is None and base_labels is not None:
        wrapped = [_wrap_label_two_rows(t) for t in base_labels]
        if wrapped != list(base_labels):
            leg, anchor = try_ladder(wrapped)
    # stage 3: give up -> caller decides (per 'always check with me')
    if leg is None:
        raise LegendFitError(
            f"Legend does not fit inside the panel at the in-plot text size "
            f"({fixed_fs:.0f} pt = 0.80*axis label) even after wrapping labels "
            "to two rows. Shorten the category names, enlarge the panel, or "
            "place the legend outside — checking with the user first.")
    ax._xinger_used_anchors = used + [anchor]
    return leg


def place_stats(ax, text, scale: float = 1.0, **kw):
    """Stats text placed INSIDE the panel in a data-free region, at the fixed
    in-plot text size = 0.80*L (= tick size), no frame (rules 6, 7, 8). Draw the
    data first."""
    s = _sizes(scale)
    fs = min(s["legend_cap"], kw.pop("fontsize", s["legend_cap"]))
    ax.figure.canvas.draw()
    used = getattr(ax, "_xinger_used_anchors", [])
    loc, x, y = _best_inside_anchor(ax, box_w=0.42, box_h=0.18, avoid=used)
    ax._xinger_used_anchors = used + [(x, y)]
    ha = "left" if "left" in loc else ("right" if "right" in loc else "center")
    va = "top" if "upper" in loc else ("bottom" if "lower" in loc else "center")
    return ax.text(kw.pop("x", x), kw.pop("y", y), text,
                   transform=ax.transAxes, ha=kw.pop("ha", ha),
                   va=kw.pop("va", va), fontsize=fs, color="#000000", **kw)


def style_legend(ax_or_fig, loc="best", italic=True, **kw):
    """[Legacy] Legend with blank title and italic text. Prefer place_legend(),
    which enforces the in-panel + size-cap + non-overlap rules. Kept so older
    scripts keep working."""
    leg = ax_or_fig.legend(loc=loc, frameon=False, title=None, **kw)
    if leg is not None and italic:
        for txt in leg.get_texts():
            txt.set_fontstyle("italic")
    return leg


def series_colors(n):
    """n distinct palette colors for multiple series in one panel (no gradient).
    Cycles the categorical palette."""
    return [CYCLE[i % len(CYCLE)] for i in range(n)]


def shade(hexcol, factor):
    """Lighten a hex color toward white by `factor` in [0,1]. Use sparingly —
    prefer distinct series_colors() over same-hue gradients."""
    hexcol = hexcol.lstrip("#")
    rgb = np.array([int(hexcol[i:i + 2], 16) for i in (0, 2, 4)]) / 255
    return tuple(rgb + (1 - rgb) * factor)


def save_xinger(fig, path_stem, exts=("png", "pdf")):
    from pathlib import Path
    Path(path_stem).parent.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight",
                    transparent=True)
    return [f"{path_stem}.{ext}" for ext in exts]
