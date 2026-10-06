"""
gui_tab1.py  –  Tab 1: Elution Volume Analysis
===============================================

Fixes in this version
---------------------
1. Axis selector shows type-based names once columns are assigned:
     Before: "0", "1", "2", "3"
     After:  "time (min)", "weight (g)", "signal (mV)", "Col 3"
2. "Apply" / "Apply to all files" (all files, not just selected).
3. Selection persistence – buttons have takefocus=0 and Listbox has
   exportselection=0. Panel actions never touch the Listbox selection.
4. Plot never disappears – _redraw() saves/restores selection
   context before running.
5. Pipeline order: cut → baseline → smoothing → fitting → momentum.
   Disabling any step re-runs from the step before it with raw data.
6. Smooth scrolling: single pixel-unit canvas scroll accumulator;
   global <MouseWheel> handler bound at the Tk-level so it fires
   regardless of which child widget the pointer is over.
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm

from .data_io import (
    FileData, COL_TYPES, UNIT_OPTIONS, DEFAULT_UNIT,
    SOLVENTS, SOLVENT_NAMES, export_elution_volumes,
    export_analyte_elution_data,
)
from .standards import (
    STANDARD_KEYS, STANDARD_LABELS, detect_standard,
    compute_rh, standard_needs_mw,
)
from .analysis import analyze_file, FIT_MODEL_NAMES
from .state import AppState, GlobalSettings, normalize_analysis_axes
from .volume_calibration import volume_from_time, volume_from_weight, regress_mass_flow_per_file

# ── palette ──────────────────────────────────────────────────────────────────
ACCENT = "#1565C0"
BG     = "#F5F5F5"
PANEL  = "#FFFFFF"
BORDER = "#E0E0E0"
TEXT   = "#212121"
MUTED  = "#757575"
RED_HL = "#C62828"
GREEN  = "#2E7D32"
BOLD9  = ("Segoe UI", 9, "bold")
NORM8  = ("Segoe UI", 8)
MONO8  = ("Courier New", 8)

_MULTI = "— multiple —"


# ─────────────────────────────────────────────────────────────────────────────
# Smooth-scroll canvas wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _ScrollCanvas(tk.Canvas):
    """
    Canvas that accumulates fractional scroll units for smooth trackpad
    scrolling, and registers a global Tk-level handler so the scroll fires
    even when the pointer is over a child widget (button, label, combobox…).
    """
    _PIXELS_PER_UNIT = 20   # how many pixels count as one canvas 'unit'

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._accum = 0.0

        # Register on the root window so we catch events that land on children
        self.bind("<Map>", self._on_map)

    def _on_map(self, e=None):
        root = self.winfo_toplevel()
        root.bind("<MouseWheel>", self._global_wheel, add="+")
        root.bind("<Button-4>",   self._global_wheel, add="+")
        root.bind("<Button-5>",   self._global_wheel, add="+")

    def _global_wheel(self, event):
        # Only scroll if the pointer is actually inside this canvas or its children
        wx, wy = event.widget.winfo_rootx(), event.widget.winfo_rooty()
        cx, cy = self.winfo_rootx(), self.winfo_rooty()
        cw, ch = self.winfo_width(), self.winfo_height()
        ex = wx + event.x
        ey = wy + event.y
        if not (cx <= ex <= cx + cw and cy <= ey <= cy + ch):
            return

        # Delta normalisation: Windows gives multiples of 120, macOS gives
        # small pixel-level values, Linux gives +1/-1 via Button-4/5.
        if event.num in (4, 5):
            delta = -3 if event.num == 4 else 3
        else:
            # event.delta is in platform units; normalise to ~3 lines per notch
            delta = -event.delta / 40.0   # positive = scroll down

        self._accum += delta
        units = int(self._accum / self._PIXELS_PER_UNIT)
        if units != 0:
            self._accum -= units * self._PIXELS_PER_UNIT
            self.yview_scroll(units, "units")


def _scrollable_frame(parent: tk.Widget) -> ttk.Frame:
    """
    Grid a _ScrollCanvas + always-visible ttk.Scrollbar into *parent*.
    The scrollbar is placed first (column 1) so it always occupies its
    natural width; the canvas (column 0) gets the remaining space.
    Returns the inner ttk.Frame (the scrollable content area).
    """
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    parent.columnconfigure(1, weight=0)          # scrollbar column – fixed width

    sb = ttk.Scrollbar(parent, orient=tk.VERTICAL)
    sb.grid(row=0, column=1, sticky="ns")        # reserve column before canvas

    canvas = _ScrollCanvas(parent, bg=BG, highlightthickness=0,
                           yscrollcommand=sb.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    sb.configure(command=canvas.yview)

    inner = ttk.Frame(canvas)
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _resize(e=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(win_id, width=max(canvas.winfo_width(), 10))

    inner.bind("<Configure>", _resize)
    canvas.bind("<Configure>", _resize)

    return inner


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _forn(s: str) -> Optional[float]:
    s = str(s).strip()
    return float(s) if s else None


def _uniform(values) -> tuple[bool, object]:
    s = set(str(v) for v in values)
    if len(s) == 1:
        return True, values[0]
    return False, _MULTI


def _col_label_for_axis(fd: FileData, idx: int) -> str:
    """
    Human-readable axis label for the plot axis selector.
    Before type assignment:  "Col 0"
    After type assignment:   "time (min)"  or  "weight (g)"  etc.
    """
    if idx >= len(fd.col_info):
        return f"Col {idx}"
    ci = fd.col_info[idx]
    if ci.col_type == "other":
        return f"Col {idx}" if ci.raw_name.isdigit() else ci.raw_name
    unit = ci.unit if ci.unit and ci.unit != "—" else ""
    return f"{ci.col_type} ({unit})" if unit else ci.col_type


def _col_choices_for_axis(fd: FileData) -> list[str]:
    """Return axis-selector strings for all columns, prefixed by index."""
    return [f"{i}: {_col_label_for_axis(fd, i)}" for i in range(len(fd.col_info))]


def _col_choices_full(fd: FileData) -> list[str]:
    """Return 'N: raw_name (unit)' strings for use in range column selectors."""
    return [f"{i}: {fd.col_label(i)}" for i in range(len(fd.col_info))]


def _parse_idx(s: str) -> Optional[int]:
    try:
        return int(str(s).split(":")[0])
    except (ValueError, IndexError, AttributeError):
        return None


def _set_combo_str(cb: ttk.Combobox, value: str):
    vals = list(cb["values"])
    if value in vals:
        cb.set(value)
    elif vals:
        cb.current(0)
    else:
        cb.set("")


def _nofocus_btn(parent, **kw) -> ttk.Button:
    """Button that does not steal keyboard focus (preserves Listbox selection)."""
    return ttk.Button(parent, takefocus=0, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# _XRangeWidget
# ─────────────────────────────────────────────────────────────────────────────

class _XRangeWidget:
    """
    A compound widget:
      [header label]                     [← view]
      [column selector combobox                  ]
      From: [Entry]  unit
      To:   [Entry]  unit
    """

    def __init__(self, parent: ttk.Frame, row: int,
                 label: str, get_ax_fn, get_fd_fn):
        self._get_ax = get_ax_fn
        self._get_fd = get_fd_fn
        self._col_var  = tk.StringVar()
        self._min_var  = tk.StringVar()
        self._max_var  = tk.StringVar()
        self._unit_var = tk.StringVar(value="")

        # Header
        hdr = ttk.Frame(parent)
        hdr.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(hdr, text=label, font=BOLD9).pack(side=tk.LEFT)
        _nofocus_btn(hdr, text="← view", width=7,
                     command=self._fill_from_view).pack(side=tk.RIGHT)

        # Column selector
        self._col_cb = ttk.Combobox(parent, textvariable=self._col_var,
                                    state="readonly", width=22)
        self._col_cb.grid(row=row+1, column=0, columnspan=3, sticky="ew", pady=1)
        self._col_cb.bind("<<ComboboxSelected>>", self._on_col_changed)

        # From / To entries
        ttk.Label(parent, text="From:").grid(row=row+2, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self._min_var, width=9).grid(
            row=row+2, column=1, sticky="ew", padx=2)
        ttk.Label(parent, textvariable=self._unit_var,
                  foreground=MUTED, font=NORM8).grid(row=row+2, column=2, sticky="w")

        ttk.Label(parent, text="To:").grid(row=row+3, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self._max_var, width=9).grid(
            row=row+3, column=1, sticky="ew", padx=2)
        ttk.Label(parent, textvariable=self._unit_var,
                  foreground=MUTED, font=NORM8).grid(row=row+3, column=2, sticky="w")

    # ------------------------------------------------------------------
    def refresh_cols(self, fd: Optional[FileData]):
        if fd is None:
            self._col_cb["values"] = []
            return
        choices = _col_choices_full(fd)
        self._col_cb["values"] = choices
        cur = self._col_var.get()
        if cur not in choices:
            if choices:
                self._col_cb.current(0)
        self._on_col_changed()

    def _on_col_changed(self, e=None):
        fd = self._get_fd()
        idx = _parse_idx(self._col_var.get())
        if fd and idx is not None and idx < len(fd.col_info):
            u = fd.col_info[idx].unit
            self._unit_var.set(u if u != "—" else "")

    def _get_col_idx(self) -> Optional[int]:
        return _parse_idx(self._col_var.get())

    def _fill_from_view(self):
        xmin, xmax = self._get_ax().get_xlim()
        self._min_var.set(f"{xmin:.6g}")
        self._max_var.set(f"{xmax:.6g}")

    def get(self) -> tuple[Optional[int], Optional[float], Optional[float]]:
        return self._get_col_idx(), _forn(self._min_var.get()), _forn(self._max_var.get())

    def set(self, col_idx: Optional[int],
            lo: Optional[float], hi: Optional[float],
            fd: Optional[FileData] = None):
        if fd is not None:
            self.refresh_cols(fd)
        choices = list(self._col_cb["values"])
        if col_idx is not None:
            match = [c for c in choices if c.startswith(f"{col_idx}:")]
            if match:
                self._col_cb.set(match[0])
        self._min_var.set("" if lo is None else f"{lo:.6g}")
        self._max_var.set("" if hi is None else f"{hi:.6g}")
        self._on_col_changed()

    def set_multi(self):
        self._min_var.set(_MULTI)
        self._max_var.set(_MULTI)


# ─────────────────────────────────────────────────────────────────────────────
# Tab1
# ─────────────────────────────────────────────────────────────────────────────

class Tab1(ttk.Frame):

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app       = app
        self.file_data: list[FileData] = []
        self.state = AppState()
        self._populating = False
        # Compatibility aliases; the authoritative values are in state.global_settings.
        self._density = self.state.global_settings.density_g_ml
        self._flow_rate = self.state.global_settings.flow_rate_ml_min
        self._build_layout()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._hpane = tk.PanedWindow(
            self, orient=tk.HORIZONTAL,
            sashwidth=6, sashpad=0, bg=BORDER,
        )
        self._hpane.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        left   = self._make_left()
        center = self._make_center()
        right  = self._make_right()

        self._hpane.add(left,   minsize=160, width=190, stretch="never")
        self._hpane.add(center, minsize=300,             stretch="always")
        self._hpane.add(right,  minsize=290, width=340, stretch="never")

    # ── left: file list ──────────────────────────────────────────────────────

    def _make_left(self) -> ttk.Frame:
        frm = ttk.LabelFrame(self, text="Files", padding=4)
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        lb_wrap = ttk.Frame(frm)
        lb_wrap.grid(row=0, column=0, sticky="nsew")
        lb_wrap.rowconfigure(0, weight=1)
        lb_wrap.columnconfigure(0, weight=1)

        # exportselection=False: selection survives focus changes
        self.file_lb = tk.Listbox(
            lb_wrap, selectmode=tk.EXTENDED, activestyle="none",
            bg=PANEL, fg=TEXT,
            selectbackground=ACCENT, selectforeground="white",
            relief="flat", borderwidth=1,
            highlightthickness=1, highlightcolor=BORDER,
            font=NORM8,
            exportselection=False,
        )
        sb = ttk.Scrollbar(lb_wrap, orient=tk.VERTICAL, command=self.file_lb.yview)
        self.file_lb.configure(yscrollcommand=sb.set)
        self.file_lb.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self.file_lb.bind("<<ListboxSelect>>", self._on_selection_changed)
        self.file_lb.bind("<Button-3>", self._on_lb_rightclick)
        # Clear selection when clicking empty area below items
        self.file_lb.bind("<Button-1>", self._on_lb_click)

        btn = ttk.Frame(frm)
        btn.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        btn.columnconfigure(0, weight=1)
        _nofocus_btn(btn, text="+ Add files",
                     command=self._add_files).grid(row=0, column=0, sticky="ew")
        return frm

    def _on_lb_click(self, event):
        """Clear selection only if click lands below all items."""
        idx = self.file_lb.nearest(event.y)
        if idx >= 0:
            # Check if click is actually within the item bbox
            bbox = self.file_lb.bbox(idx)
            if bbox and event.y > bbox[1] + bbox[3]:
                self.file_lb.selection_clear(0, tk.END)
                self._redraw()

    # ── center: plot + axis selectors ────────────────────────────────────────

    def _make_center(self) -> ttk.Frame:
        frm = ttk.LabelFrame(self, text="Chromatogram", padding=4)
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        toolbar_frm = ttk.Frame(frm)
        toolbar_frm.grid(row=0, column=0, sticky="ew")

        self._fig    = Figure(figsize=(5, 4), dpi=100, facecolor=PANEL)
        self.ax      = self._fig.add_subplot(111)
        self.ax.set_facecolor(PANEL)
        self._canvas = FigureCanvasTkAgg(self._fig, master=frm)
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        NavigationToolbar2Tk(self._canvas, toolbar_frm).update()

        # Axis selectors
        sel = ttk.Frame(frm)
        sel.grid(row=2, column=0, sticky="ew", pady=(6, 2))
        sel.columnconfigure(1, weight=1)
        sel.columnconfigure(3, weight=1)

        ttk.Label(sel, text="X:").grid(row=0, column=0, sticky="w", padx=(0, 3))
        self._plot_x_var = tk.StringVar()
        self._plot_x_cb  = ttk.Combobox(sel, textvariable=self._plot_x_var,
                                         state="readonly", width=18)
        self._plot_x_cb.grid(row=0, column=1, sticky="ew")
        self._plot_x_cb.bind("<<ComboboxSelected>>", self._on_plot_axis_changed)

        ttk.Label(sel, text="Y:", padding=(10, 0, 3, 0)).grid(row=0, column=2, sticky="w")
        self._plot_y_var = tk.StringVar()
        self._plot_y_cb  = ttk.Combobox(sel, textvariable=self._plot_y_var,
                                         state="readonly", width=18)
        self._plot_y_cb.grid(row=0, column=3, sticky="ew")
        self._plot_y_cb.bind("<<ComboboxSelected>>", self._on_plot_axis_changed)

        return frm

    # ── right: vertical PanedWindow ──────────────────────────────────────────

    def _make_right(self) -> ttk.Frame:
        wrapper = ttk.Frame(self)
        wrapper.rowconfigure(0, weight=1)
        wrapper.columnconfigure(0, weight=1)

        vp = tk.PanedWindow(wrapper, orient=tk.VERTICAL,
                            sashwidth=6, sashpad=0, bg=BORDER)
        vp.grid(row=0, column=0, sticky="nsew")

        top_lf = ttk.LabelFrame(vp, text="Import Settings", padding=(4, 4))
        self._imp_inner = _scrollable_frame(top_lf)
        self._imp_inner.columnconfigure(1, weight=1)
        vp.add(top_lf, minsize=100, stretch="always")

        bot_lf = ttk.LabelFrame(vp, text="Analysis Settings", padding=(4, 4))
        self._ana_inner = _scrollable_frame(bot_lf)
        self._ana_inner.columnconfigure(1, weight=1)
        vp.add(bot_lf, minsize=100, stretch="always")

        self._build_import_panel(self._imp_inner)
        self._build_analysis_panel(self._ana_inner)
        return wrapper

    # ── import panel ─────────────────────────────────────────────────────────

    def _build_import_panel(self, p: ttk.Frame):
        r = 0

        # ── File format ──────────────────────────────────────────────
        ttk.Label(p, text="File format", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        ttk.Label(p, text="Separator:").grid(row=r, column=0, sticky="w")
        self._sep_var = tk.StringVar(value="auto")
        ttk.Combobox(p, textvariable=self._sep_var,
                     values=["auto", "space", "tab", ",", ";"],
                     state="readonly", width=9).grid(
            row=r, column=1, sticky="w", padx=2); r += 1

        ttk.Label(p, text="Decimal:").grid(row=r, column=0, sticky="w")
        self._dec_var = tk.StringVar(value="auto")
        ttk.Combobox(p, textvariable=self._dec_var,
                     values=["auto", ".", ","],
                     state="readonly", width=9).grid(
            row=r, column=1, sticky="w", padx=2); r += 1

        self._hdr_var = tk.BooleanVar()
        ttk.Checkbutton(p, text="File has header row",
                        variable=self._hdr_var).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._two_btns(p, r, "Reload",
                       self._apply_format,
                       "Reload all files",
                       lambda: self._apply_format(all_files=True)); r += 1

        # ── Column type table ─────────────────────────────────────────
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(p, text="Columns", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._col_table_frame = ttk.Frame(p)
        self._col_table_frame.grid(row=r, column=0, columnspan=3,
                                   sticky="ew", pady=(2, 4)); r += 1
        self._col_widgets: list[dict] = []

        self._two_btns(p, r, "Apply column types",
                       self._apply_col_types,
                       "Apply to all files",
                       lambda: self._apply_col_types(all_files=True)); r += 1

        # ── Volume calibration (analysis always uses volume) ───────────
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(p, text="Volume calibration", font=BOLD9).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(p, text="Method:").grid(row=r, column=0, sticky="w")
        self._vol_mode_var = tk.StringVar(value="time")
        self._vol_mode_cb = ttk.Combobox(p, textvariable=self._vol_mode_var,
            values=["time", "weight", "time + weight (regression)"], state="readonly", width=25)
        self._vol_mode_cb.grid(row=r, column=1, columnspan=2, sticky="ew"); r += 1
        ttk.Label(p, text="Solvent:").grid(row=r, column=0, sticky="w")
        self._solvent_var = tk.StringVar(value="Custom")
        self._solvent_cb = ttk.Combobox(
            p, textvariable=self._solvent_var,
            values=list(SOLVENT_NAMES), state="readonly", width=18)
        self._solvent_cb.grid(row=r, column=1, columnspan=2, sticky="ew"); r += 1
        self._solvent_cb.bind("<<ComboboxSelected>>", self._on_solvent_changed)
        self._density_var = tk.StringVar(value="1.000")
        ttk.Label(p, text="Density (g/mL):").grid(row=r, column=0, sticky="w")
        self._density_entry = ttk.Entry(p, textvariable=self._density_var, width=12)
        self._density_entry.grid(row=r, column=1, sticky="ew", padx=2); r += 1
        self._flow_var = tk.StringVar(value="1.000")
        ttk.Label(p, text="Flow rate (mL/min):").grid(row=r, column=0, sticky="w")
        self._flow_entry = ttk.Entry(p, textvariable=self._flow_var, width=12)
        self._flow_entry.grid(row=r, column=1, sticky="ew", padx=2); r += 1
        ttk.Label(p, text="Regression target:").grid(row=r, column=0, sticky="w")
        self._reg_target_var = tk.StringVar(value="flow")
        self._reg_target_cb = ttk.Combobox(
            p, textvariable=self._reg_target_var,
            values=["flow", "density"], state="readonly", width=12)
        self._reg_target_cb.grid(row=r, column=1, sticky="ew", padx=2); r += 1
        ttk.Label(p, text="Regression files:").grid(row=r, column=0, sticky="w")
        ttk.Label(p, text="Use the selected files; result applies to all files", foreground=MUTED, font=NORM8).grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        self._vol_status_var = tk.StringVar(value="No calibration applied")
        ttk.Label(p, textvariable=self._vol_status_var, foreground=MUTED, font=NORM8, wraplength=260).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        _nofocus_btn(p, text="Calculate and apply to all files",
                      command=self._apply_volume_calibration).grid(
                          row=r, column=0, columnspan=3, sticky="ew", pady=2); r += 1
        self._vol_mode_cb.bind("<<ComboboxSelected>>", self._on_volume_mode_changed)
        self._on_volume_mode_changed()
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        _nofocus_btn(p, text="Export elution volumes…", command=self._export_elution).grid(row=r, column=0, columnspan=3, sticky="ew", pady=2)
        _nofocus_btn(p, text="Export analyte data…", command=self._export_analyte_data).grid(row=r + 1, column=0, columnspan=3, sticky="ew", pady=2)

    # ── analysis panel ────────────────────────────────────────────────────────

    def _build_analysis_panel(self, p: ttk.Frame):
        r = 0

        # ── Standard ─────────────────────────────────────────────────
        ttk.Label(p, text="Standard", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._std_key_var = tk.StringVar()
        self._std_combo = ttk.Combobox(p, textvariable=self._std_key_var,
                                       values=[""] + STANDARD_LABELS,
                                       state="readonly")
        self._std_combo.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        self._std_combo.bind("<<ComboboxSelected>>", self._on_std_changed)

        ttk.Label(p, text="Size (MW / #C):").grid(row=r, column=0, sticky="w")
        self._std_value_var = tk.StringVar()
        ttk.Entry(p, textvariable=self._std_value_var, width=9).grid(
            row=r, column=1, sticky="ew", padx=2)
        self._std_value_var.trace_add("write", self._on_std_val_changed); r += 1

        self._rh_label = ttk.Label(p, text="Rh = — nm",
                                   foreground=MUTED, font=NORM8)
        self._rh_label.grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._two_btns(p, r, "Apply standard",
                       lambda: self._apply_ana_sel("std"),
                       "Apply to all files",
                       lambda: self._apply_ana_all_files("std")); r += 1

        # ── Signal pre-processing ─────────────────────────────────────
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(p, text="Signal pre-processing", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._invert_var = tk.BooleanVar()
        ttk.Checkbutton(p, text="Invert Y signal", variable=self._invert_var,
                        command=self._on_toggle).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._xreset_var = tk.BooleanVar()
        ttk.Checkbutton(p, text="Reset X (first point → 0)",
                        variable=self._xreset_var,
                        command=self._on_toggle).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        # ── Step 1: X range cut ───────────────────────────────────────
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1

        ttk.Label(p, text="Step 1 — X range cut", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._cut_rng = _XRangeWidget(p, r, "Range",
                                      lambda: self.ax, self._get_primary_fd)
        r += 4
        self._two_btns(p, r, "Apply cut",
                       lambda: self._apply_ana_sel("cut"),
                       "Apply to all files",
                       lambda: self._apply_ana_all_files("cut")); r += 1

        # ── Step 2: Baseline ──────────────────────────────────────────
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(p, text="Step 2 — Baseline correction", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._bl_var = tk.BooleanVar()
        ttk.Checkbutton(p, text="Enable", variable=self._bl_var,
                        command=self._on_toggle).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._bl_rng = _XRangeWidget(p, r, "Baseline region",
                                     lambda: self.ax, self._get_primary_fd)
        r += 4

        ttk.Label(p, text="Poly. degree:").grid(row=r, column=0, sticky="w")
        self._bl_degree_var = tk.StringVar(value="1")
        ttk.Entry(p, textvariable=self._bl_degree_var, width=4).grid(
            row=r, column=1, sticky="w", padx=2); r += 1

        self._two_btns(p, r, "Apply baseline",
                       lambda: self._apply_ana_sel("baseline"),
                       "Apply to all files",
                       lambda: self._apply_ana_all_files("baseline")); r += 1

        # ── Step 3: Smoothing ─────────────────────────────────────────
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(p, text="Step 3 — Smoothing (Savitzky-Golay)", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._smooth_var = tk.BooleanVar()
        ttk.Checkbutton(p, text="Enable", variable=self._smooth_var,
                        command=self._on_toggle).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        ttk.Label(p, text="Window:").grid(row=r, column=0, sticky="w")
        self._smooth_win_var = tk.StringVar(value="11")
        ttk.Entry(p, textvariable=self._smooth_win_var, width=5).grid(
            row=r, column=1, sticky="w", padx=2); r += 1

        self._two_btns(p, r, "Apply smoothing",
                       lambda: self._apply_ana_sel("smooth"),
                       "Apply to all files",
                       lambda: self._apply_ana_all_files("smooth")); r += 1

        # ── Step 4: Peak fitting ──────────────────────────────────────
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(p, text="Step 4 — Peak fitting", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._fit_var = tk.BooleanVar()
        ttk.Checkbutton(p, text="Enable", variable=self._fit_var,
                        command=self._on_toggle).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._fit_model_var = tk.StringVar(value="gaussian")
        ttk.Combobox(p, textvariable=self._fit_model_var,
                     values=FIT_MODEL_NAMES, state="readonly").grid(
            row=r, column=0, columnspan=3, sticky="ew"); r += 1

        self._two_btns(p, r, "Apply fit",
                       lambda: self._apply_ana_sel("fit"),
                       "Apply to all files",
                       lambda: self._apply_ana_all_files("fit")); r += 1

        # ── Step 5: Momentum ──────────────────────────────────────────
        ttk.Separator(p, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(p, text="Step 5 — Elution volume (momentum)", font=BOLD9).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        ttk.Label(p, text="Degree:").grid(row=r, column=0, sticky="w")
        self._mom_degree_var = tk.StringVar(value="1")
        ttk.Entry(p, textvariable=self._mom_degree_var, width=4).grid(
            row=r, column=1, sticky="w", padx=2); r += 1

        ttk.Label(p, text="Threshold (%):").grid(row=r, column=0, sticky="w")
        self._mom_thresh_var = tk.StringVar(value="75.0")
        ttk.Entry(p, textvariable=self._mom_thresh_var, width=6).grid(
            row=r, column=1, sticky="w", padx=2); r += 1

        self._two_btns(p, r, "Apply moment",
                       lambda: self._apply_ana_sel("momentum"),
                       "Apply to all files",
                       lambda: self._apply_ana_all_files("momentum")); r += 1

    # ── widget helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _two_btns(parent, row, t1, c1, t2, c2):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        f.columnconfigure(0, weight=1); f.columnconfigure(1, weight=1)
        _nofocus_btn(f, text=t1, command=c1).grid(
            row=0, column=0, sticky="ew", padx=(0, 1))
        _nofocus_btn(f, text=t2, command=c2).grid(
            row=0, column=1, sticky="ew", padx=(1, 0))

    # ── file list ─────────────────────────────────────────────────────────────

    def _selected_indices(self) -> list[int]:
        return list(self.file_lb.curselection())

    def _selected_fds(self) -> list[FileData]:
        return [self.file_data[i] for i in self._selected_indices()
                if i < len(self.file_data)]

    def _get_primary_fd(self) -> Optional[FileData]:
        sel = self._selected_indices()
        return self.file_data[sel[0]] if sel and sel[0] < len(self.file_data) else None

    def _on_selection_changed(self, e=None):
        fds = self._selected_fds()
        if fds:
            self._populate_panels(fds)
            self._refresh_plot_selectors(fds[0])
        self._redraw()

    def _on_lb_rightclick(self, event):
        idx = self.file_lb.nearest(event.y)
        menu = tk.Menu(self, tearoff=False)
        if idx >= 0 and idx < len(self.file_data):
            if idx not in self.file_lb.curselection():
                self.file_lb.selection_clear(0, tk.END)
                self.file_lb.selection_set(idx)
            menu.add_command(label="Remove from list",
                             command=self._remove_selected)
            menu.add_command(label="Delete file(s) to .trash…",
                             command=self._delete_selected)
            menu.add_separator()
        menu.add_command(label="Clear selection",
                         command=lambda: (self.file_lb.selection_clear(0, tk.END),
                                         self._redraw()))
        menu.tk_popup(event.x_root, event.y_root)

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Import chromatography files",
            filetypes=[("Data files", "*.dat *.txt *.csv *.tsv *.asc"),
                       ("All", "*.*")],
        )
        first_new = len(self.file_data)
        for path in paths:
            try:
                fd = FileData(path)
                std_key, std_val = detect_standard(path)
                fd.std_key   = std_key
                fd.std_value = std_val
                if std_val is not None and std_key:
                    fd.rh = compute_rh(std_key, std_val)
                elif std_key and not standard_needs_mw(std_key):
                    fd.rh = compute_rh(std_key, 0.0)
                else:
                    fd.std_key = ""
                    fd.rh = None
                normalize_analysis_axes(fd)
                self.file_data.append(fd)
                self.state.files = self.file_data
                self.file_lb.insert(tk.END, fd.get_display_name())
            except Exception as exc:
                messagebox.showerror("Import error",
                                     f"Cannot load {path}:\n{exc}")

        if first_new < len(self.file_data):
            self.file_lb.selection_clear(0, tk.END)
            self.file_lb.selection_set(first_new, tk.END)
            self._on_selection_changed()
        self.app.notify_data_changed()

    def _remove_selected(self):
        indices = sorted(self._selected_indices(), reverse=True)
        for i in indices:
            if i < len(self.file_data):
                self.file_data.pop(i)
                self.file_lb.delete(i)
        if self.file_data:
            new_sel = max(0, min(indices[-1] if indices else 0,
                                 len(self.file_data) - 1))
            self.file_lb.selection_set(new_sel)
            self._on_selection_changed()
        self._redraw()
        self.app.notify_data_changed()

    def _delete_selected(self):
        """
        Move every selected file into a .trash sub-folder next to the file,
        then remove it from the list.  The .trash folder is created if needed.

        Each file is moved to  <original_dir>/.trash/<filename>.
        If a file with the same name already exists in .trash, a numeric
        suffix is appended: e.g.  data.dat  →  data_1.dat, data_2.dat, …
        """
        import os, shutil

        indices = sorted(self._selected_indices(), reverse=True)
        if not indices:
            return

        fds     = [self.file_data[i] for i in indices if i < len(self.file_data)]
        n_ok    = 0
        errors  = []

        for fd in fds:
            src_path = fd.path
            src_dir  = os.path.dirname(os.path.abspath(src_path))
            trash_dir = os.path.join(src_dir, ".trash")

            try:
                os.makedirs(trash_dir, exist_ok=True)

                filename = os.path.basename(src_path)
                dst_path = os.path.join(trash_dir, filename)

                # Avoid overwriting an existing file in .trash
                if os.path.exists(dst_path):
                    name, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dst_path):
                        dst_path = os.path.join(trash_dir,
                                                 f"{name}_{counter}{ext}")
                        counter += 1

                shutil.move(src_path, dst_path)
                n_ok += 1

            except Exception as exc:
                errors.append(f"{fd.filename}: {exc}")

        if errors:
            err_msg = "{} file(s) could not be moved to .trash:\n{}".format(
                len(errors), "\n".join(errors))
            messagebox.showerror("Delete error", err_msg)

        if n_ok > 0:
            # Remove from the list (reverse order to keep indices stable)
            for i in indices:
                if i < len(self.file_data):
                    self.file_data.pop(i)
                    self.file_lb.delete(i)

            if self.file_data:
                new_sel = max(0, min(indices[-1] if indices else 0,
                                     len(self.file_data) - 1))
                self.file_lb.selection_set(new_sel)
                self._on_selection_changed()

            self._redraw()
            self.app.notify_data_changed()

            trash_dirs = set()
            for fd in fds:
                trash_dirs.add(
                    os.path.join(os.path.dirname(os.path.abspath(fd.path)),
                                 ".trash"))
            loc = "\n".join(sorted(trash_dirs))
            messagebox.showinfo(
                "Deleted",
                "{} file(s) moved to .trash:\n{}".format(n_ok, loc),
            )

    # ── populate panels ───────────────────────────────────────────────────────

    def _populate_panels(self, fds: list[FileData]):
        if not fds:
            return
        self._populating = True
        try:
            fd0 = fds[0]

            # ─ Import panel ───────────────────────────────────────────
            same_sep, sep_v = _uniform([f.separator for f in fds])
            self._sep_var.set(
                (sep_v if sep_v != " " else "space") if same_sep else _MULTI)

            same_dec, dec_v = _uniform([f.decimal for f in fds])
            self._dec_var.set(dec_v if same_dec else _MULTI)

            same_hdr, hdr_v = _uniform([f.has_header for f in fds])
            self._hdr_var.set(bool(hdr_v) if same_hdr else False)

            self._rebuild_col_table(fd0)

            # Analysis is always volume vs signal. Volume controls are populated separately.

            # Plot X/Y selectors: real columns + derived virtual columns.
            # _refresh_plot_selectors preserves current selection when valid.
            self._refresh_plot_selectors(fd0)
            # Also update type-based names in the plot selectors
            self._refresh_axis_selectors(fd0)

            self._density_var.set(f"{self._density:.4g}")
            self._flow_var.set(f"{self._flow_rate:.4g}")
            # ─ Analysis panel ─────────────────────────────────────────
            same_sk, sk_v = _uniform([f.std_key   for f in fds])
            same_sv, sv_v = _uniform([f.std_value for f in fds])
            if same_sk:
                lbl = _std_key_to_label(sk_v)
                _set_combo_str(self._std_combo, lbl)
            else:
                self._std_combo.set(_MULTI)
            self._std_value_var.set(
                ("" if sv_v is None else str(sv_v)) if same_sv else _MULTI)
            self._update_rh_label(fds)

            same_inv, inv_v = _uniform([f.invert_y for f in fds])
            self._invert_var.set(bool(inv_v) if same_inv else False)
            same_xr, xr_v = _uniform([f.x_reset for f in fds])
            self._xreset_var.set(bool(xr_v) if same_xr else False)

            same_xm, xm_v = _uniform([f.x_min for f in fds])
            same_xx, xx_v = _uniform([f.x_max for f in fds])
            self._cut_rng.set(fd0.analysis_x_col,
                              xm_v if same_xm else None,
                              xx_v if same_xx else None, fd=fd0)
            if not (same_xm and same_xx):
                self._cut_rng.set_multi()

            same_bl, bl_v = _uniform([f.baseline_enabled for f in fds])
            self._bl_var.set(bool(bl_v) if same_bl else False)
            same_bm, bm_v = _uniform([f.baseline_x_min for f in fds])
            same_bx, bx_v = _uniform([f.baseline_x_max for f in fds])
            self._bl_rng.set(fd0.baseline_x_col,
                             bm_v if same_bm else None,
                             bx_v if same_bx else None, fd=fd0)
            if not (same_bm and same_bx):
                self._bl_rng.set_multi()
            same_bd, bd_v = _uniform([f.baseline_degree for f in fds])
            self._bl_degree_var.set(str(bd_v) if same_bd else _MULTI)

            same_sm, sm_v = _uniform([f.smoothing_enabled for f in fds])
            self._smooth_var.set(bool(sm_v) if same_sm else False)
            same_sw, sw_v = _uniform([f.smoothing_window for f in fds])
            self._smooth_win_var.set(str(sw_v) if same_sw else _MULTI)

            same_fe, fe_v = _uniform([f.fit_enabled for f in fds])
            self._fit_var.set(bool(fe_v) if same_fe else False)
            same_fm, fm_v = _uniform([f.fit_model for f in fds])
            self._fit_model_var.set(fm_v if same_fm else "gaussian")

            same_md, md_v = _uniform([f.momentum_degree    for f in fds])
            same_mt, mt_v = _uniform([f.momentum_threshold for f in fds])
            self._mom_degree_var.set(str(md_v) if same_md else _MULTI)
            self._mom_thresh_var.set(str(mt_v) if same_mt else _MULTI)

        finally:
            self._populating = False

    def _refresh_axis_selectors(self, fd: FileData):
        """
        After _refresh_plot_selectors has rebuilt the plot comboboxes with
        real + virtual entries, update only the real-column entries to show
        type-based names ("time (min)", "signal (mV)", "Col 3").
        Virtual and separator entries are left unchanged.
        """
        # Build mapping: old "N: raw_label" -> new "N: type_label"
        relabelled = {
            f"{i}: {fd.col_label(i)}": f"{i}: {_col_label_for_axis(fd, i)}"
            for i in range(len(fd.col_info))
        }
        for cb, var in ((self._plot_x_cb, self._plot_x_var),
                        (self._plot_y_cb, self._plot_y_var)):
            cur = var.get()
            new_choices = [relabelled.get(c, c) for c in cb["values"]]
            cb["values"] = new_choices
            # Update current selection label if it was a real column entry
            new_cur = relabelled.get(cur, cur)
            if new_cur in new_choices:
                var.set(new_cur)

    # ── virtual / derived column helpers ────────────────────────────────────────

    def _virtual_x_choices(self, fd: "FileData") -> list[tuple[str, str]]:
        """
        Extra X-axis options derived from the analysis pipeline.
        'v:volume' is offered only when the analysis_x_col is time or weight
        AND there is no existing real volume column already in col_info.
        Once _apply_conversion has run, a real volume column is added to
        df_raw and col_info; at that point the virtual option disappears
        because the real column is already visible in the normal column list.
        """
        choices: list[tuple[str, str]] = []
        # Check whether a real volume column already exists
        has_real_volume = any(ci.col_type == "volume" for ci in fd.col_info)
        if not has_real_volume and fd.analysis_x_col < len(fd.col_info):
            ci = fd.col_info[fd.analysis_x_col]
            if ci.col_type in ("time", "weight"):
                choices.append(("v:volume", "volume (calculated, ml)"))
        return choices

    def _virtual_y_choices(self, fd: "FileData") -> list[tuple[str, str]]:
        """
        Extra Y-axis options derived from the analysis pipeline.
        Available options grow as the user enables processing steps.
        """
        choices: list[tuple[str, str]] = []
        # Raw signal is always available as a virtual alias
        choices.append(("v:signal_raw", "signal (raw)"))
        if fd.baseline_enabled:
            choices.append(("v:signal_sub", "signal − baseline"))
            choices.append(("v:baseline",   "baseline curve"))
        if fd.smoothing_enabled:
            choices.append(("v:smoothed", "signal (smoothed)"))
        if fd.fit_enabled:
            choices.append(("v:fit", "signal (fit)"))
        return choices

    def _build_plot_x_choices(self, fd: "FileData") -> list[str]:
        """Combined real + virtual choices for the plot X combobox."""
        real = [f"{i}: {_col_label_for_axis(fd, i)}" for i in range(len(fd.col_info))]
        virt = [f"v: {lbl}" for _, lbl in self._virtual_x_choices(fd)]
        if virt:
            return real + ["─── derived ───"] + virt
        return real

    def _build_plot_y_choices(self, fd: "FileData") -> list[str]:
        """Combined real + virtual choices for the plot Y combobox."""
        real = [f"{i}: {_col_label_for_axis(fd, i)}" for i in range(len(fd.col_info))]
        virt = [f"v: {lbl}" for _, lbl in self._virtual_y_choices(fd)]
        return real + ["─── derived ───"] + virt

    def _refresh_plot_selectors(self, fd: "FileData"):
        """
        Rebuild only the plot X/Y comboboxes to reflect the current set of
        real and virtual columns.  Preserves the current selection when valid.
        Separators ("─── derived ───") are non-selectable.
        """
        x_choices = self._build_plot_x_choices(fd)
        y_choices = self._build_plot_y_choices(fd)

        for cb, var, choices in (
            (self._plot_x_cb, self._plot_x_var, x_choices),
            (self._plot_y_cb, self._plot_y_var, y_choices),
        ):
            cur = var.get()
            cb["values"] = choices
            if cur in choices and not cur.startswith("─"):
                pass   # keep current selection
            elif choices:
                # Pick first non-separator
                first = next((c for c in choices if not c.startswith("─")), choices[0])
                var.set(first)

    @staticmethod
    def _resolve_plot_choice(choice: str) -> tuple[str, object]:
        """
        Parse a plot axis combobox value.
        Returns ("real", int_index) or ("virtual", key_str) or ("sep", None).
        """
        if choice.startswith("─"):
            return "sep", None
        if choice.startswith("v: "):
            return "virtual", choice  # full string is the virtual label
        idx = _parse_idx(choice)
        if idx is not None:
            return "real", idx
        return "sep", None

    def _key_for_virtual_label(self, fd: "FileData", label: str) -> str | None:
        """Map a 'v: display label' string back to its 'v:key'."""
        for key, disp in self._virtual_x_choices(fd) + self._virtual_y_choices(fd):
            if f"v: {disp}" == label:
                return key
        return None

    def _get_virtual_y(self, fd: "FileData", key: str,
                        res: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return (x, y) arrays for a virtual Y column.
        x is always the analysis x array from res (after cut).
        Returns (None, None) if the data is not available.
        """
        x = res.get("x")
        if x is None or len(x) == 0:
            return None, None

        if key == "v:signal_raw":
            return x, res.get("y_raw")
        if key == "v:signal_sub":
            return x, res.get("y")           # baseline-subtracted
        if key == "v:baseline":
            bc = res.get("baseline_curve")
            if bc is not None:
                # Baseline lives in raw-signal space; reconstruct by adding to y
                y_sub = res.get("y", np.zeros_like(x))
                return x, y_sub + bc         # = original signal level
            return None, None
        if key == "v:smoothed":
            return x, res.get("y_smooth")
        if key == "v:fit":
            return x, res.get("y_fit")
        return None, None

    def _get_virtual_x(self, fd: "FileData", key: str) -> np.ndarray | None:
        """Return x array for a virtual X column."""
        if key == "v:volume":
            from .data_io import _TO_MIN, _TO_G, time_to_volume, weight_to_volume
            xi  = fd.analysis_x_col
            if xi >= len(fd.col_info):
                return None
            ci  = fd.col_info[xi]
            raw = fd.get_col(xi).astype(float)
            if ci.col_type == "time":
                factor = _TO_MIN.get(ci.unit, 1.0)
                return time_to_volume(raw * factor, self._flow_rate)
            if ci.col_type == "weight":
                factor = _TO_G.get(ci.unit, 1.0)
                return weight_to_volume(raw * factor, self._density)
        return None

    def _rebuild_col_table(self, fd: FileData):
        for w in self._col_table_frame.winfo_children():
            w.destroy()
        self._col_widgets = []
        p = self._col_table_frame
        p.columnconfigure(0, weight=2)
        p.columnconfigure(1, weight=2)
        p.columnconfigure(2, weight=1)

        # Header row
        for col, txt in enumerate(("Column", "Type", "Unit")):
            ttk.Label(p, text=txt, font=NORM8, foreground=MUTED).grid(
                row=0, column=col, sticky="w", padx=(0 if col else 0, 2))

        for i, ci in enumerate(fd.col_info):
            row = i + 1
            # Column name: prefer raw_name, fall back to index
            display = ci.raw_name if not ci.raw_name.isdigit() else f"Col {i}"
            ttk.Label(p, text=display, font=MONO8, anchor="w",
                      width=10).grid(row=row, column=0, sticky="ew")

            type_var = tk.StringVar(value=ci.col_type)
            type_cb  = ttk.Combobox(p, textvariable=type_var,
                                    values=COL_TYPES, state="readonly", width=7)
            type_cb.grid(row=row, column=1, sticky="ew", padx=2, pady=1)

            unit_var = tk.StringVar(value=ci.unit)
            unit_cb  = ttk.Combobox(p, textvariable=unit_var,
                                    values=UNIT_OPTIONS.get(ci.col_type, ["—"]),
                                    state="readonly", width=5)
            unit_cb.grid(row=row, column=2, sticky="ew")

            def _on_type(e, tv=type_var, uv=unit_var, ucb=unit_cb,
                         ci_ref=ci, ii=i):
                ctype = tv.get()
                opts  = UNIT_OPTIONS.get(ctype, ["—"])
                ucb["values"] = opts
                uv.set(DEFAULT_UNIT.get(ctype, "—"))
                self._write_col_type(ii, ctype, uv.get())

            def _on_unit(e, tv=type_var, uv=unit_var, ii=i):
                self._write_col_type(ii, tv.get(), uv.get())

            type_cb.bind("<<ComboboxSelected>>", _on_type)
            unit_cb.bind("<<ComboboxSelected>>", _on_unit)
            self._col_widgets.append(dict(type_var=type_var, unit_var=unit_var))

    def _write_col_type(self, col_idx: int, ctype: str, unit: str):
        """Immediately write col type/unit to all selected files and refresh selectors."""
        for fd in self._selected_fds():
            if col_idx < len(fd.col_info):
                fd.col_info[col_idx].col_type = ctype
                fd.col_info[col_idx].unit     = unit
        fd0 = self._get_primary_fd()
        if fd0:
            self._refresh_axis_selectors(fd0)

    # ── plot ─────────────────────────────────────────────────────────────────

    def _redraw(self):
        ax = self.ax
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)

        fds = self._selected_fds()
        if not fds:
            ax.text(0.5, 0.5, "No file selected", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED, fontsize=10)
            self._canvas.draw()
            return

        fd0 = fds[0]

        # Refresh plot selectors based on current analysis state of primary file
        self._refresh_plot_selectors(fd0)

        x_choice = self._plot_x_var.get()
        y_choice = self._plot_y_var.get()
        x_kind, x_val = self._resolve_plot_choice(x_choice)
        y_kind, y_val = self._resolve_plot_choice(y_choice)

        colors = cm.tab10.colors
        xlab = x_choice.split(": ", 1)[-1] if ": " in x_choice else x_choice
        ylab = y_choice.split(": ", 1)[-1] if ": " in y_choice else y_choice

        for k, fd in enumerate(fds):
            color = colors[k % len(colors)]
            label = fd.get_display_name()

            # Determine whether we need the analysis result (virtual columns
            # or the full pipeline overlay both require it)
            need_res = (x_kind == "virtual" or y_kind == "virtual")

            # Check if both axes are real column indices pointing at the
            # analysis columns → show full pipeline overlay for single file
            x_is_ana = (x_kind == "real" and x_val == fd.analysis_x_col)
            y_is_ana = (y_kind == "real" and y_val == fd.analysis_y_col)
            axes_match = x_is_ana and y_is_ana

            if len(fds) == 1 and axes_match:
                # Full pipeline overlay handles its own drawing
                self._draw_single(ax, fd, color)
                continue

            # Compute analysis result if needed for virtual columns
            res: dict = {}
            if need_res:
                try:
                    res = analyze_file(fd)
                except Exception:
                    res = {}

            # ── Resolve X ────────────────────────────────────────────
            if x_kind == "real" and x_val is not None:
                xr = fd.get_col(x_val).astype(float)
            elif x_kind == "virtual":
                vkey = self._key_for_virtual_label(fd, x_choice)
                xr_v = self._get_virtual_x(fd, vkey) if vkey else None
                if xr_v is None:
                    # Fall back: try to get x from res
                    xr_v = res.get("x")
                xr = xr_v if xr_v is not None else np.array([])
            else:
                xr = fd.get_col(fd.plot_x_col).astype(float)

            # ── Resolve Y ────────────────────────────────────────────
            if y_kind == "real" and y_val is not None:
                yr = fd.get_col(y_val).astype(float)
                # For a real column plotted against a virtual X, we need the
                # arrays to be the same length; use analysis x if shapes mismatch
            elif y_kind == "virtual":
                vkey = self._key_for_virtual_label(fd, y_choice)
                if vkey and res:
                    _, yr_v = self._get_virtual_y(fd, vkey, res)
                    # When Y is virtual, x should come from the analysis pipeline
                    # unless the user has explicitly chosen a real X column
                    if x_kind != "real" or xr is None or len(xr) == 0:
                        xr = res.get("x", np.array([]))
                    yr = yr_v if yr_v is not None else np.array([])
                else:
                    yr = np.array([])
            else:
                yr = fd.get_col(fd.plot_y_col).astype(float)

            if xr is None:
                xr = np.array([])
            if yr is None:
                yr = np.array([])

            # Trim to common length (virtual Y shares x from pipeline)
            n = min(len(xr), len(yr))
            if n == 0:
                continue
            xr, yr = xr[:n], yr[:n]

            mask = np.isfinite(xr) & np.isfinite(yr)
            xr, yr = xr[mask], yr[mask]
            if len(xr) == 0:
                continue

            lbl = label if len(fds) > 1 else y_choice.split(": ", 1)[-1]
            ax.plot(xr, yr, color=color, linewidth=1.0, label=lbl, alpha=0.9)

        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.25, color=BORDER)
        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()

    def _draw_single(self, ax, fd: FileData, color):
        """
        Draw one file with the full analysis pipeline overlay.
        Pipeline: raw → cut → baseline → smoothing → fitting → momentum.
        Each disabled step is bypassed; the data from the previous step flows
        through unchanged. No data is ever lost.
        """
        try:
            res = analyze_file(fd)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Analysis error:\n{exc}",
                    transform=ax.transAxes, ha="center", va="center",
                    color=RED_HL, fontsize=8)
            return

        if not res:
            # Fall back to raw plot column data
            xr, yr = fd.get_plot_xy()
            if len(xr):
                ax.plot(xr, yr, color=color, linewidth=1.2, label="Signal")
            return

        x    = res["x"]        # after cut
        y_bl = res["y"]        # after baseline (or = cut data if baseline off)
        y_sm = res.get("y_smooth")   # after smoothing (None if off)
        y_ft = res.get("y_fit")      # after fitting   (None if off)
        baseline = res.get("baseline_curve")

        # The "current working signal" for the elution overlay
        y_work = y_ft if y_ft is not None else (y_sm if y_sm is not None else y_bl)

        # ── Always show the baseline-subtracted raw signal ────────────
        # (behind anything else; semi-transparent if further processing is on)
        if fd.smoothing_enabled or fd.fit_enabled:
            ax.plot(x, y_bl, color=MUTED, alpha=0.30, linewidth=0.8,
                    label="After baseline")
        else:
            ax.plot(x, y_bl, color=color, linewidth=1.2, label="Signal")

        # ── Baseline curve on the un-subtracted data ──────────────────
        if fd.baseline_enabled and baseline is not None:
            # The baseline lives in the original (pre-subtraction) y space
            y_raw_signal = res.get("y_raw_for_baseline",
                                   y_bl + baseline)   # reconstruct pre-sub
            ax.plot(x, baseline, color=GREEN, linewidth=0.9,
                    linestyle="--", alpha=0.8, label="Baseline")

        # ── Smoothed curve ────────────────────────────────────────────
        if fd.smoothing_enabled and y_sm is not None:
            lw = 0.9 if fd.fit_enabled else 1.3
            ax.plot(x, y_sm, color=color, linewidth=lw,
                    alpha=0.7 if fd.fit_enabled else 1.0, label="Smoothed")

        # ── Fitted curve ──────────────────────────────────────────────
        if fd.fit_enabled and y_ft is not None:
            ax.plot(x, y_ft, color=RED_HL, linewidth=1.8,
                    label=f"Fit ({fd.fit_model})", zorder=5)

        # ── Threshold line ────────────────────────────────────────────
        thr = res.get("threshold_y", 0)
        if thr > 0:
            ax.axhline(thr, color=GREEN, linewidth=0.7, linestyle=":",
                       label=f"Threshold ({fd.momentum_threshold:.0f}%)")

        # ── Elution vertical ──────────────────────────────────────────
        ev = res.get("elution_x")
        if ev is not None:
            ax.axvline(ev, color=RED_HL, linewidth=1.1, linestyle="--",
                       label=f"Ve = {ev:.4f}")
            fd.elution_volume = ev
            self._update_rh_label([fd])

        ax.set_title(fd.get_display_name(), fontsize=9)
        ax.legend(fontsize=7, framealpha=0.7)

    def _on_plot_axis_changed(self, e=None):
        if self._populating:
            return
        x_choice = self._plot_x_var.get()
        y_choice = self._plot_y_var.get()
        # Ignore separator clicks
        if x_choice.startswith("─") or y_choice.startswith("─"):
            return
        # Only persist real column indices; virtual column choices are
        # ephemeral (they depend on the current analysis state) and are
        # re-resolved on every _redraw call.
        xi = _parse_idx(x_choice)
        yi = _parse_idx(y_choice)
        for fd in self._selected_fds():
            if xi is not None:
                fd.plot_x_col = xi
            if yi is not None:
                fd.plot_y_col = yi
        self._redraw()

    # ── import actions ────────────────────────────────────────────────────────

    def _apply_format(self, all_files: bool = False):
        sep_map = {"auto": None, "space": " ", "tab": "\t", ",": ",", ";": ";"}
        sep_raw = self._sep_var.get()
        sep = sep_map.get(sep_raw) if sep_raw != _MULTI else None
        dec_raw = self._dec_var.get()
        dec = None if dec_raw in ("auto", _MULTI) else dec_raw
        hdr = self._hdr_var.get()

        targets = self.file_data if all_files else self._selected_fds()
        errors = []
        for fd in targets:
            try:
                fd._reload(
                    sep=sep if sep is not None else fd.separator,
                    dec=dec if dec is not None else fd.decimal,
                    has_header=hdr,
                )
            except Exception as exc:
                errors.append(f"{fd.filename}: {exc}")
        if errors:
            messagebox.showerror("Reload errors", "\n".join(errors))
        fds = self._selected_fds()
        if fds:
            self._populate_panels(fds)
        self._redraw()
        self.app.notify_data_changed()

    def _apply_col_types(self, all_files: bool = False):
        targets = self.file_data if all_files else self._selected_fds()
        fd0 = self._get_primary_fd()
        if fd0 is None:
            return
        for i, w in enumerate(self._col_widgets):
            ctype = w["type_var"].get()
            unit  = w["unit_var"].get()
            for fd in targets:
                if i < len(fd.col_info):
                    fd.col_info[i].col_type = ctype
                    fd.col_info[i].unit     = unit
        self._refresh_axis_selectors(fd0)
        self._redraw()
        self.app.notify_data_changed()

    def _on_volume_mode_changed(self, event=None):
        regression = self._vol_mode_var.get() == "time + weight (regression)"
        self._reg_target_cb.configure(state="readonly" if regression else "disabled")
        self._flow_entry.configure(state="normal")
        self._on_solvent_changed()

    def _apply_volume_calibration(self):
        """Calibrate using selected files, then apply the resulting transform to every file."""
        targets = list(self.file_data)
        calibration_files = self._selected_fds()
        if not targets:
            messagebox.showinfo("Volume calibration", "No files loaded.")
            return
        mode = self._vol_mode_var.get()
        try:
            t_idx = next((i for i, c in enumerate(targets[0].col_info) if c.col_type == "time"), None)
            w_idx = next((i for i, c in enumerate(targets[0].col_info) if c.col_type == "weight"), None)
            if mode in ("time", "time + weight (regression)") and t_idx is None:
                raise ValueError("Select a time column.")
            if mode in ("weight", "time + weight (regression)") and w_idx is None:
                raise ValueError("Select a weight column.")
            primary = targets[0]
            density = float(self._density_var.get())
            flow = float(self._flow_var.get())
            if density <= 0 or flow <= 0:
                raise ValueError("Density and flow rate must be positive.")

            def check_column(fd, idx, expected):
                if idx is None:
                    return None
                if idx >= len(fd.col_info):
                    raise ValueError(f"{fd.filename}: selected column {idx} is missing.")
                ci = fd.col_info[idx]
                if ci.col_type != expected:
                    raise ValueError(f"{fd.filename}: column {idx} is typed as '{ci.col_type}', expected '{expected}'.")
                return ci

            # The selected column indices are global and must be valid for all files.
            for fd in targets:
                check_column(fd, t_idx, "time")
                check_column(fd, w_idx, "weight")

            result = None
            if mode == "time + weight (regression)":
                if not calibration_files:
                    raise ValueError("Select at least one calibration file for regression.")
                pairs = []
                for fd in calibration_files:
                    tc = check_column(fd, t_idx, "time")
                    wc = check_column(fd, w_idx, "weight")
                    pairs.append((fd.get_col(t_idx), fd.get_col(w_idx)))
                result = regress_mass_flow_per_file(
                    pairs, primary.col_info[t_idx].unit,
                    primary.col_info[w_idx].unit)
                if self._reg_target_var.get() == "flow":
                    flow = result.mass_flow_g_min / density
                    self._flow_var.set(f"{flow:.8g}")
                else:
                    density = result.mass_flow_g_min / flow
                    self._density_var.set(f"{density:.8g}")
                self._vol_status_var.set(
                    f"Regression from {len(calibration_files)} selected file(s): "
                    f"mass flow={result.mass_flow_g_min:.6g} g/min, "
                    f"R²={result.r2:.6f}, n={result.n_points}. "
                    f"Applied to all {len(targets)} files.")

            for fd in targets:
                if mode == "time" or (mode == "time + weight (regression)" and self._reg_target_var.get() == "flow"):
                    vol = volume_from_time(fd.get_col(t_idx), fd.col_info[t_idx].unit, flow)
                else:
                    vol = volume_from_weight(fd.get_col(w_idx), fd.col_info[w_idx].unit, density)
                old = next((i for i, c in enumerate(fd.col_info)
                            if c.raw_name == "Calculated volume"), None)
                if old is None:
                    old = fd.add_derived_column(vol, "Calculated volume", "volume", "ml")
                else:
                    fd.df_raw.iloc[:, old] = vol
                fd.analysis_x_col = old
                fd.baseline_x_col = old

            if mode != "time + weight (regression)":
                self._vol_status_var.set(
                    f"Manual {mode} conversion applied to all {len(targets)} files.")
            self.state.global_settings.density_g_ml = density
            self.state.global_settings.flow_rate_ml_min = flow
            self.state.global_settings.validate()
            self._populate_panels(self._selected_fds() or [targets[0]])
            self._redraw()
            self.app.notify_data_changed()
        except Exception as exc:
            messagebox.showerror("Volume calibration", str(exc))

    def _on_ana_x_changed(self, e=None):
        # Kept for project compatibility. Analysis X is always calculated volume.
        return

    def _on_ana_y_changed(self, e=None):
        # Kept for project compatibility. Analysis Y is always a signal column.
        return

    def _on_solvent_changed(self, *args):
        name = self._solvent_var.get()
        density = next((d for sol_name, d in SOLVENTS if sol_name == name), None)
        if density is None:
            self._density_entry.configure(state="normal")
            return
        self._populating = True
        self._density_var.set(f"{density:.3f}")
        self._populating = False
        self._density_entry.configure(state="disabled")

    def _on_wcorr_toggle(self):
        # Legacy callback retained for compatibility with older project files.
        # Weight correction is no longer a separate GUI toggle; density and flow
        # are controlled directly by the volume-calibration panel.
        return

    def _apply_conversion(self, all_files: bool = False):
        from .data_io import (weight_to_volume, time_to_volume,
                             weight_correct_time, _TO_G, _TO_MIN)
        # add_derived_column is a FileData method — no extra import needed
        try:
            density   = float(self._density_var.get())
            flow_rate = (float(self._flow_var.get())
                         if not self._wcorr_var.get() else 1.0)
        except ValueError:
            messagebox.showerror("Conversion error",
                                 "Density and flow rate must be numbers.")
            return

        weight_corr = self._wcorr_var.get()
        targets = self.file_data if all_files else self._selected_fds()
        errors  = []

        for fd in targets:
            self._density = density
            self._flow_rate = flow_rate
            self.state.global_settings.density_g_ml = density
            self.state.global_settings.flow_rate_ml_min = flow_rate
            self.state.global_settings.validate()
            fd.weight_correct = weight_corr
            xi = fd.analysis_x_col
            if xi >= len(fd.col_info):
                continue
            ci = fd.col_info[xi]
            try:
                x_raw = fd.get_col(xi).astype(float)
                if ci.col_type == "volume":
                    continue
                elif ci.col_type == "weight":
                    x_g   = x_raw * _TO_G.get(ci.unit, 1.0)
                    x_vol = weight_to_volume(x_g, density)
                elif ci.col_type == "time":
                    x_min = x_raw * _TO_MIN.get(ci.unit, 1.0)
                    if weight_corr:
                        w_col = next((j for j, c in enumerate(fd.col_info)
                                      if c.col_type == "weight"), None)
                        if w_col is not None:
                            w_raw = fd.get_col(w_col).astype(float)
                            w_g   = w_raw * _TO_G.get(fd.col_info[w_col].unit, 1.0)
                            x_vol, fr = weight_correct_time(x_min, w_g, density)
                            self._populating = True
                            self._flow_var.set(f"{fr:.4f}")
                            self._populating = False
                            self._flow_rate = fr
                        else:
                            messagebox.showwarning(
                                "No weight column",
                                f"{fd.filename}: no weight column for regression.")
                            continue
                    else:
                        x_vol = time_to_volume(x_min, flow_rate)
                else:
                    continue

                # Add as a new column — do NOT overwrite the source column.
                # This lets the user keep the original time/weight data and
                # select the new volume column independently in the plot.
                new_idx = fd.add_derived_column(x_vol, "volume", "volume", "ml")
                # Point the analysis x-axis to the new volume column
                fd.analysis_x_col = new_idx
                fd.baseline_x_col = new_idx

            except Exception as exc:
                errors.append(f"{fd.filename}: {exc}")

        if errors:
            messagebox.showerror("Conversion errors", "\n".join(errors))
        fds = self._selected_fds()
        if fds:
            self._populate_panels(fds)
        self._redraw()
        self.app.notify_data_changed()

    def _export_elution(self):
        if not self.file_data:
            messagebox.showinfo("Export", "No files loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Export elution volumes",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            export_elution_volumes(self.file_data, path)
            messagebox.showinfo("Export", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _export_analyte_data(self):
        if not self.file_data:
            messagebox.showinfo("Export", "No files loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Export analyte data",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self._compute_all_elutions()
            export_analyte_elution_data(self.file_data, path)
            messagebox.showinfo("Export", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    # ── analysis actions ──────────────────────────────────────────────────────

    def _on_toggle(self):
        if self._populating:
            return
        for fd in self._selected_fds():
            fd.invert_y          = self._invert_var.get()
            fd.x_reset           = self._xreset_var.get()
            fd.baseline_enabled  = self._bl_var.get()
            fd.smoothing_enabled = self._smooth_var.get()
            fd.fit_enabled       = self._fit_var.get()
        # Virtual column options change when steps are toggled; refresh selectors
        fd0 = self._get_primary_fd()
        if fd0:
            self._refresh_plot_selectors(fd0)
        self._redraw()
        self.app.notify_data_changed()

    def _on_std_changed(self, e=None):
        if not self._populating:
            self._apply_ana_sel("std")

    def _on_std_val_changed(self, *a):
        if self._populating:
            return
        if hasattr(self, "_std_after"):
            self.after_cancel(self._std_after)
        self._std_after = self.after(600, lambda: self._apply_ana_sel("std"))

    def _write_ana_to(self, fd: FileData, key: str):
        if key == "std":
            lbl = self._std_key_var.get()
            if lbl and lbl != _MULTI:
                fd.std_key = _std_label_to_key(lbl)
            sv = self._std_value_var.get().strip()
            if sv and sv != _MULTI:
                try:
                    fd.std_value = float(sv)
                except ValueError:
                    pass
            if fd.std_value is not None and fd.std_key and standard_needs_mw(fd.std_key):
                fd.rh = compute_rh(fd.std_key, fd.std_value)
            elif fd.std_key and not standard_needs_mw(fd.std_key):
                fd.rh = compute_rh(fd.std_key, 0.0)
            else:
                fd.rh = None

        elif key == "cut":
            _, lo, hi = self._cut_rng.get()
            fd.x_min = lo
            fd.x_max = hi

        elif key == "baseline":
            fd.baseline_enabled = self._bl_var.get()
            col_i, lo, hi = self._bl_rng.get()
            if col_i is not None:
                fd.baseline_x_col = col_i
            fd.baseline_x_min = lo
            fd.baseline_x_max = hi
            sv = self._bl_degree_var.get()
            if sv and sv != _MULTI:
                try:
                    fd.baseline_degree = int(sv)
                except ValueError:
                    pass

        elif key == "smooth":
            fd.smoothing_enabled = self._smooth_var.get()
            sv = self._smooth_win_var.get()
            if sv and sv != _MULTI:
                try:
                    fd.smoothing_window = int(sv)
                except ValueError:
                    pass

        elif key == "fit":
            fd.fit_enabled = self._fit_var.get()
            fd.fit_model   = self._fit_model_var.get()

        elif key == "momentum":
            sv = self._mom_degree_var.get()
            if sv and sv != _MULTI:
                try:
                    fd.momentum_degree = int(sv)
                except ValueError:
                    pass
            sv = self._mom_thresh_var.get()
            if sv and sv != _MULTI:
                try:
                    fd.momentum_threshold = float(sv)
                except ValueError:
                    pass

    def _apply_ana_sel(self, key: str):
        """Apply to all SELECTED files."""
        for fd in self._selected_fds():
            self._write_ana_to(fd, key)
        if key == "std":
            self._update_rh_label(self._selected_fds())
        # Virtual column availability changes with analysis state
        fd0 = self._get_primary_fd()
        if fd0:
            self._refresh_plot_selectors(fd0)
        self._redraw()
        self.app.notify_data_changed()

    def _apply_ana_all_files(self, key: str):
        """Apply to ALL loaded files (even unselected)."""
        for fd in self.file_data:
            self._write_ana_to(fd, key)
        if key == "std":
            self._update_rh_label(self._selected_fds())
        fd0 = self._get_primary_fd()
        if fd0:
            self._refresh_plot_selectors(fd0)
        self._redraw()
        self.app.notify_data_changed()

    # ── rh label ─────────────────────────────────────────────────────────────

    def _update_rh_label(self, fds: list[FileData]):
        if not fds:
            self._rh_label.configure(text="Rh = — nm")
            return
        fd = fds[0]
        if fd.rh is not None:
            ve = f", Ve = {fd.elution_volume:.4f} ml" if fd.elution_volume else ""
            self._rh_label.configure(text=f"Rh = {fd.rh:.4f} nm{ve}")
        else:
            self._rh_label.configure(text="Rh = — nm")

    # ── public API ────────────────────────────────────────────────────────────

    def _compute_all_elutions(self):
        """
        Run analyze_file() for every loaded FileData and write the result
        into fd.elution_volume.  Does NOT draw anything.  Called before
        exporting data to Tab 2 so that all files with an auto-detected
        standard appear immediately without needing to be plotted first.
        """
        for fd in self.file_data:
            try:
                res = analyze_file(fd)
                ev = res.get("elution_x") if res else None
                if ev is not None:
                    fd.elution_volume = ev
            except Exception:
                pass

    def get_elution_data(self) -> list[dict]:
        # Ensure all files have up-to-date elution volumes before returning
        self._compute_all_elutions()
        return [
            {"filename": fd.filename, "std_key": fd.std_key,
             "std_value": fd.std_value, "rh": fd.rh, "ve": fd.elution_volume}
            for fd in self.file_data
            if fd.rh is not None and fd.elution_volume is not None
        ]

    def refresh_all(self):
        self._redraw()


# ── module-level helpers ──────────────────────────────────────────────────────

def _std_key_to_label(key: str) -> str:
    if not key:
        return ""
    try:
        return STANDARD_LABELS[STANDARD_KEYS.index(key)]
    except (ValueError, IndexError):
        return ""


def _std_label_to_key(label: str) -> str:
    if not label:
        return ""
    try:
        return STANDARD_KEYS[STANDARD_LABELS.index(label)]
    except (ValueError, IndexError):
        return ""
