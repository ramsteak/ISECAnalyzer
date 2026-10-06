"""
gui_tab2.py
===========
Tab 2: ISEC pore-model fitting.

Layout:
  ┌─LeftSidebar ─────┬─── Main Plot Area ─────────┬─ RightSidebar ─┐
  │ Standards table  │  Exclusion curve + PSD plot │ Fit settings   │
  │ [Manual add]     │                             │                │
  └──────────────────┴─────────────────────────────┴────────────────┘
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .fitting import ISECFitter, fit_knox_scott, FitResult, KnoxScottResult
from .standards import STANDARD_KEYS, STANDARD_LABELS, compute_rh, standard_needs_mw

ACCENT = "#1565C0"
BG     = "#F5F5F5"
PANEL  = "#FFFFFF"
BORDER = "#E0E0E0"
TEXT   = "#212121"
MUTED  = "#757575"
RED_HL = "#C62828"
GREEN  = "#2E7D32"
ORANGE = "#E65100"


class Tab2(ttk.Frame):
    """Main frame for Tab 2."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._fit_result = None
        self._ks_result  = None

        self._build_layout()

    # ------------------------------------------------------------------
    def _build_layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._left  = self._make_left_sidebar()
        self._left.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)

        self._center = self._make_center()
        self._center.grid(row=0, column=1, sticky="nsew", padx=2, pady=4)

        self._right = self._make_right_sidebar()
        self._right.grid(row=0, column=2, sticky="nsew", padx=(2, 4), pady=4)

    # ------------------------------------------------------------------
    # Left sidebar
    # ------------------------------------------------------------------

    def _make_left_sidebar(self) -> ttk.Frame:
        frm = ttk.LabelFrame(self, text="Standards & Elution Volumes", padding=4)
        frm.configure(width=300)
        frm.grid_propagate(False)
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        # Treeview: filename / std / Rh / Ve
        cols = ("std", "rh", "ve")
        tree = ttk.Treeview(frm, columns=cols, show="headings", selectmode="browse", height=18)
        tree.heading("std",  text="Standard")
        tree.heading("rh",   text="Rh (nm)")
        tree.heading("ve",   text="Ve (ml)")
        tree.column("std",   width=110)
        tree.column("rh",    width=80)
        tree.column("ve",    width=80)

        sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.grid(row=1, column=0, sticky="nsew")
        sb.grid(row=1, column=1, sticky="ns")
        self._tree = tree

        # Sort state: (column_id, ascending)
        self._sort_col = None
        self._sort_asc = True

        # Bind heading clicks for sorting
        for col in cols:
            tree.heading(col, command=lambda c=col: self._sort_by(c))

        # Selection and clear-on-empty-click
        tree.bind("<<TreeviewSelect>>", self._on_row_select)
        tree.bind("<Button-1>",         self._on_tree_click)

        # Status label
        self._status_label = ttk.Label(frm, text="", foreground=MUTED, font=("Segoe UI", 8))
        self._status_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))

        # Buttons
        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        btn_frm.columnconfigure(0, weight=1)
        btn_frm.columnconfigure(1, weight=1)

        ttk.Button(btn_frm, text="+ Add manual row",
                   command=self._add_manual_row).grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(btn_frm, text="Edit selected",
                   command=self._edit_row).grid(row=0, column=1, sticky="ew", padx=1, pady=1)

        # Right-click context menu on the tree
        self._tree.bind("<Button-3>", self._on_tree_rightclick)

        # Column entries for adding global V0 / Vc
        sep = ttk.Separator(frm, orient=tk.HORIZONTAL)
        sep.grid(row=3, column=0, columnspan=2, sticky="ew", pady=6)

        param_frm = ttk.Frame(frm)
        param_frm.grid(row=4, column=0, columnspan=2, sticky="ew")
        param_frm.columnconfigure(1, weight=1)

        # V0 row with two helper buttons
        ttk.Label(param_frm, text="V₀ (dead volume, ml):").grid(
            row=0, column=0, sticky="w")
        v0_row = ttk.Frame(param_frm)
        v0_row.grid(row=0, column=1, sticky="ew", padx=4)
        v0_row.columnconfigure(0, weight=1)
        self._v0_var = tk.StringVar(value="0.0")
        v0_entry = ttk.Entry(v0_row, textvariable=self._v0_var, width=7)
        v0_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(v0_row, text="fit", width=3,
                   command=self._fit_v0, takefocus=0).grid(
            row=0, column=1, padx=(2, 1))
        ttk.Button(v0_row, text="min", width=3,
                   command=self._v0_from_min, takefocus=0).grid(
            row=0, column=2, padx=(1, 0))

        ttk.Label(param_frm, text="Vc (column volume, ml):").grid(
            row=1, column=0, sticky="w")
        self._vc_var = tk.StringVar(value="10.0")
        vc_entry = ttk.Entry(param_frm, textvariable=self._vc_var, width=8)
        vc_entry.grid(row=1, column=1, sticky="ew", padx=4)

        ttk.Label(param_frm, text="Sample weight (g):").grid(
            row=2, column=0, sticky="w")
        self._weight_var = tk.StringVar(value="1.0")
        weight_entry = ttk.Entry(param_frm, textvariable=self._weight_var, width=8)
        weight_entry.grid(row=2, column=1, sticky="ew", padx=4)

        ttk.Label(param_frm, text="Vz (zero-col. vol., ml):").grid(
            row=3, column=0, sticky="w")
        self._vz_var = tk.StringVar(value="0.0")
        vz_entry = ttk.Entry(param_frm, textvariable=self._vz_var, width=8)
        vz_entry.grid(row=3, column=1, sticky="ew", padx=4)

        # Contextual description label — updates when an entry gains focus
        self._param_desc_var = tk.StringVar(
            value="V₀: dead volume of the column system (elution of fully excluded species).")
        param_desc = ttk.Label(param_frm, textvariable=self._param_desc_var,
                               font=("Segoe UI", 7), foreground=MUTED, wraplength=260)
        param_desc.grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))

        _DESCS = {
            "v0":     "V₀: dead (hold-up) volume — elution volume of a fully excluded species.",
            "vc":     "Vc: total column volume (bed volume). Used to compute residual pore volume.",
            "weight": "Sample weight (g) — used to compute specific pore volumes (ml/g).",
            "vz":     ("Vz: extra-column (zero-column) volume measured with a zero-volume joiner. "
                       "Subtracted from all Ve values before fitting to remove instrument dead volume."),
        }
        v0_entry.bind("<FocusIn>",
            lambda e: self._param_desc_var.set(_DESCS["v0"]))
        vc_entry.bind("<FocusIn>",
            lambda e: self._param_desc_var.set(_DESCS["vc"]))
        weight_entry.bind("<FocusIn>",
            lambda e: self._param_desc_var.set(_DESCS["weight"]))
        vz_entry.bind("<FocusIn>",
            lambda e: self._param_desc_var.set(_DESCS["vz"]))

        # V0 fit state flag (set by _fit_v0 button)
        self._v0_fit_enabled = False

        return frm

    def _on_row_select(self, event=None):
        """Re-draw the exclusion curve to highlight the selected standard."""
        self._redraw_excl()

    def _on_tree_click(self, event):
        """Clear the selection when the user clicks on empty space below items."""
        region = self._tree.identify_region(event.x, event.y)
        if region not in ("cell", "tree"):
            self._tree.selection_remove(self._tree.selection())
            self._redraw_excl()

    def _on_tree_rightclick(self, event):
        """Context menu: Enable/Disable and Remove."""
        item = self._tree.identify_row(event.y)
        if not item:
            return
        # Ensure the right-clicked row is selected
        self._tree.selection_set(item)

        tags = set(self._tree.item(item, "tags"))
        is_disabled = "disabled" in tags
        toggle_label = "Enable" if is_disabled else "Disable"

        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label=toggle_label,
                         command=lambda: self._toggle_disabled(item))
        menu.add_separator()
        menu.add_command(label="Remove",
                         command=lambda: self._remove_row_item(item))
        menu.tk_popup(event.x_root, event.y_root)

    def _toggle_disabled(self, item: str):
        """Toggle the 'disabled' state of a tree row."""
        tags = set(self._tree.item(item, "tags"))
        if "disabled" in tags:
            tags.discard("disabled")
        else:
            tags.add("disabled")
        self._tree.item(item, tags=tuple(tags))
        self._apply_row_tags()
        self._redraw_excl()

    def _apply_row_tags(self):
        """
        Re-apply all tag styles so that disabled rows always appear with
        a distinct striped background regardless of their origin tag.
        Call after any tag change.
        """
        # Base colours for origin tags
        self._tree.tag_configure("auto",     foreground=ACCENT,  background="")
        self._tree.tag_configure("manual",   foreground=ORANGE,  background="")
        # Disabled: grey text + light salmon background to make it unmissable.
        # In ttk.Treeview, the last tag in the item's tag list takes priority
        # for conflicting attributes, so we always put "disabled" last.
        self._tree.tag_configure("disabled", foreground=MUTED, background="#FFE4E1")
        # Re-order each item's tags so "disabled" is last if present
        for item in self._tree.get_children():
            tags = list(self._tree.item(item, "tags"))
            if "disabled" in tags:
                tags = [t for t in tags if t != "disabled"] + ["disabled"]
                self._tree.item(item, tags=tuple(tags))

    def _remove_row_item(self, item: str):
        """Delete a specific tree item."""
        self._tree.delete(item)
        self._redraw_excl()

    def _sort_by(self, col: str):
        """Sort tree rows by *col*, toggling direction on repeated clicks."""
        # Collect all rows as (sort_key, item_id, values, tags)
        rows = []
        for item in self._tree.get_children():
            vals = self._tree.item(item, "values")
            tags = self._tree.item(item, "tags")
            # Numeric sort key for rh and ve; string for std
            try:
                key = float(vals[{"std": 0, "rh": 1, "ve": 2}[col]])
            except (ValueError, KeyError):
                key = str(vals[{"std": 0, "rh": 1, "ve": 2}.get(col, 0)])
            rows.append((key, item, vals, tags))

        # Toggle direction if same column clicked again
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True

        rows.sort(key=lambda t: t[0], reverse=not self._sort_asc)

        # Re-insert in sorted order
        for pos, (_, item, vals, tags) in enumerate(rows):
            self._tree.move(item, "", pos)

        # Update heading indicators
        col_names = {"std": "Standard", "rh": "Rh (nm)", "ve": "Ve (ml)"}
        for c, base in col_names.items():
            if c == col:
                arrow = " ▲" if self._sort_asc else " ▼"
                self._tree.heading(c, text=base + arrow)
            else:
                self._tree.heading(c, text=base)

    def _add_manual_row(self):
        dialog = _ManualRowDialog(self, title="Add Standard")
        if dialog.result:
            std_key, std_val, rh, ve = dialog.result
            label = f"{std_key.upper()}" + (f" {std_val}" if std_val else "")
            self._tree.insert("", tk.END, values=(label, f"{rh:.4f}", f"{ve:.4f}"),
                              tags=("manual",))
            self._apply_row_tags()

    def _remove_row(self):
        """Remove the currently selected row (called from edit flow if needed)."""
        sel = self._tree.selection()
        if sel:
            self._remove_row_item(sel[0])

    def _edit_row(self):
        sel = self._tree.selection()
        if not sel:
            return
        item = sel[0]
        vals = self._tree.item(item, "values")
        # Parse existing values
        try:
            rh_cur = float(vals[1])
            ve_cur = float(vals[2])
        except (ValueError, IndexError):
            rh_cur, ve_cur = 0.0, 0.0

        dialog = _ManualRowDialog(self, title="Edit Standard",
                                   initial_rh=rh_cur, initial_ve=ve_cur)
        if dialog.result:
            std_key, std_val, rh, ve = dialog.result
            label = f"{std_key.upper()}" + (f" {std_val}" if std_val else "")
            self._tree.item(item, values=(label, f"{rh:.4f}", f"{ve:.4f}"))

    # ------------------------------------------------------------------
    # Center plot area
    # ------------------------------------------------------------------

    def _make_center(self) -> ttk.Frame:
        frm = ttk.LabelFrame(self, text="ISEC Fitting", padding=4)
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        fig = Figure(figsize=(7, 5), dpi=100, facecolor=PANEL)
        self.ax_excl  = fig.add_subplot(121)
        self.ax_psd   = fig.add_subplot(122)
        fig.tight_layout(pad=1.5)
        self._fig = fig

        canvas = FigureCanvasTkAgg(fig, master=frm)
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._canvas = canvas

        toolbar_frm = ttk.Frame(frm)
        toolbar_frm.grid(row=0, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frm)
        toolbar.update()

        # Results table: independently scrollable and resizable
        res_frm = ttk.LabelFrame(frm, text="Fit Results", padding=4)
        res_frm.grid(row=2, column=0, sticky="nsew", pady=(4, 0))
        res_frm.rowconfigure(0, weight=1); res_frm.columnconfigure(0, weight=1)
        cols = ("parameter", "volume", "ci95", "spec_volume", "spec_surface")
        self._result_tree = ttk.Treeview(res_frm, columns=cols, show="headings", height=8)
        headings = {"parameter":"Pore parameter", "volume":"Volume (ml)",
                    "ci95":"95% uncertainty (ml)", "spec_volume":"Specific volume (ml/g)",
                    "spec_surface":"Specific surface (m²/g)"}
        widths = {"parameter":130, "volume":100, "ci95":150, "spec_volume":150, "spec_surface":150}
        for c in cols:
            self._result_tree.heading(c, text=headings[c])
            self._result_tree.column(c, width=widths[c], minwidth=70, stretch=True, anchor="e")
        sbv = ttk.Scrollbar(res_frm, orient=tk.VERTICAL, command=self._result_tree.yview)
        sbh = ttk.Scrollbar(res_frm, orient=tk.HORIZONTAL, command=self._result_tree.xview)
        self._result_tree.configure(yscrollcommand=sbv.set, xscrollcommand=sbh.set)
        self._result_tree.grid(row=0, column=0, sticky="nsew")
        sbv.grid(row=0, column=1, sticky="ns")
        sbh.grid(row=1, column=0, sticky="ew")
        self._result_summary = tk.StringVar(value="No fit results yet")
        ttk.Label(res_frm, textvariable=self._result_summary).grid(row=2, column=0, columnspan=2, sticky="w", pady=(3,0))

        return frm

    def _set_result_text(self, text: str):
        # Kept as a compatibility helper for non-tabular summaries.
        self._result_summary.set(text.splitlines()[0] if text else "No fit results yet")

    def _populate_result_table(self, res):
        for item in self._result_tree.get_children():
            self._result_tree.delete(item)
        if isinstance(res, FitResult):
            for i, p in enumerate(res.pore_params):
                if True:
                    lo, hi = res.ci_lower[i], res.ci_upper[i]
                    ci = f"±{0.5 * (hi - lo):.4f}" if np.isfinite(lo) and np.isfinite(hi) else "n/a"
                    self._result_tree.insert("", tk.END, values=(
                        f"{p:.4f}", f"{res.pore_volumes[i]:.4f}", ci,
                        f"{res.spec_volumes[i]:.4f}", f"{res.spec_surface[i]:.4f}"))
            self._result_summary.set(f"Method: Jerabek | Model: {res.model} | R²={res.r2:.6f} | iterations={res.n_iter} | converged={res.converged} | residual volume={res.residual_volume:.4f} ml")
        elif isinstance(res, KnoxScottResult):
            for i, (p, lo, hi) in enumerate(zip(res.popt, res.ci_lower, res.ci_upper)):
                ci = f"±{0.5 * (hi - lo):.5g}" if np.isfinite(lo) and np.isfinite(hi) else "n/a"
                self._result_tree.insert("", tk.END, values=(f"p{i}", f"{p:.5g}", ci, "", ""))
            self._result_summary.set(f"Method: Knox-Scott | R²={res.r2:.6f} | iterations={res.n_iter} | converged={res.converged} | SSR={res.ssr:.6g}")

    # ------------------------------------------------------------------
    # Right sidebar
    # ------------------------------------------------------------------

    def _make_right_sidebar(self) -> ttk.Frame:
        frm = ttk.Frame(self, width=280)
        frm.grid_propagate(False)
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        outer = ttk.Frame(frm)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        self._build_right_widgets(inner)
        return frm

    def _build_right_widgets(self, parent):
        parent.columnconfigure(0, weight=1)

        # --- Fit method ---
        fit_frm = ttk.LabelFrame(parent, text="Fit Method", padding=6)
        fit_frm.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        fit_frm.columnconfigure(1, weight=1)

        ttk.Label(fit_frm, text="Fit method:").grid(row=0, column=0, sticky="w")
        self._fit_kind_var = tk.StringVar(value="Jerabek cylindrical")
        self._show_errors_var = tk.BooleanVar(value=True)
        ttk.Combobox(fit_frm, textvariable=self._fit_kind_var,
                     values=["Knox-scott", "Jerabek cylindrical", "Jerabek Ogston"],
                     state="readonly", width=22).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        self._fit_kind_var.trace_add("write", self._on_fit_kind_changed)
        ttk.Checkbutton(
            fit_frm, text="Show fitting errors",
            variable=self._show_errors_var,
            command=self._toggle_error_display
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=2, pady=(3, 0))

        # --- Jerabek pore sizes ---
        self._jerabek_frm = ttk.LabelFrame(parent, text="Jerabek Pore Sizes (nm)", padding=6)
        self._jerabek_frm.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self._jerabek_frm.columnconfigure(0, weight=1)

        ttk.Label(self._jerabek_frm,
                  text="Enter pore radii separated by commas or spaces.\n"
                       "Or use the range generator below.",
                  font=("Segoe UI", 8), foreground=MUTED, wraplength=200).grid(
            row=0, column=0, columnspan=2, sticky="w")

        self._pore_text = tk.Text(self._jerabek_frm, height=4, width=22,
                                  font=("Courier New", 8), relief="solid", borderwidth=1)
        self._pore_text.insert("1.0", "2 5 10 15 20 30 40 60 80 100 150")
        self._pore_text.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

        rng_frm = ttk.Frame(self._jerabek_frm)
        rng_frm.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(rng_frm, text="Start:").grid(row=0, column=0, sticky="w")
        self._pore_start = tk.StringVar(value="2")
        ttk.Entry(rng_frm, textvariable=self._pore_start, width=5).grid(row=0, column=1, padx=2)
        ttk.Label(rng_frm, text="End:").grid(row=0, column=2, sticky="w")
        self._pore_end = tk.StringVar(value="150")
        ttk.Entry(rng_frm, textvariable=self._pore_end, width=5).grid(row=0, column=3, padx=2)
        ttk.Label(rng_frm, text="Step:").grid(row=1, column=0, sticky="w")
        self._pore_step = tk.StringVar(value="10")
        ttk.Entry(rng_frm, textvariable=self._pore_step, width=5).grid(row=1, column=1, padx=2)
        ttk.Button(rng_frm, text="Generate", command=self._generate_pore_range).grid(
            row=1, column=2, columnspan=2, sticky="ew", padx=2)

        # --- Ogston ---
        self._ogston_frm = ttk.LabelFrame(parent, text="Ogston Parameters", padding=6)
        ttk.Label(self._ogston_frm, text="Chain diameter (nm):").grid(row=0, column=0, sticky="w")
        self._chain_diam_var = tk.StringVar(value="0.4")
        ttk.Entry(self._ogston_frm, textvariable=self._chain_diam_var, width=8).grid(
            row=0, column=1, padx=4)
        ttk.Label(self._ogston_frm,
                  text="Chain concentrations (nm/nm³):", font=("Segoe UI", 8)).grid(
            row=1, column=0, columnspan=2, sticky="w")
        self._chain_conc_text = tk.Text(self._ogston_frm, height=3, width=22,
                                        font=("Courier New", 8), relief="solid", borderwidth=1)
        self._chain_conc_text.insert("1.0", "0.05 0.1 0.2 0.4 0.8 1.5")
        self._chain_conc_text.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)

        # Knox-Scott params
        self._ks_frm = ttk.LabelFrame(parent, text="Knox-Scott Parameters", padding=6)
        self._ks_frm.columnconfigure(1, weight=1)
        ttk.Label(self._ks_frm, text="Numerator degree:").grid(row=0, column=0, sticky="w")
        self._ks_nnu = tk.StringVar(value="1")
        ttk.Entry(self._ks_frm, textvariable=self._ks_nnu, width=5).grid(row=0, column=1, padx=4)
        ttk.Label(self._ks_frm, text="Denominator degree:").grid(row=1, column=0, sticky="w")
        self._ks_nde = tk.StringVar(value="3")
        ttk.Entry(self._ks_frm, textvariable=self._ks_nde, width=5).grid(row=1, column=1, padx=4)

        # Initially show jerabek, hide ogston/ks
        self._ks_frm.grid(row=3, column=0, sticky="ew", padx=4, pady=4)
        self._ogston_frm.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        self._on_fit_kind_changed()

        # --- Run button ---
        run_frm = ttk.Frame(parent)
        run_frm.grid(row=4, column=0, sticky="ew", padx=4, pady=4)
        run_frm.columnconfigure(0, weight=1)
        ttk.Button(run_frm, text="▶ Run fit", command=self._run_fit).grid(
            row=0, column=0, sticky="ew", ipady=4)

        # --- Export ---
        exp_frm = ttk.LabelFrame(parent, text="Export", padding=6)
        exp_frm.grid(row=5, column=0, sticky="ew", padx=4, pady=4)
        exp_frm.columnconfigure(0, weight=1)
        ttk.Button(exp_frm, text="Save figure…",
                   command=self._export_figure).grid(row=0, column=0, sticky="ew", pady=1)
        ttk.Button(exp_frm, text="Create PDF report…",
                   command=self._export_pdf_report).grid(row=1, column=0, sticky="ew", pady=1)
        ttk.Button(exp_frm, text="Export results (CSV/XLSX)…",
                   command=self._export_results).grid(row=2, column=0, sticky="ew", pady=1)

    def _fit_v0(self):
        """
        Toggle V0 fitting mode.  When active the button text changes and the
        next Jerabek fit will optimise V0 jointly with the pore volumes.
        """
        self._v0_fit_enabled = not self._v0_fit_enabled
        # Visual feedback: update the V0 entry background
        # (we find the entry through the widget tree)
        state = "active" if self._v0_fit_enabled else "inactive"
        # Locate v0_entry via the v0_row frame child
        try:
            for widget in self.winfo_children():
                pass  # just ensure we can iterate
        except Exception:
            pass
        msg = "V0 fitting ON — V0 will be optimised during the next Jerabek fit." if self._v0_fit_enabled               else "V0 fitting OFF — V0 is treated as fixed."
        self._param_desc_var.set(msg)

    def _v0_from_min(self):
        """
        Set V0 to the smallest enabled Ve value minus Vz.
        This approximates the dead volume from the most-excluded standard.
        """
        Vz = self._get_float(self._vz_var, 0.0)
        ve_vals = []
        for item in self._tree.get_children():
            if "disabled" in set(self._tree.item(item, "tags")):
                continue
            vals = self._tree.item(item, "values")
            try:
                ve_vals.append(float(vals[2]))
            except (ValueError, IndexError):
                pass
        if not ve_vals:
            messagebox.showinfo("No data", "No enabled standards in the table.")
            return
        v0_est = min(ve_vals) - Vz
        self._v0_var.set(f"{v0_est:.4f}")
        self._param_desc_var.set(
            "V0 set to min(Ve) - Vz = " + f"{min(ve_vals):.4f}" + " - " + f"{Vz:.4f}" + " = " + f"{v0_est:.4f}" + " ml.")

    def _toggle_error_display(self):
        """Show or hide fitted uncertainty visualizations without refitting."""
        if isinstance(self._fit_result, FitResult):
            self._draw_jerabek(self._fit_result)
        elif isinstance(self._ks_result, KnoxScottResult):
            self._draw_knox_scott(self._ks_result)

    def _on_fit_kind_changed(self, *args):
        kind = self._fit_kind_var.get()
        is_ks = kind == "Knox-scott"
        is_og = kind == "Jerabek Ogston"
        if is_ks:
            self._jerabek_frm.grid_remove(); self._ks_frm.grid()
        else:
            self._ks_frm.grid_remove(); self._jerabek_frm.grid()
        if is_og: self._ogston_frm.grid()
        else: self._ogston_frm.grid_remove()

    def _generate_pore_range(self):
        try:
            start = float(self._pore_start.get())
            end   = float(self._pore_end.get())
            step  = float(self._pore_step.get())
            pores = np.arange(start, end + step * 0.5, step)
            self._pore_text.delete("1.0", tk.END)
            self._pore_text.insert("1.0", " ".join(f"{p:.1f}" for p in pores))
        except ValueError:
            messagebox.showerror("Range error", "Start, end and step must be numbers.")

    # ------------------------------------------------------------------
    # Public: update from Tab 1
    # ------------------------------------------------------------------

    def update_from_tab1(self, elution_data: list[dict]):
        """Refresh the standards table from Tab 1 computed data."""
        # Clear auto rows (those without 'manual' tag)
        for item in self._tree.get_children():
            tags = self._tree.item(item, "tags")
            if "manual" not in tags:
                self._tree.delete(item)

        for d in elution_data:
            label = f"{d['std_key'].upper()}" + (f" {d['std_value']}" if d['std_value'] else "")
            self._tree.insert("", tk.END,
                              values=(label, f"{d['rh']:.4f}", f"{d['ve']:.4f}"),
                              tags=("auto",))
        self._apply_row_tags()
        self._status_label.configure(
            text=f"{len(elution_data)} standard(s) from Tab 1")

    # ------------------------------------------------------------------
    # Collect data from tree
    # ------------------------------------------------------------------

    def _collect_data(self) -> tuple[list[float], list[float], list[str]] | None:
        """
        Collect enabled rows for fitting.

        Returns (Rh_list, Ve_corrected_list, fitted_item_ids) or None.
        Ve values are corrected for Vz (zero-column volume) before fitting.
        Disabled rows are skipped but remembered for plot colouring.
        """
        Vz = self._get_float(self._vz_var, 0.0)
        Rh_list: list[float] = []
        Ve_list: list[float] = []
        fitted_items: list[str] = []

        for item in self._tree.get_children():
            tags = set(self._tree.item(item, "tags"))
            if "disabled" in tags:
                continue
            vals = self._tree.item(item, "values")
            try:
                Rh_list.append(float(vals[1]))
                Ve_list.append(float(vals[2]) - Vz)
                fitted_items.append(item)
            except (ValueError, IndexError):
                continue

        if len(Rh_list) < 2:
            messagebox.showwarning("Insufficient data",
                                   "At least 2 enabled data points are needed to run the fit.")
            return None
        # Cache which items were used so _selected_row_index can map correctly
        self._fit_row_items = fitted_items
        return Rh_list, Ve_list, fitted_items

    def _get_float(self, var: tk.StringVar, default: float) -> float:
        try:
            return float(var.get())
        except ValueError:
            return default

    # ------------------------------------------------------------------
    # Run fit
    # ------------------------------------------------------------------

    def _run_fit(self):
        data = self._collect_data()
        if data is None:
            return
        Rh, Ve, _ = data   # third element is fitted_items (already cached)

        V0     = self._get_float(self._v0_var, 0.0)
        Vc     = self._get_float(self._vc_var, 10.0)
        weight = self._get_float(self._weight_var, 1.0)
        kind = self._fit_kind_var.get()
        model = "ogston" if kind == "Jerabek Ogston" else "cylindrical"
        try:
            if kind == "Knox-scott": self._run_knox_scott(Rh, Ve, weight)
            else: self._run_jerabek(Rh, Ve, V0, Vc, weight, model)
        except Exception as exc:
            messagebox.showerror("Fit error", str(exc))

    def _run_jerabek(self, Rh, Ve, V0, Vc, weight, model):
        # Parse pore sizes
        pore_text = self._pore_text.get("1.0", tk.END).strip()
        try:
            import re
            pore_params = [float(x) for x in re.split(r"[\s,;]+", pore_text) if x]
        except ValueError:
            messagebox.showerror("Pore size error", "Pore sizes must be numbers.")
            return

        chain_diam = self._get_float(self._chain_diam_var, 0.4)
        if model == "ogston":
            conc_text = self._chain_conc_text.get("1.0", tk.END).strip()
            try:
                import re
                pore_params = [float(x) for x in re.split(r"[\s,;]+", conc_text) if x]
            except ValueError:
                messagebox.showerror("Chain conc. error", "Values must be numbers.")
                return

        # Validity check: for the cylindrical model, analytes are sensitive to
        # pores larger than their Rh.  To properly characterise the largest
        # analyte (Rh_max) the pore list should include pores at least as large
        # as Rh_max.  Conventionally a safety factor of 2× is recommended
        # (pores 2× the analyte size ensure the analyte is clearly in the
        # partially-excluded regime, not at the margin).
        if model == "cylindrical" and pore_params:
            rh_max   = max(Rh)
            pore_max = max(pore_params)
            if pore_max < 2 * rh_max:
                limit = 2 * rh_max
                msg = (
                    "The largest pore radius (" + f"{pore_max:.1f}" + " nm) is smaller than\n"
                    "2x the largest analyte Rh (" + f"{rh_max:.4f}" + " nm, 2x = " + f"{limit:.4f}" + " nm).\n\n"
                    "The largest standard may be partially or fully excluded from all\n"
                    "pores in the list, which can produce unreliable fit results.\n\n"
                    "Consider adding pore sizes up to at least " + f"{limit:.1f}" + " nm.\n\n"
                    "Continue with the current pore list?"
                )
                if not messagebox.askyesno("Pore range warning", msg,
                                           icon="warning", default="yes"):
                    return

        fitter = ISECFitter(
            Rh=Rh, Ve=Ve, V0=V0, Vc=Vc, model=model,
            pore_params=pore_params,
            chain_diam=chain_diam,
            weight=weight,
            fit_v0=getattr(self, "_v0_fit_enabled", False),
        )
        res = fitter.fit()
        # If V0 was fitted, update the V0 entry with the optimised value
        if getattr(self, "_v0_fit_enabled", False) and hasattr(res, "V0"):
            self._v0_var.set(f"{res.V0:.4f}")
        self._fit_result = res
        self._ks_result  = None
        self._draw_jerabek(res)
        self._format_jerabek_result(res)
        self._populate_result_table(res)

    def _run_knox_scott(self, Rh, Ve, weight):
        nnu = int(self._ks_nnu.get()) if self._ks_nnu.get().isdigit() else 1
        nde = int(self._ks_nde.get()) if self._ks_nde.get().isdigit() else 3
        res = fit_knox_scott(Rh, Ve, weight=weight, nnu=nnu, nde=nde)
        self._ks_result  = res
        self._fit_result = None
        self._draw_knox_scott(res)
        self._format_ks_result(res)
        self._populate_result_table(res)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _selected_row_index(self) -> int | None:
        """
        Return the 0-based index of the selected row within the fitted data
        arrays (res.Rh, res.Ve_obs), or None if the row is not in the fit
        (e.g. it is disabled or nothing is selected).

        Uses self._fit_row_items — the ordered list of item IDs that were
        passed to the fitter — which is set by _collect_data each time a
        fit is run.
        """
        sel = self._tree.selection()
        if not sel:
            return None
        fitted = getattr(self, "_fit_row_items", [])
        try:
            return fitted.index(sel[0])
        except ValueError:
            return None  # selected row was disabled / not in last fit

    def _redraw_excl(self):
        """
        Redraw only the exclusion-curve axes from the cached fit result,
        applying a highlight to the currently selected row.
        Called both after a fit and on selection changes.
        """
        if self._fit_result is not None:
            self._draw_excl_jerabek(self._fit_result)
        elif self._ks_result is not None:
            self._draw_excl_knox_scott(self._ks_result)
        # else: nothing to draw yet

    def _disabled_rh_ve(self) -> tuple[list[float], list[float]]:
        """Return (Rh_list, Ve_list) for disabled rows, for grey plot markers."""
        Vz = self._get_float(self._vz_var, 0.0)
        rh_list, ve_list = [], []
        for item in self._tree.get_children():
            if "disabled" in set(self._tree.item(item, "tags")):
                vals = self._tree.item(item, "values")
                try:
                    rh_list.append(float(vals[1]))
                    ve_list.append(float(vals[2]) - Vz)
                except (ValueError, IndexError):
                    pass
        return rh_list, ve_list

    def _draw_excl_jerabek(self, res: FitResult):
        """Draw the Jerabek exclusion curve axes, highlighting the selected point."""
        ax1 = self.ax_excl
        ax1.cla()

        sel_idx = self._selected_row_index()
        Rh      = np.asarray(res.Rh)
        Ve_obs  = np.asarray(res.Ve_obs)
        Ve_fit  = np.asarray(res.Ve_fit)

        if sel_idx is not None and 0 <= sel_idx < len(Rh):
            mask_rest = np.ones(len(Rh), dtype=bool)
            mask_rest[sel_idx] = False
            if mask_rest.any():
                ax1.scatter(Rh[mask_rest], Ve_obs[mask_rest],
                            color=RED_HL, zorder=5, s=40, marker="x",
                            linewidths=1.5, label="Observed")
            ax1.scatter([Rh[sel_idx]], [Ve_obs[sel_idx]],
                        color=ORANGE, zorder=8, s=120, marker="x",
                        linewidths=2.5, label="Selected")
        else:
            ax1.scatter(Rh, Ve_obs, color=RED_HL, zorder=5,
                        s=40, marker="x", linewidths=1.5, label="Observed")

        ax1.scatter(Rh, Ve_fit, color=ACCENT, zorder=4,
                    s=20, alpha=0.8, label="Fitted")

        # Disabled rows: grey x markers (not in fit, but shown for reference)
        dis_rh, dis_ve = self._disabled_rh_ve()
        if dis_rh:
            ax1.scatter(dis_rh, dis_ve, color=MUTED, zorder=3, s=40, marker="x",
                        linewidths=1.5, alpha=0.6, label="Disabled")

        rang = np.geomspace(max(0.05, 0.9 * Rh.min()), 1.1 * Rh.max(), 500)
        if res.model == "cylindrical":
            from .fitting import K_cylindrical
            Ve_curve = np.dot(K_cylindrical(rang, res.pore_params), res.pore_volumes) + res.V0
        else:
            from .fitting import K_ogston
            Ve_curve = np.dot(K_ogston(rang, res.pore_params), res.pore_volumes) + res.V0

        ax1.plot(rang, Ve_curve, color=ACCENT, linewidth=1.2, alpha=0.8, label="Fitted curve")
        if (self._show_errors_var.get() and
                np.all(np.isfinite(res.ve_ci_lower)) and
                np.all(np.isfinite(res.ve_ci_upper))):
            ax1.fill_between(rang, res.ve_ci_lower, res.ve_ci_upper,
                             color=ACCENT, alpha=0.18, label="95% CI")
        ax1.set_xscale("log")
        ax1.set_xlabel("Rh (nm)", fontsize=8)
        ax1.set_ylabel("Ve (ml)", fontsize=8)
        ax1.set_title("Exclusion Curve", fontsize=9)
        ax1.legend(fontsize=7)
        ax1.grid(True, which="both", alpha=0.3)

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()

    def _draw_excl_knox_scott(self, res: KnoxScottResult):
        """Draw the Knox-Scott exclusion curve axes, highlighting the selected point."""
        ax1 = self.ax_excl
        ax1.cla()

        sel_idx = self._selected_row_index()
        Rh     = np.asarray(res.Rh)
        Ve_obs = np.asarray(res.Ve_obs)

        if sel_idx is not None and 0 <= sel_idx < len(Rh):
            mask_rest = np.ones(len(Rh), dtype=bool)
            mask_rest[sel_idx] = False
            if mask_rest.any():
                ax1.scatter(Rh[mask_rest], Ve_obs[mask_rest],
                            color=RED_HL, zorder=5, s=40, marker="x",
                            linewidths=1.5, label="Observed")
            ax1.scatter([Rh[sel_idx]], [Ve_obs[sel_idx]],
                        color=ORANGE, zorder=8, s=120, marker="x",
                        linewidths=2.5, label="Selected")
        else:
            ax1.scatter(Rh, Ve_obs, color=RED_HL, zorder=5, s=40,
                        marker="x", linewidths=1.5, label="Observed")

        # Disabled rows: grey markers (not in fit)
        dis_rh, dis_ve = self._disabled_rh_ve()
        if dis_rh:
            ax1.scatter(dis_rh, dis_ve, color=MUTED, zorder=3, s=40, marker="x",
                        linewidths=1.5, alpha=0.6, label="Disabled")

        ax1.plot(res.rang, res.Ve_curve, color=ACCENT, linewidth=1.2, label="Knox-Scott fit")
        if (self._show_errors_var.get() and
                np.all(np.isfinite(res.ve_ci_lower)) and
                np.all(np.isfinite(res.ve_ci_upper))):
            ax1.fill_between(res.rang, res.ve_ci_lower, res.ve_ci_upper,
                             color=ACCENT, alpha=0.18, label="95% CI")
        ax1.set_xscale("log")
        ax1.set_xlabel("Rh (nm)", fontsize=8)
        ax1.set_ylabel("Ve (ml)", fontsize=8)
        ax1.set_title("Exclusion Curve (Knox-Scott)", fontsize=9)
        ax1.legend(fontsize=7)
        ax1.grid(True, which="both", alpha=0.3)

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()

    def _draw_jerabek(self, res: FitResult):
        ax2 = self.ax_psd
        ax2.cla()

        # -- PSD bar chart --
        mask = res.pore_volumes > 1e-6
        xlabel = "Pore radius (nm)" if res.model == "cylindrical" else "Chain conc. (nm/nm³)"

        x_pos = np.arange(len(res.pore_params))
        # Propagate the pore-volume confidence limits to specific-volume
        # uncertainties. Specific volume is proportional to pore volume.
        spec_lower = np.full(len(res.spec_volumes), np.nan, dtype=float)
        spec_upper = np.full(len(res.spec_volumes), np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.divide(res.spec_volumes, res.pore_volumes,
                              out=np.zeros_like(res.spec_volumes, dtype=float),
                              where=np.abs(res.pore_volumes) > 0)
        valid_ci = np.isfinite(res.ci_lower) & np.isfinite(res.ci_upper)
        spec_lower[valid_ci] = res.ci_lower[valid_ci] * scale[valid_ci]
        spec_upper[valid_ci] = res.ci_upper[valid_ci] * scale[valid_ci]
        if self._show_errors_var.get():
            yerr = np.vstack((
                np.maximum(0.0, res.spec_volumes - spec_lower),
                np.maximum(0.0, spec_upper - res.spec_volumes),
            ))
            yerr[:, ~valid_ci] = 0.0
        else:
            yerr = None
        bars = ax2.bar(x_pos, res.spec_volumes, yerr=yerr,
                       capsize=3 if self._show_errors_var.get() else 0,
                       color=ACCENT, alpha=0.75, width=0.7)
        for bar, m in zip(bars, mask):
            if m:
                bar.set_edgecolor(ACCENT)
                bar.set_linewidth(1)

        labels = [f"{p:.1f}" for p in res.pore_params]
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, fontsize=6, ha="right")
        ax2.set_xlabel(xlabel, fontsize=8)
        ax2.set_ylabel("Spec. pore vol. (ml/g)", fontsize=8)
        ax2.set_title("Pore Size Distribution", fontsize=9)
        ax2.grid(True, axis="y", alpha=0.3)

        # Draw exclusion curve (handles its own tight_layout + canvas.draw)
        self._draw_excl_jerabek(res)

    def _draw_knox_scott(self, res: KnoxScottResult):
        ax2 = self.ax_psd
        ax2.cla()

        ax2.plot(res.rang, res.psd, color=GREEN, linewidth=1.4)
        ax2.set_xscale("log")
        ax2.set_xlabel("Pore size (nm)", fontsize=8)
        ax2.set_ylabel("dV/dr (ml/g·nm)", fontsize=8)
        ax2.set_title("Differential PSD", fontsize=9)
        ax2.grid(True, which="both", alpha=0.3)

        self._draw_excl_knox_scott(res)

    # ------------------------------------------------------------------
    # Result text formatting
    # ------------------------------------------------------------------

    def _format_jerabek_result(self, res: FitResult):
        lines = [
            f"Model: {res.model.title()}  |  Method: Jerabek",
            f"SSR: {res.ssr:.6f}  |  Iterations: {res.n_iter}  |  {res.message}",
            f"Residual volume: {res.residual_volume:.3f} ml",
            f"{'Pore param':>14}  {'Vol (ml)':>10}  {'95% unc.':>12}  {'Spec vol':>10}  {'Spec surf':>12}",
            "-" * 52,
        ]
        for i in range(len(res.pore_params)):
            if True:
                lines.append(
                    f"{res.pore_params[i]:>14.2f}  {res.pore_volumes[i]:>10.4f}"
                    f"  ±{0.5 * (res.ci_upper[i] - res.ci_lower[i]):.4f}" if np.isfinite(res.ci_lower[i]) and np.isfinite(res.ci_upper[i]) else "  n/a"
                    f"  {res.spec_volumes[i]:>10.4f}  {res.spec_surface[i]:>12.4f}"
                )
        lines.append("-" * 52)
        lines.append(f"Total spec. vol: {np.sum(res.spec_volumes):.4f} ml/g")
        lines.append(f"Total spec. surf: {np.sum(res.spec_surface):.4f} m²/g")
        self._set_result_text("\n".join(lines))

    def _format_ks_result(self, res: KnoxScottResult):
        lines = [
            "Method: Knox-Scott",
            f"SSR: {res.ssr:.6f}",
            f"Rational function: nnu={res.nnu}, nde={res.nde}",
            f"Parameters: {', '.join(f'{p:.4f}' for p in res.popt)}",
        ]
        self._set_result_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_figure(self):
        path = filedialog.asksaveasfilename(
            title="Export figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All", "*.*")],
        )
        if path:
            try:
                self._fig.savefig(path, dpi=200, bbox_inches="tight")
                messagebox.showinfo("Export", f"Figure saved to:\n{path}")
            except Exception as exc:
                messagebox.showerror("Export error", str(exc))

    def _export_pdf_report(self):
        if self._fit_result is None and self._ks_result is None:
            messagebox.showinfo("Export", "Run a fit first.")
            return
        path = filedialog.asksaveasfilename(title="Create PDF report", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not path: return
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.table import Table
            with PdfPages(path) as pdf:
                # First page: the two figures
                pdf.savefig(self._fig, bbox_inches="tight")
                # Second page: fit metadata and parameters
                meta_fig = Figure(figsize=(11.7, 8.3), dpi=150)
                ma = meta_fig.add_axes([0.08, 0.08, 0.84, 0.84]); ma.axis("off")
                rr = self._fit_result if self._fit_result is not None else self._ks_result
                method_name = "Jerabek " + rr.model if self._fit_result is not None else "Knox-Scott"
                lines = [f"Method: {method_name}", f"R²: {rr.r2:.8g}", f"SSR: {rr.ssr:.8g}",
                         f"Converged: {rr.converged}", f"Iterations: {rr.n_iter}", f"Data points: {rr.n_data}"]
                if self._fit_result is not None:
                    lines += [f"V₀: {rr.V0:.8g} ml", f"Vc: {rr.Vc:.8g} ml", f"Residual volume: {rr.residual_volume:.8g} ml",
                              f"Fitting parameters: {', '.join(f'{x:.8g}' for x in rr.pore_params)}"]
                else:
                    lines += [f"Numerator degree: {rr.nnu}", f"Denominator degree: {rr.nde}",
                              f"Fitting parameters: {', '.join(f'{x:.8g}' for x in rr.popt)}"]
                ma.text(0, 1, "\n".join(lines), va="top", fontsize=12)
                pdf.savefig(meta_fig, bbox_inches="tight")
                # Third page: a clean tabular report
                fig = Figure(figsize=(11.7, 8.3), dpi=150)
                ax = fig.add_axes([0.05, 0.12, 0.9, 0.78]); ax.axis("off")
                rows = []
                if self._fit_result is not None:
                    r = self._fit_result
                    headers = ["Pore parameter", "Volume (ml)", "95% uncertainty (ml)", "Specific volume (ml/g)", "Specific surface (m²/g)"]
                    rows = [[f"{p:.4f}", f"{v:.4f}", (f"±{0.5 * (hi - lo):.4f}" if np.isfinite(lo) and np.isfinite(hi) else "n/a"), f"{sv:.4f}", f"{ss:.4f}"] for p,v,lo,hi,sv,ss in zip(r.pore_params,r.pore_volumes,r.ci_lower,r.ci_upper,r.spec_volumes,r.spec_surface) ]
                    title = "ISEC Analyzer — Jerabek pore distribution"
                else:
                    r = self._ks_result; headers=["Pore size", "", "PSD", ""]; rows=[[f"{x:.5g}","",f"{y:.5g}",""] for x,y in zip(r.rang,r.psd)]; title="ISEC Analyzer — Knox-Scott distribution"
                ax.set_title(title, fontsize=16, pad=20)
                table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="right")
                table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
                pdf.savefig(fig, bbox_inches="tight")
            messagebox.showinfo("Export", f"PDF report saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("PDF export error", str(exc))

    def _export_results(self):
        import pandas as pd, os

        path = filedialog.asksaveasfilename(
            title="Export results",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All", "*.*")],
        )
        if not path:
            return

        rows = []
        if self._fit_result:
            res = self._fit_result
            for i in range(len(res.pore_params)):
                rows.append({
                    "pore_param": res.pore_params[i],
                    "pore_volume_ml": res.pore_volumes[i],
                    "spec_volume_ml_g": res.spec_volumes[i],
                    "spec_surface_m2_g": res.spec_surface[i],
                    "ci95_lower_ml": res.ci_lower[i],
                    "ci95_upper_ml": res.ci_upper[i],
                })
        elif self._ks_result:
            res = self._ks_result
            for pp, psd_v in zip(res.rang, res.psd):
                rows.append({"Rh_nm": pp, "psd": psd_v})

        if rows:
            df = pd.DataFrame(rows)
            ext = os.path.splitext(path)[1].lower()
            if ext == ".xlsx":
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False)
            messagebox.showinfo("Export", f"Results saved to:\n{path}")
        else:
            messagebox.showinfo("Export", "No fit results to export. Run a fit first.")


# ---------------------------------------------------------------------------
# Manual row entry dialog
# ---------------------------------------------------------------------------

class _ManualRowDialog(tk.Toplevel):
    def __init__(self, parent, title="Add Standard",
                 initial_rh: float = 1.0, initial_ve: float = 5.0):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        self._build(initial_rh, initial_ve)
        self.wait_window()

    def _build(self, init_rh, init_ve):
        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Standard type:").grid(row=0, column=0, sticky="w")
        self._key_var = tk.StringVar(value="ps")
        key_cb = ttk.Combobox(frm, textvariable=self._key_var,
                               values=STANDARD_KEYS, state="readonly", width=18)
        key_cb.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        key_cb.bind("<<ComboboxSelected>>", self._on_key_changed)

        ttk.Label(frm, text="MW / nC / size:").grid(row=1, column=0, sticky="w")
        self._val_var = tk.StringVar(value="")
        self._val_entry = ttk.Entry(frm, textvariable=self._val_var, width=12)
        self._val_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(frm, text="Or enter Rh (nm) directly:").grid(row=2, column=0, sticky="w")
        self._rh_var = tk.StringVar(value=str(init_rh))
        self._rh_entry = ttk.Entry(frm, textvariable=self._rh_var, width=10)
        self._rh_entry.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(frm, text="Ve (ml):").grid(row=3, column=0, sticky="w")
        self._ve_var = tk.StringVar(value=str(init_ve))
        ttk.Entry(frm, textvariable=self._ve_var, width=10).grid(
            row=3, column=1, sticky="w", padx=4, pady=2)

        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=4, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(btn_frm, text="OK", command=self._ok, width=8).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frm, text="Cancel", command=self.destroy, width=8).grid(row=0, column=1, padx=4)

    def _on_key_changed(self, event=None):
        key = self._key_var.get()
        has_mw = standard_needs_mw(key)
        self._val_entry.configure(state="normal" if has_mw else "disabled")

    def _ok(self):
        key = self._key_var.get()
        val_str = self._val_var.get().strip()
        rh_str  = self._rh_var.get().strip()
        ve_str  = self._ve_var.get().strip()

        try:
            ve = float(ve_str)
        except ValueError:
            messagebox.showerror("Error", "Ve must be a number.", parent=self)
            return

        std_val = None
        if val_str:
            try:
                std_val = float(val_str)
            except ValueError:
                messagebox.showerror("Error", "MW/nC must be a number.", parent=self)
                return

        try:
            if std_val is not None and standard_needs_mw(key):
                rh = compute_rh(key, std_val)
            elif rh_str:
                rh = float(rh_str)
            elif not standard_needs_mw(key):
                rh = compute_rh(key, 0.0)
            else:
                messagebox.showerror("Error", "Please enter MW/nC or Rh.", parent=self)
                return
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return

        self.result = (key, std_val, rh, ve)
        self.destroy()
