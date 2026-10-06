"""
app.py
======
Main ISECApp window: notebook with Tab 1 and Tab 2, styling, menus and
cross-tab communication.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
ACCENT   = "#1565C0"
BG       = "#F5F5F5"
PANEL    = "#FFFFFF"
BORDER   = "#E0E0E0"
TEXT     = "#212121"
MUTED    = "#757575"
NOTEBOOK_ACTIVE   = "#FFFFFF"
NOTEBOOK_INACTIVE = "#E3EAF6"


class ISECApp:
    """Root application wrapper."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self._setup_styles()
        self._build_menu()
        self._build_ui()

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _setup_styles(self):
        style = ttk.Style(self.root)
        self.root.configure(bg=BG)

        # Try modern theme
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".",
                        background=BG,
                        foreground=TEXT,
                        font=("Segoe UI", 9))
        style.configure("TFrame",       background=BG)
        style.configure("Tab.TFrame",   background=BG)
        style.configure("TLabel",       background=BG, foreground=TEXT)
        style.configure("TLabelframe",  background=BG, foreground=TEXT,
                        bordercolor=BORDER, relief="groove")
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                        font=("Segoe UI", 9, "bold"))
        style.configure("TButton",
                        background=ACCENT, foreground="white",
                        font=("Segoe UI", 8), padding=(6, 3),
                        relief="flat")
        style.map("TButton",
                  background=[("active", "#1976D2"), ("pressed", "#0D47A1")],
                  foreground=[("active", "white")])
        style.configure("TEntry",
                        fieldbackground=PANEL, foreground=TEXT,
                        bordercolor=BORDER, lightcolor=BORDER,
                        relief="flat")
        style.configure("TCombobox",
                        fieldbackground=PANEL, foreground=TEXT,
                        selectbackground=ACCENT, selectforeground="white")
        style.configure("TCheckbutton",
                        background=BG, foreground=TEXT)
        style.configure("TSeparator",   background=BORDER)

        # Notebook
        style.configure("TNotebook",
                        background=BG,
                        bordercolor=BORDER,
                        tabmargins=[2, 4, 0, 0])
        style.configure("TNotebook.Tab",
                        background=NOTEBOOK_INACTIVE,
                        foreground=MUTED,
                        padding=[14, 6],
                        font=("Segoe UI", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", NOTEBOOK_ACTIVE)],
                  foreground=[("selected", ACCENT)],
                  font=[("selected", ("Segoe UI", 9, "bold"))])

        # Treeview
        style.configure("Treeview",
                        background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL,
                        rowheight=22,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading",
                        background=BG, foreground=ACCENT,
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self.root, bg=PANEL, fg=TEXT, tearoff=False)
        self.root.configure(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False, bg=PANEL, fg=TEXT)
        file_menu.add_command(label="Open project…", command=self._open_project, accelerator="Ctrl+O")
        file_menu.add_command(label="Save project…", command=self._save_project, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit, accelerator="Ctrl+Q")
        self.root.bind_all("<Control-q>", lambda e: self.root.quit())
        self.root.bind_all("<Control-s>", lambda e: self._save_project())
        self.root.bind_all("<Control-o>", lambda e: self._open_project())
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False, bg=PANEL, fg=TEXT)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Help", command=self._show_help)
        menubar.add_cascade(label="Help", menu=help_menu)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Header bar
        header = tk.Frame(self.root, bg=ACCENT, height=46)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="ISEC Analyzer",
                 bg=ACCENT, fg="white",
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(header, text="Inverse Size Exclusion Chromatography",
                 bg=ACCENT, fg="#90CAF9",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=0)

        # Notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 4))
        self.notebook = nb

        from .gui_tab1 import Tab1
        from .gui_tab2 import Tab2

        self.tab1 = Tab1(nb, self)
        self.tab2 = Tab2(nb, self)

        nb.add(self.tab1, text="  1 · Elution Volume Analysis  ")
        nb.add(self.tab2, text="  2 · Pore Model Fitting  ")

        # Status bar
        status_bar = tk.Frame(self.root, bg=BORDER, height=22)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(status_bar, textvariable=self._status_var,
                 bg=BORDER, fg=MUTED, font=("Segoe UI", 8),
                 anchor="w").pack(side=tk.LEFT, padx=6)

    def _save_project(self):
        path = filedialog.asksaveasfilename(parent=self.root, title="Save ISEC project",
            defaultextension=".isecproj", filetypes=[("ISEC project", "*.isecproj"), ("All files", "*.*")])
        if not path: return
        try:
            from .session import save_project
            save_project(path, self.tab1, self.tab2)
            self._status_var.set(f"Project saved: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save project", str(exc), parent=self.root)

    def _open_project(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open ISEC project",
            filetypes=[("ISEC project", "*.isecproj"), ("All files", "*.*")])
        if not path: return
        try:
            from .session import load_project
            load_project(path, self.tab1, self.tab2)
            self._status_var.set(f"Project opened: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Open project", str(exc), parent=self.root)

    # ------------------------------------------------------------------
    # Cross-tab communication
    # ------------------------------------------------------------------

    def notify_data_changed(self):
        """Called by Tab 1 when elution data changes; updates Tab 2."""
        try:
            elution_data = self.tab1.get_elution_data()
            self.tab2.update_from_tab1(elution_data)
            n = len(elution_data)
            self._status_var.set(
                f"{n} standard{'s' if n != 1 else ''} with computed elution volume"
                + ("s" if n != 1 else ""))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _show_about(self):
        messagebox.showinfo(
            "About ISEC Analyzer",
            "ISEC Analyzer\n\n"
            "A GUI tool for Inverse Size Exclusion Chromatography (ISEC) data analysis.\n\n"
            "Based on ISECrunch by M. Bertocco, P. Centomo and L. Ulliana.\n\n"
            "Modules: data_io, standards, analysis, fitting, gui_tab1, gui_tab2",
        )

    def _show_help(self):
        win = tk.Toplevel(self.root)
        win.title("ISEC Analyzer – Help")
        win.geometry("680x520")
        win.resizable(True, True)

        text = tk.Text(win, wrap="word", font=("Segoe UI", 9),
                       bg=PANEL, fg=TEXT, padx=12, pady=8, relief="flat")
        sb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        help_text = """
ISEC ANALYZER – USER GUIDE
===========================

TAB 1: ELUTION VOLUME ANALYSIS
────────────────────────────────
PURPOSE
  Analyse individual chromatographic runs to calculate the elution volume
  for each size standard.

WORKFLOW
  1. Import files using the "+ Add files" button in the left sidebar.
     Multiple files can be imported at once. Files are auto-detected for:
       • Standard type (PS, Alkane, Dextran, PMMA, PEG, D2O, sugars…)
       • File format (separator, decimal, header)
       • X and Y columns

  2. Select a file in the left list to view its chromatogram.

  3. Configure per-file settings in the right sidebar:
       • Standard type & size → Rh is computed automatically
       • X range cut to exclude irrelevant parts of the run
       • Baseline correction (polynomial fit on regions outside the peak)
       • Signal smoothing (Savitzky-Golay)
       • Peak fitting (Gaussian, Double-Gaussian, EMG)
       • Momentum parameters (degree and threshold %)

  4. The plot shows the baseline, threshold line, and vertical line
     at the computed elution volume.

  5. Use "Apply to all" to propagate any setting to all loaded files.

GLOBAL SETTINGS (bottom of right sidebar)
  • File format: separator, decimal, header presence
  • Column selection: x column (time/weight/volume) and y column (signal)
  • Unit conversion: density [g/ml] and flow rate [ml/min]
  • Weight-correct time: fits a linear regression to the weight trace to
    determine the actual flow rate (useful when the balance is inaccurate
    in the short term but linear overall).

EXPORT
  Use "Export elution volumes…" to save all computed Ve and Rh values
  to CSV or Excel.
  Use "Export analyte data…" to save filename, analyte type, size,
  hydrodynamic radius, and elution volume with explicit column names.

STANDARD FORMULAS
  PS:              Rh = 0.0246 × MW^0.588
  Alkane (CnH):    Rh = 0.1 × (n×0.45 + 3.32)  [nm]
  Dextran:         Rh = 0.0732 × MW^0.461
  PMMA:            Rh = 0.02346 × MW^0.568
  PEG:             Rh = 0.02097 × MW^0.587
  D2O, sugars:     Fixed Rh values from literature

TAB 2: PORE MODEL FITTING
───────────────────────────
PURPOSE
  Fit the ISEC exclusion curve (Ve vs Rh) to a pore model and extract
  the pore size distribution (PSD).

DATA
  Standards computed in Tab 1 are automatically transferred here.
  You can also add rows manually or edit existing rows.
  Set V₀ (dead volume), Vc (column volume) and sample weight.

METHODS
  Jerabek (cylindrical pore model)
    Solves:  Ve_i = V₀ + Σ_j K_ij × v_j
    where K_ij = (1 − Rh_i / r_j)²  for cylindrical pores.
    Pore volumes v_j ≥ 0 are optimised by non-negative least squares.
    The pore size list (radii in nm) is user-defined.

  Jerabek (Ogston / fibre network model)
    K_ij = exp(−π/4 × c_j × (Rh_i + r_chain)²)

  Knox-Scott
    Fits a rational function R(Rh) = P(Rh)/Q(Rh) to the exclusion curve.
    The differential PSD is computed analytically from the derivative.

EXPORT
  "Export figure" → PNG, PDF or SVG
  "Export results" → CSV or Excel with pore volumes, specific volumes
                     and specific surface areas.

TIPS
  • For the Jerabek method, use more pore sizes than data points for
    a good resolution (over-determined system). The two-pass NNLS
    fitting automatically removes pore fractions with near-zero volume.
  • Start with the default pore size list and adjust if needed.
  • Knox-Scott is useful for a continuous PSD without assuming discrete
    pore sizes. Use nnu=1, nde=3 as a starting point.
"""
        text.insert("1.0", help_text.strip())
        text.configure(state="disabled")
