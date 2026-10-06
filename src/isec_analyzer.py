#!/usr/bin/env python3
"""
ISEC Analyzer
=============
A GUI application for Inverse Size Exclusion Chromatography (ISEC) data analysis.

Modules:
  - standards.py      : Hydrodynamic radius calculations for all standard types
  - data_io.py        : File import, format detection, column detection
  - analysis.py       : Baseline correction, smoothing, peak fitting, momentum
  - fitting.py        : ISEC pore-model fitting (Jerabek cylindrical/Ogston, Knox-Scott)
  - gui_tab1.py       : Tab 1 – Elution volume analysis
  - gui_tab2.py       : Tab 2 – Elution volume fitting
  - app.py            : Main application window

Run:
  python isec_analyzer.py
"""

from .app import ISECApp
import tkinter as tk


def main():
    root = tk.Tk()
    root.title("ISEC Analyzer")
    root.minsize(1100, 700)
    # Set a reasonable default size
    root.geometry("1400x850")
    app = ISECApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
