"""Portable project/session persistence for ISEC Analyzer.

A project is a ZIP containing a JSON manifest and one CSV per imported run.
It stores raw data, column metadata, analysis settings, computed outputs, and
Tab 2 fitting controls. No original data files are required to reopen it.
"""
from __future__ import annotations
import json, zipfile
import tkinter as tk
from pathlib import Path
import pandas as pd
from .data_io import FileData, ColumnInfo
from .state import GlobalSettings

FORMAT = "isec-project"
VERSION = 1

def _fd_to_dict(fd):
    attrs = ["separator","decimal","has_header","plot_x_col","plot_y_col",
             "analysis_x_col","analysis_y_col","x_reset","invert_y","x_min","x_max",
             "baseline_enabled","baseline_x_col","baseline_x_min","baseline_x_max",
             "baseline_degree","smoothing_enabled","smoothing_window","fit_enabled",
             "fit_model","momentum_degree","momentum_threshold","weight_correct",
             "std_key","std_value","rh","elution_volume"]
    d = {a: getattr(fd, a) for a in attrs}
    d["col_info"] = [ci.__dict__ for ci in fd.col_info]
    d["filename"] = fd.filename
    d["path"] = fd.path
    return d

def save_project(path, tab1, tab2=None):
    settings = tab1.state.global_settings
    manifest = {"format": FORMAT, "version": VERSION,
                "global_settings": settings.__dict__,
                "selected_indices": list(tab1._selected_indices()),
                "files": []}
    if tab2 is not None:
        for name in ("_v0_var","_vc_var","_weight_var","_vz_var","_fit_kind_var","_chain_diam_var","_pore_start","_pore_end","_pore_step","_pore_text","_chain_conc_text","_ks_nnu","_ks_nde"):
            var = getattr(tab2, name, None)
            if var is not None:
                manifest.setdefault("tab2", {})[name] = var.get("1.0", "end-1c") if isinstance(var, tk.Text) else var.get()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, fd in enumerate(tab1.file_data):
            name = f"data/{i:04d}.csv"
            z.writestr(name, fd.df_raw.to_csv(index=False))
            meta = _fd_to_dict(fd)
            meta["data_file"] = name
            manifest["files"].append(meta)
        z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

def _restore_fd(meta, csv_text):
    import io
    fd = FileData.__new__(FileData)
    fd.path = meta.get("path", "")
    fd.filename = meta.get("filename", "untitled")
    fd.df_raw = pd.read_csv(io.StringIO(csv_text))
    fd.col_info = [ColumnInfo(**x) for x in meta.get("col_info", [])]
    # Ensure metadata remains consistent if a legacy project omitted it.
    if len(fd.col_info) != fd.df_raw.shape[1]:
        fd.col_info = [ColumnInfo(str(c)) for c in fd.df_raw.columns]
    for key, value in meta.items():
        if key not in {"col_info","data_file","filename","path"}:
            setattr(fd, key, value)
    fd.density = 1.0
    fd.flow_rate = 1.0
    fd._normalize_axes()
    return fd

def load_project(path, tab1, tab2=None):
    with zipfile.ZipFile(path, "r") as z:
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError("Not an ISEC Analyzer project file")
        gs = GlobalSettings(**manifest.get("global_settings", {}))
        gs.validate()
        files = [_restore_fd(meta, z.read(meta["data_file"]).decode("utf-8"))
                 for meta in manifest.get("files", [])]
    tab1.state.global_settings = gs
    tab1.file_data = files
    tab1.state.files = files
    tab1.file_lb.delete(0, "end")
    for fd in files:
        tab1.file_lb.insert("end", fd.get_display_name())
    for i in manifest.get("selected_indices", []):
        if 0 <= i < len(files): tab1.file_lb.selection_set(i)
    if files and not tab1.file_lb.curselection(): tab1.file_lb.selection_set(0)
    if tab2 is not None:
        saved_tab2 = manifest.get("tab2", {})
        for name, value in saved_tab2.items():
            var = getattr(tab2, name, None)
            if var is not None:
                if isinstance(var, tk.Text):
                    var.delete("1.0", "end")
                    var.insert("1.0", value)
                else:
                    var.set(value)
        # Projects created by older versions may have stored the internal
        # method identifier. Normalize it to the current combobox label.
        if hasattr(tab2, "_fit_kind_var"):
            method = tab2._fit_kind_var.get()
            aliases = {"jerabek": "Jerabek cylindrical", "cylindrical": "Jerabek cylindrical",
                       "ogston": "Jerabek Ogston", "knox_scott": "Knox-scott", "knox-scott": "Knox-scott"}
            tab2._fit_kind_var.set(aliases.get(method.lower(), method))
    tab1._on_selection_changed()
    tab1._redraw()
    tab1.app.notify_data_changed()
