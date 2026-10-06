# ISEC Analyzer

ISEC Analyzer is a desktop Python/Tkinter application for processing Inverse Size Exclusion Chromatography (ISEC) data. It loads chromatographic data, converts retention time to elution volume or related coordinates, associates standards with hydrodynamic sizes, and fits pore-size distributions using ISEC models such as the Jerabek method.

## Main capabilities

- Import tabular chromatographic data.
- Select time/volume/weight coordinates and signal columns.
- Apply baseline correction, smoothing, peak fitting, and momentum calculations.
- Calculate elution volumes for standards.
- Fit pore-size distributions using Tab 2 models.
- Display exclusion curves and pore-size distributions.
- Save and reopen complete `.isecproj` projects.
- Export figures and generate PDF reports containing figures and output tables.

## Running

From the `isec` directory, run:

```bash
python isec_analyzer.py
```

The project currently uses a flat application-module layout and imports modules such as `gui_tab1`, `gui_tab2`, `data_io`, and `session` as top-level modules when launched from the application directory.

## Application layout

- `isec_analyzer.py`: executable entry point.
- `app.py`: root window, menus, notebook, status bar, and cross-tab coordination.
- `gui_tab1.py`: data import, file management, chromatographic analysis, and standard/elution-volume workflow.
- `gui_tab2.py`: pore-model fitting, plots, output table, figure export, and PDF report generation.
- `data_io.py`: file parsing, column metadata, data representation, and filename/sample-size parsing.
- `standards.py`: standard compounds and hydrodynamic-size information.
- `analysis.py`: numerical operations used by the pipeline.
- `pipeline.py`: canonical, GUI-independent analysis orchestration.
- `fitting.py`: pore-model and distribution fitting algorithms.
- `state.py`: global settings, analysis settings, and axis invariants.
- `session.py`: project persistence and restoration.

See `ARCHITECTURE.md` for the detailed design and `SKILLS.md` for instructions intended for developers and LLM-based coding agents.

## Running with uv

The source directory is a Python package. From the directory containing `isec/`:

```powershell
uv venv
uv pip install -r isec/requirements.txt
py -m isec
```

Do not `cd` into `isec/` before running `py -m isec`. If the package is imported as a package, all internal relative imports work. The application must not be launched by executing an individual module such as `py isec/gui_tab1.py`.
