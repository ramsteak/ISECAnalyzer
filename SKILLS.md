# Developer and LLM Skills Guide

This file is a practical map for anyone modifying ISEC Analyzer without reading every source file first.

## First read

Read these files in order:

1. `README.md` for scope and entry point.
2. `ARCHITECTURE.md` for module responsibilities and state ownership.
3. `state.py` for invariants and global settings.
4. `pipeline.py` for the canonical signal-processing flow.
5. `session.py` for project persistence.
6. The relevant GUI module only after understanding the backend.

## Important concepts

### FileData
`FileData` is the legacy central representation of one imported data file. It contains source data, column metadata, selected axes, processing options, and derived values. Treat it as a compatibility boundary, not the ideal long-term architecture.

### Global settings
Density and flow rate describe the experiment/solvent context and must not silently diverge between files. New code should read them from `GlobalSettings` or an equivalent experiment object.

### AnalysisResult
`AnalysisResult` is the structured result returned by `run_pipeline()`. Prefer it over GUI dictionaries. It contains raw and processed arrays, baseline, smoothing, fit information, elution coordinate, and threshold.

### Tab 2 output
Tab 2 produces model-dependent pore-distribution output. The UI must keep the result table resizable and scrollable. PDF export should use the current plotted figures and the current output table data, not scrape rendered widget text.

## Safe modification workflow

1. Reproduce the issue with a minimal dataset.
2. Identify whether it is parsing, state, numerical processing, GUI, or persistence.
3. Fix the lowest appropriate layer.
4. Add or update a unit test for backend behavior.
5. Run the full test suite.
6. Manually check the GUI workflow if Tkinter or Matplotlib is involved.
7. Verify that saving and reopening a project preserves the change.

## Testing commands

From the project directory:

```bash
python -m pytest
```

For syntax checking:

```bash
python -m compileall .
```

## Rules for new code

- Do not put scientific calculations in Tkinter callbacks.
- Do not use widget values directly inside fitting functions.
- Prefer NumPy arrays and explicit dataclass configuration.
- Validate dimensions, finite values, and parameter ranges at public backend boundaries.
- Keep file parsing tolerant of spaces, dots, dates, run numbers, and unrelated numeric tokens.
- Never assume a filename consists only of the sample identifier.
- Preserve backward compatibility when loading older projects.
- Avoid silently changing units; make the active time unit explicit.
- Use deterministic fallback behavior for invalid axis selections.
- Keep exports reproducible from project state.

## Common change locations

| Task | Primary file(s) |
|---|---|
| Add a source-file format | `data_io.py`, tests |
| Change filename/sample-size parsing | `data_io.py`, tests |
| Change baseline/smoothing/momentum | `analysis.py`, `pipeline.py`, tests |
| Change global density/flow behavior | `state.py`, `session.py`, GUI bindings |
| Add a pore model | `fitting.py`, `gui_tab2.py`, tests |
| Change output table columns | `gui_tab2.py`, report/export code |
| Change PDF report layout | `gui_tab2.py` or a dedicated report module |
| Change project file format | `session.py`, format documentation, round-trip tests |
| Add a menu or shortcut | `app.py` |

## Things to watch for

- The project has historically mixed GUI state, file state, and derived data.
- A GUI variable changing does not necessarily mean the backend state changed.
- A derived array may be stale after changing an axis or processing option; invalidate or recompute it.
- Fitting can fail for physically or numerically unsuitable inputs. Report errors without destroying valid preceding results.
- Large output tables need scrollbars, adjustable columns, and PDF pagination.
