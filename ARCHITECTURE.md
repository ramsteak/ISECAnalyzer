# Architecture

## Design goals

The application should have one numerical implementation, explicit state ownership, reproducible project files, and a GUI that primarily coordinates user interaction rather than implementing scientific calculations itself.

## High-level structure

```text
isec_analyzer.py
       |
       v
     app.py
       |
       +--------------------+
       |                    |
       v                    v
   gui_tab1.py          gui_tab2.py
       |                    |
       v                    v
 data_io.py             fitting.py
       |                    ^
       v                    |
    FileData          analysis.py
       |                    ^
       +------> pipeline.py+
       |
       v
   session.py <---- state.py
```

## Responsibilities

### `app.py`
Owns the root Tk window, menus, notebook tabs, status bar, and cross-tab notification. It should not contain scientific calculations.

### `gui_tab1.py`
Owns the user interface for imported chromatograms and standard/elution-volume analysis. It currently contains legacy UI-facing state and compatibility logic. New numerical behavior should be routed through `pipeline.py`.

### `gui_tab2.py`
Owns the pore-model fitting interface, plots, result table, figure saving, and PDF report creation. Fitting algorithms belong in `fitting.py`; this module should eventually become a thin view/controller layer.

### `data_io.py`
Defines how source files become `FileData` objects. It owns parsing, column metadata, data types, and source-file compatibility. It should not own global experiment settings.

### `state.py`
Defines application-level configuration and invariants:

- `GlobalSettings`: solvent, density, flow rate, time unit, and weight correction.
- `AnalysisSettings`: processing options.
- `AppState`: global settings, files, and selection.
- `normalize_analysis_axes()`: repairs invalid legacy axis choices.

A Y axis must represent a signal whenever a signal column is available. X should be a coordinate-like column such as time, volume, or weight.

### `pipeline.py`
The canonical analysis orchestration layer. `run_pipeline(x, y, config)` is pure with respect to the GUI: it accepts arrays and explicit configuration and returns an `AnalysisResult`. `run_file_pipeline(fd)` is a compatibility adapter for the current `FileData` model.

### `analysis.py`
Contains lower-level signal-processing operations such as baseline subtraction, smoothing, peak fitting, and momentum calculations. It should remain independent of Tkinter.

### `fitting.py`
Contains pore-model equations, model fitting, and distribution calculations. It should receive numerical arrays and parameters, not widgets.

### `session.py`
Serializes and restores a complete project. Project persistence should include raw data, metadata, global settings, per-file analysis configuration, Tab 2 fitting settings, derived results where useful, and output metadata.

## Data flow

```text
Source file
   -> data_io parser
   -> FileData
   -> axis validation
   -> PipelineConfig
   -> run_pipeline()
   -> AnalysisResult
   -> GUI plots/tables/export
```

The GUI should never duplicate baseline, smoothing, fitting, or momentum logic. If a calculation is needed in more than one place, move it to a pure backend function.

## State ownership rules

1. Experiment-wide solvent, density, and flow rate are global.
2. Raw imported data is immutable in concept; transformations produce derived arrays.
3. Widget values are not the authoritative scientific state. They are an editable view of state.
4. Analysis results are derived and can be regenerated from raw data plus configuration.
5. Explicitly exported files are outputs, not the source of truth for the analysis.
6. Project loading must validate its format version and reject malformed or incompatible data clearly.

## Current compatibility debt

The application still has legacy fields attached to `FileData` and GUI dictionaries. These should be migrated gradually:

1. Introduce typed data classes for file metadata and results.
2. Replace ad-hoc dictionaries with `AnalysisResult` and serializable DTOs.
3. Make Tab 1 and Tab 2 controllers consume state objects rather than reading widgets directly.
4. Add integration tests for GUI actions and project round trips.
5. Remove compatibility fields only after all consumers are migrated.
