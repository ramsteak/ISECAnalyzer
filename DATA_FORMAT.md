# Data and Project Format

## Source data

Input files are parsed into a `FileData` object. A file generally contains one coordinate column (time, volume, or weight) and one or more signal columns. Column metadata identifies the semantic type used by axis validation and the GUI selectors.

The parser should tolerate filenames and headers containing spaces, dots, dates, run numbers, and additional descriptive text.

## Units

- Time defaults to seconds (`s`).
- Flow rate is expressed in mL/min.
- Density is expressed in g/mL.
- Elution volume is expressed in mL unless the UI explicitly indicates another unit.

Unit conversion must be centralized and should use global experiment settings where appropriate.

## `.isecproj` projects

A project is a portable archive containing a versioned manifest and embedded data. The original source files should not be required to reopen a project.

Conceptually, the manifest contains:

```text
format/version
project metadata
global experiment settings
file list
  raw data
  filename and metadata
  axis selections
  processing configuration
  derived analysis results (optional/cache)
Tab 2 fitting configuration
output registry (optional/cache)
```

Raw data and configuration are authoritative. Derived arrays and exported outputs are caches or reproducibility artifacts and must not override raw data/configuration when reopening.

## Versioning rules

When changing the project schema:

1. Increment the format version when compatibility changes.
2. Keep migrations explicit.
3. Reject unknown future versions with a useful message.
4. Add a save/load round-trip test.
5. Document newly added fields and defaults.
