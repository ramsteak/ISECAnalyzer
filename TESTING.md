# Testing Strategy

## Unit tests

Unit tests should cover pure functions first:

- Filename/sample-size parsing.
- Column classification.
- Axis normalization.
- Unit conversion.
- Baseline correction.
- Smoothing.
- Momentum/elution calculation.
- Pore-model equations and fitting validation.
- Project serialization/deserialization.

## Integration tests

The most valuable workflow test is:

1. Create or load a small synthetic dataset.
2. Set global solvent, density, flow rate, and time unit.
3. Select a signal column.
4. Run Tab 1 analysis.
5. Transfer elution data to Tab 2.
6. Configure and run a pore-model fit.
7. Save a project.
8. Load it into a fresh application state.
9. Assert that data, settings, results, and output metadata are preserved.

## GUI tests

Tkinter tests should avoid relying on screen coordinates. Prefer invoking controller methods and inspecting state. Where possible, separate widget construction from callbacks so callbacks can be tested with mocked services.

## Regression cases

Maintain regression tests for:

- A filename containing multiple dots.
- A filename containing spaces.
- A filename containing unrelated numbers such as dates or run IDs.
- A legacy file with time selected as both X and Y.
- Multiple files sharing one solvent, density, and flow rate.
- A large Jerabek output table.
- A project reopened after the original source files were moved or deleted.
- A PDF report with enough rows to require multiple pages.
