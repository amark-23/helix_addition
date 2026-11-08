# Conventions

## Save results that cost a run

Any run whose output may be needed later is saved to disk, together with the
parameters that produced it. Before running, the code checks whether a saved
result with the same parameters exists and loads it instead.

- Where: `results/<step_name>/`
- What: the data file, plus a small JSON file with every parameter that
  affects the result (template, seed, n, model id, dtype, library versions).
- Same parameters, same file: load. Any parameter different: new run, new file.
