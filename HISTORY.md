# History

<!-- ## unreleased -->
<!-- ### Breaking changes -->
 
<!-- ### New Features -->

<!-- ### Bug fixes -->

<!-- ### Documentation -->

<!-- ### Internal Changes -->


## 2026.03.0

### New Features
-   `sensor_drift`: optional iterative subtraction of large-drift outliers.
    Set `iterate_subtract=True` (plus `amplitude_threshold_mK` and/or
    `manual_outlier_sns`) in `drift_parameters` to fit one extra pass
    after the standard pipeline: flagged sensors' pass-1 drift is
    subtracted from `offsets`, and the neighbour-stack stages are
    re-run so the neighbours' fits aren't contaminated by the outlier's
    drift leakage. Pass-1 state is preserved as `*_pass1` attributes
    for diagnostics; new `plot_iteration_diagnostic` emits before/after
    figures for flagged sensors. Output `drift` DataArray gains
    `iteration_count` and `flagged_outlier_sns` attrs for provenance.


## 2023.12.0
-   Created package.
