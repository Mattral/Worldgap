PGM Length vs Force datasets — digitized from Figure 4(a)
"Relationship of air pressure, force and elongation ratio"

Method: pixel-color tracing (no WebPlotDigitizer). Axis pixel positions were
calibrated from the tick-label text (x: 0-50 N, y: 300-550 mm), each curve's
line color was matched against its legend swatch, connected-component
analysis isolated the correct curve trace at each color from any
cross-contamination where curves overlap, then values were smoothed
(rolling median) and resampled onto a 0.5 N grid.

Files (one per pressure level):
  PGM_length_0MPa.csv
  PGM_length_0p05MPa.csv
  PGM_length_0p1MPa.csv
  PGM_length_0p15MPa.csv
  PGM_length_0p2MPa.csv
  PGM_length_0p25MPa.csv
  PGM_length_0p3MPa.csv

Each file has two columns: Force_N, PGM_Length_mm

Notes / limitations:
- In regions where multiple curves run very close together or cross
  (e.g. 0.1-0.3MPa curves between ~Force 15-40N), small local wiggles may
  not perfectly reproduce the original due to anti-aliased color blending
  between adjacent lines. Overall shape, ordering, and endpoints match the
  published figure closely (verified visually against a re-plot of these
  files).
- Values are estimates reconstructed from the published chart, not the
  original experimental measurements.
