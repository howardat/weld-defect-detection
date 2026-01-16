**VISION**
User sends image and gets a report based on ISO standards. Particularly: porosity size, discontinuities, cracks.

1. Discontinuity detection

- Crop image
- Detect weld image again
- Mask refining
- Skeletonize
- Fitting a linear function
- Comparison and inference
- Add to JSON
- Visualize pipeline

2. Porosity detection

- Crop image
- Detect pores
- Determine size of pores
- Check against the standards
- Add to JSON
- Visualize pipeline

3. Cracks detection

- Crop image
- Detect crack image
- Add to JSON
- Visualize pipeline

4. VLM inference / experiments

- Complete pipeline. One input - one report

One issue - one feature.
