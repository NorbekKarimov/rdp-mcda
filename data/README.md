# Input data

The stacks are not stored in this repository because of their size (up to 2.4 GB). They are archived on Zenodo: [Zenodo DOI].

| Stack | Cells K | Valid cells m | File size |
|---|---|---|---|
| `orol_v2_1000m` | 69,048 | 41,991 | 2.2 MB |
| `orol_v2_100m` | 6,843,202 | 4,149,251 | 219 MB |
| `orol_v2_30m` | 75,969,060 | 46,060,601 | 2.43 GB |

Each folder contains `data.npy`, a band-major float32 array of shape (8, K) with missing values coded as −9999, and `meta.json` (grid size, transform, CRS EPSG:32640, nodata value, source file).

Study area: Muynak District, Republic of Karakalpakstan, Uzbekistan (FAO GAUL Level 2), about 41,300 km².

Layers, in order, all for 1 May – 30 September 2025, prepared in Google Earth Engine (script in `gee/`):

| # | Criterion | Direction | Source product |
|---|---|---|---|
| 1 | Precipitation | + | CHIRPS Daily, seasonal sum |
| 2 | Land surface temperature | − | MODIS MOD11A2 |
| 3 | Wind speed | − | ERA5-Land, 10 m u and v components |
| 4 | NDWI | + | Sentinel-2 SR Harmonized, median composite |
| 5 | SI2 | − | Sentinel-2 SR Harmonized, median composite |
| 6 | NDVI | + | Sentinel-2 SR Harmonized, median composite |
| 7 | Elevation | − | SRTM |
| 8 | Slope | − | derived from SRTM |

Layers were exported without an explicit resampling method, so the Earth Engine defaults apply (nearest neighbour to finer grids, mean pyramids to coarser grids).

The derived stacks are redistributed under the terms of the source products; users must also cite the source datasets.
