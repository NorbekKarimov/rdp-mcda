# Google Earth Engine script

Place here the Earth Engine script (`Orol_v2.js`) that exported `Orol_stack_v2_<res>.tif`. It is a copy of the earlier processing script with two changes: pixels without any cloud-free Sentinel-2 observation are filled with the unmasked seasonal median, and slope is computed on the unclipped DEM.
