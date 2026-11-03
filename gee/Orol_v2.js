//==========================================================
// RDP-MCDA — Google Earth Engine script used to prepare the
// eight-criterion stacks orol_v2_1000m / _100m / _30m.
//
// Study area : Muynak District (FAO GAUL 2015, level 2)
// Period     : startDate = 2025-05-01, endDate = 2025-09-30
//              (Earth Engine filterDate: end date exclusive)
// Output     : Orol_stack_v2_<scale>m.tif, bands in this order:
//              Precipitation, LST, WindSpeed, NDWI, SI2, NDVI,
//              Elevation, Slope; nodata = -9999; EPSG:32640;
//              scales 1000, 100 and 30 m (no explicit resampling,
//              Earth Engine defaults apply).
//
// Only the data preparation and the Export block at the end are
// used by RDP-MCDA. The sections "Module 9 Part 2-4" (min-max
// normalization, fixed weights, suitability map and legend) belong
// to an earlier PMESA workflow and do not affect the exported
// stacks; their weights are not those computed in the paper.
//
// The script is published as it was run; comments in Uzbek are
// original.
//==========================================================

//----------------------------------------------------------
// Study Area
//----------------------------------------------------------

var studyArea = ee.FeatureCollection("FAO/GAUL/2015/level2")
    .filter(ee.Filter.eq('ADM0_NAME', 'Uzbekistan'))
    .filter(ee.Filter.eq('ADM2_NAME', 'Muynak district'));

Map.centerObject(studyArea, 8);
Map.setOptions('ROADMAP');



Map.addLayer(
    studyArea,
    {color:'red'},
    'Study Area',
    false
);
//----------------------------------------------------------
// Study Period
//----------------------------------------------------------

var startDate = '2025-05-01';
var endDate   = '2025-09-30';
//==========================================================
// PMESA Framework
// Module 2
// Sentinel-2 Environmental Variables
// NDVI, NDWI, SI2
//==========================================================






//----------------------------------------------------------
// 3. Sentinel-2 Surface Reflectance
//----------------------------------------------------------

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(studyArea)
    .filterDate(startDate,endDate);
    



//----------------------------------------------------------
// 4. Sentinel-2 Cloud Probability
//----------------------------------------------------------

var cloudCollection =
ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterBounds(studyArea)
    .filterDate(startDate,endDate);



//----------------------------------------------------------
// 5. Join Image Collections
//----------------------------------------------------------

var joined = ee.ImageCollection(
    ee.Join.saveFirst('cloud_mask').apply({

        primary: s2,

        secondary: cloudCollection,

        condition: ee.Filter.equals({

            leftField: 'system:index',

            rightField: 'system:index'

        })

    })
);



//----------------------------------------------------------
// 6. Cloud Mask
//----------------------------------------------------------

function maskClouds(image){

    var cloud =
    ee.Image(image.get('cloud_mask'))
        .select('probability');

    var mask = cloud.lt(20);

    return image
        .updateMask(mask)
        .divide(10000)
        .copyProperties(image,image.propertyNames());

}



//----------------------------------------------------------
// 7. Median Composite
//----------------------------------------------------------

// 7. Median kompozit (v2): avval bulut niqobi bilan; birorta ham toza kuzatuv
//    qolmagan piksellar niqobsiz mavsumiy mediana bilan to'ldiriladi
var compMasked = joined.map(maskClouds).median();
var compAll = s2.map(function (i) {
  return i.divide(10000).copyProperties(i, i.propertyNames());
}).median();
var composite = compMasked.unmask(compAll).clip(studyArea).toFloat();
var fillFlag = compMasked.select('B4').mask().eq(0).clip(studyArea).rename('S2_filled');


//----------------------------------------------------------
// 8. NDVI
//----------------------------------------------------------

var NDVI = composite
.normalizedDifference(['B8','B4'])
.rename('NDVI');



//----------------------------------------------------------
// 9. NDWI
//----------------------------------------------------------

var NDWI = composite
.normalizedDifference(['B3','B8'])
.rename('NDWI');



//----------------------------------------------------------
// 10. SI2
//----------------------------------------------------------

var SI2 = composite.expression(

    'sqrt(B3 * B3 + B4 * B4)',

    {

        B3: composite.select('B3'),

        B4: composite.select('B4')

    }

).rename('SI2');



//----------------------------------------------------------
// 11. Visualization
//----------------------------------------------------------

Map.addLayer(

composite,

{

bands:['B4','B3','B2'],

min:0,

max:0.30

},

'Sentinel RGB',
false

);



Map.addLayer(

NDVI,

{

min:-1,

max:1,

palette:['brown','yellow','green']

},

'NDVI',
false

);



Map.addLayer(
    NDWI,
    {
        min:-0.5,
        max:0.5,
        palette:['white','cyan','blue']
    },
    'NDWI',
    false
);



Map.addLayer(

SI2,

{

min:0,

max:0.5,

palette:['green','yellow','orange','red']

},

'SI2',
false

);



//----------------------------------------------------------
// 12. Information
//----------------------------------------------------------

print('Sentinel-2 Images:', joined.size());

print('Composite Image:', composite);





//==========================================================
// PMESA Framework
// Module 3
// Terrain Variables
// Elevation and Slope
//==========================================================



//----------------------------------------------------------
// 1. Study Area
//----------------------------------------------------------



//----------------------------------------------------------
// 2. Load SRTM DEM
//----------------------------------------------------------

var DEM = ee.Image('USGS/SRTMGL1_003')
            .clip(studyArea)
            .rename('Elevation');



//----------------------------------------------------------
// 3. Calculate Slope
//----------------------------------------------------------

var Slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).clip(studyArea).rename('Slope');



//----------------------------------------------------------
// 4. Visualization Parameters
//----------------------------------------------------------

var demVis = {
  min: -10,
  max: 250,
  palette: [
    '#08306B',
    '#2171B5',
    '#6BAED6',
    '#C7E9B4',
    '#FFFFBF',
    '#FDAE61',
    '#D73027'
  ]
};

var slopeVis = {
  min: 0,
  max: 15,
  palette: [
    '#FFFFFF',
    '#FFF7BC',
    '#FEC44F',
    '#FE9929',
    '#EC7014',
    '#CC4C02',
    '#993404'
  ]
};



//----------------------------------------------------------
// 5. Display Layers
//----------------------------------------------------------

Map.addLayer(
    DEM,
    demVis,
    'Elevation',
    false
);

Map.addLayer(
    Slope,
    slopeVis,
    'Slope',
    false
);



//----------------------------------------------------------
// 6. Information
//----------------------------------------------------------

print('DEM Image', DEM);
print('Slope Image', Slope);

//----------------------------------------------------------
// 3. MODIS LST
//----------------------------------------------------------

var modis = ee.ImageCollection("MODIS/061/MOD11A2")
    .filterBounds(studyArea)
    .filterDate(startDate,endDate)
    .select("LST_Day_1km");


//----------------------------------------------------------
// 4. Mean Composite
//----------------------------------------------------------

var LST = modis
    .mean()
    .multiply(0.02)
    .subtract(273.15)
    .rename("LST")
    .clip(studyArea);


//----------------------------------------------------------
// 5. Visualization
//----------------------------------------------------------

var lstVis = {

    min:20,

    max:50,

    palette:[
        "040274",
        "2c7bb6",
        "abd9e9",
        "ffffbf",
        "fdae61",
        "d7191c"
    ]

};

Map.addLayer(LST,lstVis,"LST",false);

//----------------------------------------------------------
// 3. ERA5-Land Daily
//----------------------------------------------------------

var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(startDate,endDate)
    .filterBounds(studyArea);



//----------------------------------------------------------
// 4. Mean Wind Components
//----------------------------------------------------------

var u = era5
    .select("u_component_of_wind_10m")
    .mean();

var v = era5
    .select("v_component_of_wind_10m")
    .mean();



//----------------------------------------------------------
// 5. Wind Speed
//----------------------------------------------------------

var WS = u.pow(2)
          .add(v.pow(2))
          .sqrt()
          .rename("WindSpeed")
          .clip(studyArea);

//----------------------------------------------------------
// 3. CHIRPS Daily
//----------------------------------------------------------

var chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterDate(startDate,endDate)
    .filterBounds(studyArea);


//----------------------------------------------------------
// 4. Total Precipitation
//----------------------------------------------------------

var Precipitation = chirps
    .sum()
    .rename("Precipitation")
    .clip(studyArea);
//==========================================================
// PMESA Framework
// Module 9
// Part 1
// Raster Harmonization
//==========================================================

// Common projection
//var targetProjection = ee.Projection('EPSG:4326');
//var targetScale = 1000;


//------------------------------
// Sentinel
//------------------------------

//==========================================================
// PMESA Framework
// Module 9
// Part 1
// Raster Harmonization
//==========================================================

var targetScale = 1000;

//------------------------------
// Sentinel
//------------------------------

//NDVI = NDVI.resample('bilinear');

//NDWI = NDWI.resample('bilinear');

//SI2 = SI2.resample('bilinear');

//------------------------------
// DEM
//------------------------------

//DEM = DEM.resample('bilinear');

//Slope = Slope.resample('bilinear');

//------------------------------
// LST
//------------------------------

//LST = LST.resample('bilinear');

//------------------------------
// Wind
//------------------------------

//WS = WS.resample('bilinear');

//------------------------------
// Precipitation
//------------------------------

//Precipitation = Precipitation.resample('bilinear');



print("Raster Harmonization Completed");
//==========================================================
// PMESA Framework
// Module 9
// Part 2
// Min-Max Normalization
//==========================================================

//----------------------------------------------------------
// Benefit Normalization
//----------------------------------------------------------

function normalizeBenefit(image){

  var band = ee.String(image.bandNames().get(0));

  var stats = image.reduceRegion({

      reducer: ee.Reducer.minMax(),

      geometry: studyArea,

      scale: targetScale,

      bestEffort: true,

      maxPixels: 1e13

  });

  var min = ee.Number(stats.get(band.cat('_min')));

  var max = ee.Number(stats.get(band.cat('_max')));

  return image.subtract(min)
              .divide(max.subtract(min));
}


//----------------------------------------------------------
// Cost Normalization
//----------------------------------------------------------

function normalizeCost(image){

  var band = ee.String(image.bandNames().get(0));

  var stats = image.reduceRegion({

      reducer: ee.Reducer.minMax(),

      geometry: studyArea,

      scale: targetScale,

      bestEffort: true,

      maxPixels: 1e13

  });

  var min = ee.Number(stats.get(band.cat('_min')));

  var max = ee.Number(stats.get(band.cat('_max')));

  return ee.Image.constant(max)
          .subtract(image)
          .divide(
              ee.Image.constant(max.subtract(min))
          );

}

//----------------------------------------------------------
// Benefit Criteria
//----------------------------------------------------------

var NDVI_N = normalizeBenefit(NDVI);

var NDWI_N = normalizeBenefit(NDWI);

var Prec_N = normalizeBenefit(Precipitation);


//----------------------------------------------------------
// Cost Criteria
//----------------------------------------------------------

var SI2_N = normalizeCost(SI2);

var DEM_N = normalizeCost(DEM);

var Slope_N = normalizeCost(Slope);

var LST_N = normalizeCost(LST);

var Wind_N = normalizeCost(WS);


//----------------------------------------------------------
// Check
//----------------------------------------------------------

print("NDVI Normalized", NDVI_N);

print("NDWI Normalized", NDWI_N);

print("SI2 Normalized", SI2_N);

print("DEM Normalized", DEM_N);

print("Slope Normalized", Slope_N);

print("LST Normalized", LST_N);

print("Wind Normalized", Wind_N);

print("Precipitation Normalized", Prec_N);

print("Normalization Completed");
//==========================================================
// PMESA Framework
// Module 9
// Part 3
// Entropy Weight Method (EWM)
// Weighted Overlay
//==========================================================

//----------------------------------------------------------
// Entropy Weights
//----------------------------------------------------------

var wNDVI  = 0.012167;
var wNDWI  = 0.050534;
var wSI2   = 0.047894;
var wDEM   = 0.007897;
var wSlope = 0.001694;
var wLST   = 0.291283;
var wWind  = 0.140117;
var wPrec  = 0.448413;


//----------------------------------------------------------
// PMESA Suitability Index
//----------------------------------------------------------

var PMESA = NDVI_N.multiply(wNDVI)

.add(NDWI_N.multiply(wNDWI))

.add(SI2_N.multiply(wSI2))

.add(DEM_N.multiply(wDEM))

.add(Slope_N.multiply(wSlope))

.add(LST_N.multiply(wLST))

.add(Wind_N.multiply(wWind))

.add(Prec_N.multiply(wPrec))

.rename('PMESA');


//----------------------------------------------------------
// Clip
//----------------------------------------------------------

PMESA = PMESA.clip(studyArea);


//----------------------------------------------------------
// Statistics
//----------------------------------------------------------

var pmesaStats = PMESA.reduceRegion({

    reducer: ee.Reducer.minMax()

        .combine({

            reducer2: ee.Reducer.mean(),

            sharedInputs:true

        })

        .combine({

            reducer2: ee.Reducer.stdDev(),

            sharedInputs:true

        }),

    geometry: studyArea,

    scale: targetScale,

    bestEffort:true,

    maxPixels:1e13

});

print('PMESA Statistics', pmesaStats);


//----------------------------------------------------------
// Check
//----------------------------------------------------------

print('PMESA Suitability Index', PMESA);
//==========================================================
// PMESA Framework
// Module 9
// Part 4
// Visualization
//==========================================================

// Visualization parameters
var visPMESA = {
    min: 0,
    max: 1,
    palette: [
        '#8B0000', // Very Low
        '#FF4500',
        '#FFD700',
        '#7CFC00',
        '#006400'  // Very High
    ]
};

// Display PMESA
Map.addLayer(
    PMESA,
    visPMESA,
    'PMESA Suitability Index',
    false
);

//----------------------------------------------------------
// Histogram
//----------------------------------------------------------

print(ui.Chart.image.histogram({
    image: PMESA,
    region: studyArea,
    scale: targetScale,
    maxPixels: 1e13
}).setOptions({
    title: 'PMESA Suitability Distribution',
    hAxis: {title: 'Suitability Index'},
    vAxis: {title: 'Pixel Count'}
}));
//----------------------------------------------------------
// PMESA Classes
//----------------------------------------------------------

//----------------------------------------------------------
// PMESA Classes (TO'G'RILANGAN KOD)
//----------------------------------------------------------
var PMESA_Class = PMESA.expression(
  "(b <= 0.2) ? 1 " +
  ": (b <= 0.4) ? 2 " +
  ": (b <= 0.6) ? 3 " +
  ": (b <= 0.8) ? 4 " +
  ": 5",
  { b: PMESA.select('PMESA') }
)
.rename('Suitability')
.clip(studyArea); // <-- Mana shu clip() qatlamni faqat tuman chegarasida qoldiradi!

Map.addLayer(

PMESA_Class,

{

min:1,

max:5,

palette:[

'#8B0000',

'#FF4500',

'#FFD700',

'#7CFC00',

'#006400'

]

},

'PMESA Suitability Classes'

);
var boundary = ee.Image().byte().paint(studyArea,1,2);

Map.addLayer(

boundary,

{

palette:['black']

},

'Study Area Boundary'

);
var title = ui.Label({

value:'PMESA Suitability Map\nMuynak District',

style:{

position:'top-center',

fontWeight:'bold',

fontSize:'22px',

backgroundColor:'white',

padding:'8px'

}

});

Map.add(title);
var legend = ui.Panel({

style:{

position:'bottom-right',

padding:'8px',

backgroundColor:'white'

}

});

legend.add(ui.Label('Suitability Classes'));

function addLegend(color,name){

var row = ui.Panel({

layout:ui.Panel.Layout.Flow('horizontal')

});

row.add(ui.Label({

style:{

backgroundColor:color,

padding:'8px',

margin:'0 8px 4px 0'

}

}));

row.add(ui.Label(name));

legend.add(row);

}

addLegend('#006400','Very High');
addLegend('#7CFC00','High');
addLegend('#FFD700','Moderate');
addLegend('#FF4500','Low');
addLegend('#8B0000','Very Low');

Map.add(legend);
var north = ui.Label({

value:'↑\nN',

style:{

position:'top-right',

fontWeight:'bold',

fontSize:'26px',

backgroundColor:'white',

padding:'6px'

}

});

Map.add(north);

// Tekshiruv
var cnt = ee.Image.cat([Precipitation, NDVI, Slope]).reduceRegion({
  reducer: ee.Reducer.count(), geometry: studyArea.geometry(), scale: 1000, maxPixels: 1e13});
print('Kataklar soni (1000 m):', cnt);
print("To'ldirilgan kataklar soni:", fillFlag.reduceRegion({
  reducer: ee.Reducer.sum(), geometry: studyArea.geometry(), scale: 1000, maxPixels: 1e13}));
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, 'RGB v2');
Map.addLayer(fillFlag.selfMask(), {palette: ['red']}, "To'ldirilgan joylar");

// Eksport (v2): 8 mezonli stack + to'ldirish bayrog'i, 1000 / 100 / 30 m
var stack = ee.Image.cat([Precipitation, LST, WS, NDWI, SI2, NDVI, DEM, Slope])
              .toFloat().unmask(-9999);
[1000, 100, 30].forEach(function (s) {
  Export.image.toDrive({
    image: stack, description: 'Orol_stack_v2_' + s + 'm', folder: 'Orol_export_v2',
    region: studyArea.geometry(), scale: s, crs: 'EPSG:32640',
    maxPixels: 1e13, fileFormat: 'GeoTIFF'});
  Export.image.toDrive({
    image: fillFlag.unmask(255).toByte(), description: 'Orol_s2flag_v2_' + s + 'm',
    folder: 'Orol_export_v2', region: studyArea.geometry(), scale: s, crs: 'EPSG:32640',
    maxPixels: 1e13, fileFormat: 'GeoTIFF'});
});
