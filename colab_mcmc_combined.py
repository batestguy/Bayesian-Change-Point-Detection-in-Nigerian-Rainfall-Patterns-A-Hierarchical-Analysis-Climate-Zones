"""
Combined MCMC pipeline — all steps in one script.
Saves traces after EVERY model so partial progress is preserved.
Reduced draws for faster completion (~60 min total).
"""
import io, os, time, shutil, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"PyMC {pm.__version__}, ArviZ {az.__version__}")
print(f"Start time: {time.strftime('%H:%M:%S')}")

OUTPUT_DIR = Path("/content/output")
TRACES_DIR = OUTPUT_DIR / "traces"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
for d in [TRACES_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ZONE_ORDER = ["Sahel", "Guinea Savanna", "Coastal"]
ZONE_COLORS = {"Sahel": "#E74C3C", "Guinea Savanna": "#2ECC71", "Coastal": "#3498DB"}
ZONES = {
    "Sahel": [
        {"city": "Maiduguri", "lat": 11.846, "lon": 13.160},
        {"city": "Kano",      "lat": 12.000, "lon": 8.517},
        {"city": "Sokoto",    "lat": 13.060, "lon": 5.240},
    ],
    "Guinea Savanna": [
        {"city": "Abuja",  "lat": 9.058, "lon": 7.489},
        {"city": "Jos",    "lat": 9.897, "lon": 8.858},
        {"city": "Ilorin", "lat": 8.490, "lon": 4.542},
    ],
    "Coastal": [
        {"city": "Lagos",          "lat": 6.524, "lon": 3.379},
        {"city": "Port Harcourt",  "lat": 4.815, "lon": 7.050},
        {"city": "Calabar",        "lat": 4.976, "lon": 8.337},
    ],
}
IEEE_SINGLE_COL = (3.5, 2.5)
IEEE_DOUBLE_COL = (7.16, 3.0)
IEEE_DOUBLE_COL_TALL = (7.16, 5.0)

def get_all_stations():
    stations = []
    for zone, cities in ZONES.items():
        for c in cities:
            stations.append({**c, "zone": zone})
    return stations

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "lines.linewidth": 1.0, "lines.markersize": 3,
})

def savefig(fig, name, formats=("pdf", "png")):
    for fmt in formats:
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}")
    plt.close(fig)

def compute_hdi(samples, prob=0.94):
    try:
        return az.hdi(samples, prob=prob)
    except TypeError:
        return az.hdi(samples, hdi_prob=prob)

def checkpoint(msg):
    """Save progress marker and zip current output."""
    print(f"\n>>> CHECKPOINT: {msg} at {time.strftime('%H:%M:%S')}")
    shutil.make_archive("/content/mcmc_output", "zip", str(OUTPUT_DIR))
    sz = os.path.getsize("/content/mcmc_output.zip") / 1024 / 1024
    print(f"    Zip updated: {sz:.1f} MB")

CSV_DATA = """\
year,zone,annual_precip_mm,city_std,city_min,city_max
1950,Coastal,2466.2666666666664,369.5500552473687,2096.6,2835.7
1950,Guinea Savanna,1102.3333333333333,390.8888887309709,663.5,1413.2
1950,Sahel,788.5666666666666,54.30150396935004,726.1,824.5
1951,Coastal,2676.9666666666667,485.5027737648195,2217.4,3184.8
1951,Guinea Savanna,1529.3333333333333,561.4095682595134,932.7,2047.2
1951,Sahel,684.6,145.94440722412082,566.9,847.9
1952,Coastal,2702.6,453.6138004955317,2298.4,3193.2
1952,Guinea Savanna,1195.6000000000001,478.2675088274343,695.1,1648.0
1952,Sahel,730.4666666666667,75.40599003615914,644.6,785.9
1953,Coastal,2363.7,607.0209304463891,1769.3,2982.6
1953,Guinea Savanna,1154.5666666666666,484.47553429800075,614.3,1550.3999999999999
1953,Sahel,762.3333333333334,131.50062103782372,615.2,868.4
1954,Coastal,2244.633333333333,608.6105514475848,1580.2,2775.1
1954,Guinea Savanna,1094.8333333333333,422.08452155147,675.9,1520.0
1954,Sahel,765.5,166.13521601394447,658.6,956.9
1955,Coastal,2575.3333333333335,519.1978941919289,2044.8,3082.4
1955,Guinea Savanna,1389.0,430.4524480125534,906.0,1732.1
1955,Sahel,512.1,146.04427410891532,371.7,663.2
1956,Coastal,2541.1,585.589045321034,1918.4,3080.7
1956,Guinea Savanna,1116.1333333333334,382.18829565193823,684.4,1411.2
1956,Sahel,899.2333333333335,45.490695019238075,870.8000000000001,951.7
1957,Coastal,2729.9333333333334,260.8682106607344,2544.9,3028.3
1957,Guinea Savanna,1499.0333333333335,519.2517340686821,934.8,1956.8
1957,Sahel,736.3000000000001,106.8529831123118,654.9,857.3
1958,Coastal,2160.7999999999997,633.7185810752277,1515.8,2782.6
1958,Guinea Savanna,961.1666666666666,463.69180856829183,467.1,1386.9
1958,Sahel,631.1666666666666,85.89868062626651,552.0,722.5
1959,Coastal,2125.0333333333333,677.5100761858331,1445.4,2800.4
1959,Guinea Savanna,1040.9666666666667,220.29801028001444,786.9,1178.9
1959,Sahel,525.5666666666667,164.3296787964162,399.8,711.5
1960,Coastal,2506.6666666666665,718.6210429241085,1740.3,3165.4
1960,Guinea Savanna,1267.1333333333334,390.896678590818,904.2,1681.0
1960,Sahel,431.5,123.50157893727513,341.5,572.3
1961,Coastal,2138.9666666666667,416.1223417858422,1700.5,2528.4
1961,Guinea Savanna,1035.6666666666667,419.20172152954393,577.5,1400.0
1961,Sahel,578.4666666666667,114.38777615345678,488.5,707.2
1962,Coastal,2441.4,550.1576955746416,1886.7,2986.9
1962,Guinea Savanna,1360.8333333333333,550.9686046712039,745.8,1809.3
1962,Sahel,578.4666666666667,145.49331027003728,410.5,665.4
1963,Coastal,2154.5,121.96446203710319,2068.6,2294.1
1963,Guinea Savanna,1337.6000000000001,619.3731589276372,730.1,1968.2
1963,Sahel,520.5333333333333,189.18404619135654,384.1,736.5
1964,Coastal,1913.7666666666667,473.7350771616277,1386.4,2303.3
1964,Guinea Savanna,854.9,448.6992645414075,427.2,1322.0
1964,Sahel,483.93333333333334,97.28064213055612,403.1,591.9
1965,Coastal,2435.5,465.2862774679693,1899.3,2732.9
1965,Guinea Savanna,1100.5,456.6046320395797,659.8,1571.5
1965,Sahel,402.90000000000003,138.5602035217905,257.2,533.0
1966,Coastal,2254.3,465.1985705051124,1747.2,2661.3
1966,Guinea Savanna,1205.0666666666666,593.8174158218444,644.2,1827.1
1966,Sahel,435.8666666666666,124.70911487671351,309.0,558.3
1967,Coastal,2665.9666666666667,721.890825079065,1918.4,3359.1
1967,Guinea Savanna,1253.5,475.69930628496815,749.6,1694.8
1967,Sahel,718.1333333333333,149.5730367858236,545.8,814.2
1968,Coastal,2932.9,327.5276476879472,2648.3,3290.9
1968,Guinea Savanna,1520.3666666666668,480.31557681729777,1042.3,2002.9
1968,Sahel,581.3666666666667,221.50197139830007,438.2,836.5
1969,Coastal,2824.7000000000003,698.4895203222451,2081.6,3467.8
1969,Guinea Savanna,1425.3666666666668,567.9968691228265,792.8,1891.7
1969,Sahel,661.6333333333333,189.83709683129197,502.6,871.8
1970,Coastal,2583.1,555.0052972720171,2000.7,3105.9
1970,Guinea Savanna,1311.2,461.4809530197318,785.0,1647.1
1970,Sahel,541.7333333333333,160.99174927099006,382.90000000000003,704.8
1971,Coastal,2821.6,460.0000326086945,2361.7,3281.7
1971,Guinea Savanna,1403.7666666666664,441.1472807729107,908.9,1755.8
1971,Sahel,568.1666666666666,178.06853549499795,391.6,747.7
1972,Coastal,2585.7333333333336,653.9286989675048,1909.9,3215.3
1972,Guinea Savanna,1491.0999999999997,659.3038449758957,833.1,2151.7
1972,Sahel,486.5333333333333,168.6939931750189,355.6,676.9
1973,Coastal,2855.8333333333335,641.6783799796696,2204.5,3487.4
1973,Guinea Savanna,1250.7666666666667,443.2879011808617,767.2,1637.9
1973,Sahel,440.7,114.8141106310544,331.1,560.1
1974,Coastal,2744.5,666.9354091664347,2048.3,3377.7
1974,Guinea Savanna,1291.4333333333334,513.9922405380586,749.8,1772.4
1974,Sahel,618.8333333333334,173.33716662427975,418.7,721.3
1975,Coastal,2572.6666666666665,563.1589858408844,1995.0,3120.1
1975,Guinea Savanna,1379.7333333333336,430.76324510493396,919.6,1773.4
1975,Sahel,680.5666666666667,214.09820955190943,499.8,917.0
1976,Coastal,2755.6,789.2324815920847,1944.1,3520.5
1976,Guinea Savanna,1433.6333333333332,606.5749445314514,798.2,2006.5
1976,Sahel,532.4,171.21585791041667,432.6,730.1
1977,Coastal,2294.233333333333,608.8196476242643,1668.0,2884.0
1977,Guinea Savanna,1077.1000000000001,436.6558255651697,616.2,1484.6
1977,Sahel,587.6999999999999,109.82882135395971,493.0,708.1
1978,Coastal,2590.9666666666667,642.572239778014,1901.3,3172.8
1978,Guinea Savanna,1600.8666666666668,706.3397647968953,818.6,2191.9
1978,Sahel,615.6666666666666,176.95695333423134,464.7,810.4
1979,Coastal,2798.5333333333333,570.8103917531051,2214.3,3354.9
1979,Guinea Savanna,1414.8999999999999,509.0600259301451,864.0,1867.9
1979,Sahel,583.9666666666667,113.02461383846146,511.5,714.2
1980,Coastal,3098.6,538.6884349974482,2579.2,3654.7
1980,Guinea Savanna,1465.6000000000001,427.79976624584543,975.6,1764.8
1980,Sahel,568.3000000000001,209.92922616920214,445.7,810.7
1981,Coastal,2539.0333333333333,635.4322492078392,1933.7,3200.8
1981,Guinea Savanna,1332.6000000000001,538.7535707538281,785.5,1862.6
1981,Sahel,579.1666666666666,185.08631319828413,392.0,762.1
1982,Coastal,2453.1666666666665,868.969264895677,1530.5,3256.0
1982,Guinea Savanna,1426.2666666666667,578.6661933561812,773.0,1874.5
1982,Sahel,435.1333333333334,110.29326059797728,323.1,543.6
1983,Coastal,2158.9666666666667,799.8650532016844,1321.5,2915.0
1983,Guinea Savanna,1062.3666666666666,403.0059097995131,633.1,1432.6
1983,Sahel,354.5,122.61859565335105,251.9,490.3
1984,Coastal,2385.7999999999997,741.4359244061484,1550.7,2966.7
1984,Guinea Savanna,1404.2,646.9413651947137,718.6,2003.9
1984,Sahel,354.6333333333334,106.53226428333028,250.6,463.5
1985,Coastal,2763.6666666666665,834.9785885478341,1826.4,3428.1
1985,Guinea Savanna,1305.4333333333334,362.68843837835993,901.4,1602.9
1985,Sahel,511.3,149.16759031371396,348.1,640.6
1986,Coastal,2530.9,862.3483576838306,1549.7,3168.4
1986,Guinea Savanna,1282.0333333333333,481.53145622413217,734.2,1638.3
1986,Sahel,509.06666666666666,162.82709643463318,358.7,682.0
1987,Coastal,2551.7999999999997,641.1566033349419,1821.1,3020.3
1987,Guinea Savanna,1331.4333333333334,528.6816654030414,747.0,1776.4
1987,Sahel,449.09999999999997,162.49759998227663,310.2,627.8
1988,Coastal,2782.0333333333333,609.246750777822,2155.6,3372.5
1988,Guinea Savanna,1431.3,439.11653350790607,930.1,1748.4
1988,Sahel,645.6333333333333,70.09681400273003,602.2,726.5
1989,Coastal,2457.6666666666665,552.6649286261372,1842.7,2912.8
1989,Guinea Savanna,1314.2333333333333,499.0881819211243,847.7,1840.5
1989,Sahel,505.59999999999997,104.87339986860344,397.0,606.3
1990,Coastal,2570.3333333333335,906.8022625320987,1595.0,3387.9
1990,Guinea Savanna,1262.8999999999999,468.8274842626017,765.4,1696.5
1990,Sahel,392.1000000000001,190.62730654342258,237.8,605.2
1991,Coastal,2518.1666666666665,679.8188165484487,1790.8,3137.5
1991,Guinea Savanna,1420.9333333333334,449.48855751101536,919.8,1788.5
1991,Sahel,549.4666666666667,204.07857147023876,417.2,784.5
1992,Coastal,2359.2999999999997,836.803417775047,1507.9,3180.7
1992,Guinea Savanna,1340.9333333333334,466.4706028608162,864.1,1796.3
1992,Sahel,564.4666666666667,208.49226204666047,347.5,763.3
1993,Coastal,2529.233333333333,868.1541932936415,1588.0,3298.6
1993,Guinea Savanna,1387.8999999999999,600.9143366570645,741.7,1929.8999999999999
1993,Sahel,375.76666666666665,203.87609799418212,227.2,608.2
1994,Coastal,2382.5333333333333,819.0508551569513,1562.8,3200.9
1994,Guinea Savanna,1469.8,541.1191735653063,845.2,1796.8
1994,Sahel,623.9666666666667,131.04435635819397,474.1,717.0
1995,Coastal,2692.0666666666666,878.9711390787147,1799.7,3557.0
1995,Guinea Savanna,1420.6333333333332,481.2893239345055,870.9,1766.1
1995,Sahel,470.2333333333333,177.46842911721885,287.4,641.8
1996,Coastal,2478.7333333333336,484.13868192216717,1991.1,2959.3
1996,Guinea Savanna,1385.6333333333332,493.6841129035178,851.4,1825.0
1996,Sahel,423.2,113.47528365243245,296.4,515.2
1997,Coastal,2400.1666666666665,772.2498580986166,1569.6,3096.5
1997,Guinea Savanna,1449.3999999999999,569.5166986138333,809.7,1901.3
1997,Sahel,423.6333333333334,144.39703367220997,289.1,576.2
1998,Coastal,2158.7999999999997,684.1463951523826,1447.5,2812.1
1998,Guinea Savanna,1376.1333333333332,454.71070290167273,854.0,1685.1
1998,Sahel,592.1333333333333,151.95714308097965,483.6,765.8
1999,Coastal,2671.4333333333334,683.3954662809326,1956.0,3317.5
1999,Guinea Savanna,1365.7333333333333,411.1311753362098,913.9,1717.8
1999,Sahel,586.3000000000001,164.7162104955065,429.2,757.7
2000,Coastal,2326.133333333333,652.4289412137796,1637.0,2934.3
2000,Guinea Savanna,1331.5,419.84755566753034,877.8,1706.3
2000,Sahel,431.3,130.6762411458181,311.9,570.9
2001,Coastal,2191.4333333333334,665.6997546441887,1464.2,2770.7
2001,Guinea Savanna,1158.8999999999999,568.9661940748325,579.3,1716.6
2001,Sahel,482.6333333333334,215.71463402683958,328.2,729.1
2002,Coastal,2591.9,845.5848685968783,1668.7,3328.8
2002,Guinea Savanna,1362.5666666666666,424.0668147041611,931.8,1779.6
2002,Sahel,413.59999999999997,100.84423632513656,301.0,495.59999999999997
2003,Coastal,2336.2999999999997,711.5500474316617,1533.0,2887.4
2003,Guinea Savanna,1154.8333333333333,427.0443809878937,722.7,1576.6
2003,Sahel,496.0,220.3814647378495,340.5,748.2
2004,Coastal,2413.5333333333333,786.35319248626,1548.1,3084.2
2004,Guinea Savanna,1240.9,566.3956567630088,616.1,1720.7
2004,Sahel,386.3666666666666,111.53301454427441,315.1,514.9
2005,Coastal,2296.3333333333335,797.8702672322947,1436.1,3012.1
2005,Guinea Savanna,1143.6000000000001,430.1377337551311,665.0,1497.9
2005,Sahel,463.6666666666667,130.48656380383895,335.0,595.9
2006,Coastal,2445.5,584.2251620736649,1809.5,2958.3
2006,Guinea Savanna,1232.5,463.52800346904604,726.5,1636.6
2006,Sahel,387.6000000000001,171.29538814574082,270.5,584.2
2007,Coastal,2752.633333333333,844.9368990246154,1854.4,3531.6
2007,Guinea Savanna,1144.8999999999999,282.8958819071073,833.7,1386.5
2007,Sahel,449.0333333333333,73.84106806739277,370.40000000000003,516.9
2008,Coastal,2432.2999999999997,815.5287058099179,1597.6,3227.2
2008,Guinea Savanna,1266.0666666666666,464.2986790131255,808.7,1737.0
2008,Sahel,499.43333333333334,128.721883661378,368.3,625.6
2009,Coastal,2657.766666666667,1012.271714182182,1508.4,3416.6
2009,Guinea Savanna,1258.3,596.3357359742915,588.9,1732.8
2009,Sahel,398.6333333333334,132.77139501162642,318.8,551.9
2010,Coastal,2496.1,798.7535039046777,1590.7,3101.1
2010,Guinea Savanna,1303.9333333333332,470.35155291901964,818.1,1757.1
2010,Sahel,600.1,92.08778420615847,536.5,705.7
2011,Coastal,2397.866666666667,856.1761578865259,1488.6,3188.6
2011,Guinea Savanna,1105.5666666666666,420.8258824422915,663.2,1500.9
2011,Sahel,464.7,267.65483369444314,252.2,765.3
2012,Coastal,2372.9,527.5655693845081,1831.8,2885.8
2012,Guinea Savanna,1453.8666666666668,482.50963030113013,905.2,1812.1
2012,Sahel,587.1,192.16407052308193,426.5,800.0
2013,Coastal,2502.2999999999997,1126.1403820128287,1272.3,3482.7
2013,Guinea Savanna,1264.0666666666666,522.7848920285793,669.9,1653.5
2013,Sahel,512.0,101.14776319820429,429.1,624.7
2014,Coastal,2557.2000000000003,686.7719563290278,1827.4,3190.8
2014,Guinea Savanna,1353.2666666666667,428.6588542574775,864.0,1662.8
2014,Sahel,507.09999999999997,190.0028420840067,383.7,725.9
2015,Coastal,2420.233333333333,918.7217551213932,1455.7,3285.0
2015,Guinea Savanna,1078.7,336.55322015990276,700.0,1343.6
2015,Sahel,503.1666666666667,130.9021517521134,425.5,654.3
2016,Coastal,2767.4666666666667,1024.4264850799852,1714.8,3761.1
2016,Guinea Savanna,1429.6333333333332,345.48379605031164,1049.5,1724.5
2016,Sahel,547.0666666666667,75.3166869513877,460.1,591.0
2017,Coastal,2054.5,629.7351586182878,1361.8,2592.4
2017,Guinea Savanna,783.8333333333334,166.69401109018082,633.5,963.1
2017,Sahel,444.06666666666666,39.710745816886075,398.9,473.5
2018,Coastal,2080.633333333333,770.1463843538663,1192.7,2567.1
2018,Guinea Savanna,1015.1,162.9236937955925,847.0,1172.3
2018,Sahel,464.93333333333334,50.906810284414135,427.5,522.9
2019,Coastal,2622.5333333333333,1119.5613084299284,1452.9,3684.2
2019,Guinea Savanna,1148.8666666666666,141.14589378842493,1000.1,1280.9
2019,Sahel,476.59999999999997,121.1678175094361,338.4,564.6
2020,Coastal,1941.1333333333332,656.1985776678683,1186.0,2372.8
2020,Guinea Savanna,775.8666666666667,229.55385715194012,511.0,917.2
2020,Sahel,473.8999999999999,86.20046403587396,419.7,573.3
2021,Coastal,2113.2999999999997,782.5676456383818,1216.1,2655.1
2021,Guinea Savanna,674.9333333333334,207.173703286236,438.5,824.7
2021,Sahel,275.96666666666664,107.09422642389892,158.1,367.3
2022,Coastal,1812.2333333333333,523.2265315648025,1213.0,2178.6
2022,Guinea Savanna,1005.2666666666668,130.85412998195108,860.0,1113.9
2022,Sahel,634.6333333333333,119.53879426083121,534.8,767.1
2023,Coastal,2164.766666666667,567.9296288567215,1519.8,2590.0
2023,Guinea Savanna,810.2666666666668,145.76338131826293,646.5,925.8
2023,Sahel,482.1666666666667,135.78683048563042,353.1,623.8
"""

annual_zone = pd.read_csv(io.StringIO(CSV_DATA))
print(f"Loaded {len(annual_zone)} records, zones: {annual_zone['zone'].unique().tolist()}")
annual_zone.to_csv(DATA_DIR / "annual_zone_rainfall.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════
# STEP 1: Single Change-Point Models (per zone)
# ═════════════════════════════════════════════════════════════════════════

def build_single_changepoint_model(y, years):
    N = len(y)
    y_mean, y_std = float(y.mean()), float(y.std())
    with pm.Model(coords={"year": years}) as model:
        tau = pm.Normal("tau", mu=N / 2, sigma=N / 4)
        mu_1 = pm.Normal("mu_1", mu=y_mean, sigma=y_std * 2)
        mu_2 = pm.Normal("mu_2", mu=y_mean, sigma=y_std * 2)
        sigma = pm.HalfNormal("sigma", sigma=y_std)
        idx = np.arange(N, dtype="float64")
        weight = pm.math.sigmoid(5 * (idx - tau))
        mu = mu_1 * (1 - weight) + mu_2 * weight
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y, dims="year")
        pm.Deterministic("tau_year", tau + years[0])
        pm.Deterministic("shift_magnitude", mu_2 - mu_1)
    return model

print("\n" + "=" * 60)
print("STEP 1/4: Single Change-Point Models")
print("=" * 60)

single_cp_traces = {}
t0_all = time.time()

for zone in ZONE_ORDER:
    zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
    y = zdf["annual_precip_mm"].values.astype("float64")
    years = zdf["year"].values

    print(f"\n  Fitting {zone} ({len(y)} years, mean={y.mean():.0f} mm)...")
    t0 = time.time()
    model = build_single_changepoint_model(y, years)
    with model:
        trace = pm.sample(draws=2000, tune=1000, chains=4,
                          target_accept=0.95, random_seed=42,
                          return_inferencedata=True)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    summary = az.summary(trace, var_names=["tau_year", "mu_1", "mu_2",
                                            "sigma", "shift_magnitude"])
    print(summary)
    rhat_ok = (pd.to_numeric(summary["r_hat"], errors="coerce") < 1.01).all()
    ess_ok = (pd.to_numeric(summary["ess_bulk"], errors="coerce") > 400).all()
    print(f"    R-hat: {'PASS' if rhat_ok else 'FAIL'}  |  ESS: {'PASS' if ess_ok else 'FAIL'}")

    trace.to_netcdf(str(TRACES_DIR / f"single_changepoint_{zone.replace(' ', '_')}.nc"))
    single_cp_traces[zone] = trace
    checkpoint(f"Single CP {zone} done")

print(f"\nStep 1 total: {time.time() - t0_all:.1f}s")
print("\nChange-Point Estimates (Median [94% HDI]):")
for zone in ZONE_ORDER:
    tau_s = single_cp_traces[zone]["posterior"]["tau_year"].values.flatten()
    shift_s = single_cp_traces[zone]["posterior"]["shift_magnitude"].values.flatten()
    hdi = compute_hdi(tau_s, prob=0.94)
    print(f"  {zone}: {np.median(tau_s):.1f} [{hdi[0]:.1f}, {hdi[1]:.1f}], "
          f"shift = {np.median(shift_s):+.1f} mm/yr")


# ═════════════════════════════════════════════════════════════════════════
# STEP 2: Hierarchical Change-Point Model
# ═════════════════════════════════════════════════════════════════════════

def prepare_zone_data(df):
    zone_data = []
    for zone in ZONE_ORDER:
        zdf = df[df["zone"] == zone].sort_values("year")
        y = zdf["annual_precip_mm"].values.astype("float64")
        zone_data.append({
            "name": zone, "y": y, "years": zdf["year"].values,
            "start_year": int(zdf["year"].min()),
            "y_mean": float(y.mean()), "y_std": float(y.std()),
        })
    return zone_data

def build_hierarchical_model(zone_data):
    zone_names = [z["name"] for z in zone_data]
    with pm.Model(coords={"zone": zone_names}) as model:
        tau_mu = pm.Normal("tau_mu", mu=30, sigma=10)
        tau_sigma = pm.HalfNormal("tau_sigma", sigma=5)
        tau_offset = pm.Normal("tau_offset", mu=0, sigma=1, dims="zone")
        tau_raw = pm.Deterministic("tau_raw", tau_mu + tau_sigma * tau_offset, dims="zone")
        steepness = pm.HalfNormal("steepness", sigma=5)

        prior_means = np.array([z["y_mean"] for z in zone_data])
        prior_stds = np.array([z["y_std"] for z in zone_data])
        mu_before = pm.Normal("mu_before", mu=prior_means, sigma=200, dims="zone")
        mu_after = pm.Normal("mu_after", mu=prior_means, sigma=200, dims="zone")
        sigma = pm.HalfNormal("sigma", sigma=prior_stds, dims="zone")

        for i, z in enumerate(zone_data):
            N = len(z["y"])
            idx = np.arange(N, dtype="float64")
            weight = pm.math.sigmoid(steepness * (idx - tau_raw[i]))
            mu_t = mu_before[i] * (1 - weight) + mu_after[i] * weight
            pm.Normal(f"obs_{z['name']}", mu=mu_t, sigma=sigma[i], observed=z["y"])

        start_years = np.array([z["start_year"] for z in zone_data])
        pm.Deterministic("tau_year", tau_raw + start_years, dims="zone")
        pm.Deterministic("shift_magnitude", mu_after - mu_before, dims="zone")
        pm.Deterministic("tau_year_group", tau_mu + start_years[0])
    return model

print("\n" + "=" * 60)
print("STEP 2/4: Hierarchical Change-Point Model")
print("=" * 60)

zone_data = prepare_zone_data(annual_zone)
model_h = build_hierarchical_model(zone_data)

print("\nSampling hierarchical model...")
t0 = time.time()
with model_h:
    hier_trace = pm.sample(draws=2000, tune=1000, chains=4,
                           target_accept=0.95, random_seed=42,
                           return_inferencedata=True)
    hier_ppc = pm.sample_posterior_predictive(hier_trace, random_seed=42)

print(f"Sampling took {time.time() - t0:.1f}s")

var_names = ["tau_year", "tau_year_group", "mu_before", "mu_after",
             "sigma", "shift_magnitude", "steepness", "tau_mu", "tau_sigma"]
summary = az.summary(hier_trace, var_names=var_names)
print("\nPosterior Summary:")
print(summary)
rhat_ok = (pd.to_numeric(summary["r_hat"], errors="coerce") < 1.01).all()
ess_ok = (pd.to_numeric(summary["ess_bulk"], errors="coerce") > 400).all()
print(f"\nR-hat: {'PASS' if rhat_ok else 'FAIL'}  |  ESS: {'PASS' if ess_ok else 'FAIL'}")

hier_trace.to_netcdf(str(TRACES_DIR / "hierarchical_changepoint.nc"))

try:
    pp_groups = list(hier_ppc.children) if hasattr(hier_ppc, 'children') else (list(hier_ppc.groups()) if hasattr(hier_ppc, 'groups') else [])
    if "posterior_predictive" in pp_groups:
        hier_ppc.to_netcdf(str(TRACES_DIR / "hierarchical_ppc.nc"))
        print("Saved hierarchical PPC")
except Exception as e:
    print(f"PPC save note: {e}")

print("\nHierarchical Results:")
tau_group = hier_trace["posterior"]["tau_year_group"].values.flatten()
hdi_g = compute_hdi(tau_group, prob=0.94)
print(f"  Group: {np.median(tau_group):.1f} [{hdi_g[0]:.1f}, {hdi_g[1]:.1f}]")
print(f"  tau_sigma: {hier_trace['posterior']['tau_sigma'].values.flatten().mean():.2f}")
for zone in ZONE_ORDER:
    tau_z = hier_trace["posterior"]["tau_year"].sel(zone=zone).values.flatten()
    shift = hier_trace["posterior"]["shift_magnitude"].sel(zone=zone).values.flatten()
    hdi_z = compute_hdi(tau_z, prob=0.94)
    print(f"  {zone}: tau={np.median(tau_z):.1f} [{hdi_z[0]:.1f}, {hdi_z[1]:.1f}], "
          f"shift={np.median(shift):+.1f} mm/yr")

checkpoint("Hierarchical model done")


# ═════════════════════════════════════════════════════════════════════════
# STEP 3: Model Comparison (LOO-IC & WAIC)
# ═════════════════════════════════════════════════════════════════════════

def build_null_model(y):
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=float(y.mean()), sigma=float(y.std()) * 2)
        sigma = pm.HalfNormal("sigma", sigma=float(y.std()))
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model

def build_one_cp_model(y):
    N = len(y)
    y_mean, y_std = float(y.mean()), float(y.std())
    with pm.Model() as model:
        tau = pm.Normal("tau", mu=N / 2, sigma=N / 4)
        mu_1 = pm.Normal("mu_1", mu=y_mean, sigma=y_std * 2)
        mu_2 = pm.Normal("mu_2", mu=y_mean, sigma=y_std * 2)
        sigma = pm.HalfNormal("sigma", sigma=y_std)
        idx = np.arange(N, dtype="float64")
        weight = pm.math.sigmoid(5 * (idx - tau))
        mu = mu_1 * (1 - weight) + mu_2 * weight
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model

def build_two_cp_model(y):
    N = len(y)
    y_mean, y_std = float(y.mean()), float(y.std())
    with pm.Model() as model:
        tau_1 = pm.Normal("tau_1", mu=N / 3, sigma=N / 6)
        tau_2 = pm.Normal("tau_2", mu=2 * N / 3, sigma=N / 6)
        mu_1 = pm.Normal("mu_1", mu=y_mean, sigma=y_std * 2)
        mu_2 = pm.Normal("mu_2", mu=y_mean, sigma=y_std * 2)
        mu_3 = pm.Normal("mu_3", mu=y_mean, sigma=y_std * 2)
        sigma = pm.HalfNormal("sigma", sigma=y_std)
        idx = np.arange(N, dtype="float64")
        w1 = pm.math.sigmoid(5 * (idx - tau_1))
        w2 = pm.math.sigmoid(5 * (idx - tau_2))
        mu = mu_1 * (1 - w1) + mu_2 * w1 * (1 - w2) + mu_3 * w2
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model

MODELS = {"null": build_null_model, "one_cp": build_one_cp_model, "two_cp": build_two_cp_model}

print("\n" + "=" * 60)
print("STEP 3/4: Model Comparison (LOO-IC & WAIC)")
print("=" * 60)

all_comparisons_loo = {}
t0_all = time.time()

for zone in ZONE_ORDER:
    print(f"\n  Model comparison for {zone}")
    zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
    y = zdf["annual_precip_mm"].values.astype("float64")

    traces = {}
    for model_name, builder in MODELS.items():
        print(f"    Fitting {model_name}...", end=" ", flush=True)
        t0 = time.time()
        mdl = builder(y)
        with mdl:
            tr = pm.sample(draws=2000, tune=1000, chains=4,
                           target_accept=0.95, random_seed=42,
                           return_inferencedata=True)
            pm.compute_log_likelihood(tr)
        traces[model_name] = tr
        safe_zone = zone.replace(" ", "_")
        tr.to_netcdf(str(TRACES_DIR / f"comparison_{safe_zone}_{model_name}.nc"))
        print(f"{time.time() - t0:.1f}s  [saved]")

    comp_loo = az.compare(traces, ic="loo")
    all_comparisons_loo[zone] = comp_loo
    print(f"\n  {zone} LOO:")
    print(comp_loo)

    latex = comp_loo.to_latex(float_format="%.1f")
    safe_zone = zone.replace(" ", "_")
    with open(DATA_DIR / f"comparison_loo_{safe_zone}.tex", "w") as f:
        f.write(latex)

    checkpoint(f"Comparison {zone} done")

print(f"\nStep 3 total: {time.time() - t0_all:.1f}s")
print("\nBest Model per Zone (LOO):")
for zone in ZONE_ORDER:
    best = all_comparisons_loo[zone].index[0]
    elpd = all_comparisons_loo[zone].iloc[0]["elpd_loo"]
    print(f"  {zone}: {best} (ELPD-LOO = {elpd:.1f})")


# ═════════════════════════════════════════════════════════════════════════
# STEP 4: Publication Figures
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 4/4: Generating Figures")
print("=" * 60)

# Fig 1: Study Area Map
fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
nigeria_lon = [2.7, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
               11.0, 12.0, 13.0, 14.0, 14.5, 14.0, 13.5, 13.0, 12.5,
               12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0, 8.5, 8.0,
               7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.7]
nigeria_lat = [6.5, 6.0, 6.3, 6.0, 6.0, 5.5, 5.0, 4.5, 4.2, 4.5,
               4.8, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
               12.0, 12.5, 13.0, 13.5, 13.8, 13.5, 13.5, 13.5, 13.5,
               13.8, 13.8, 13.8, 13.5, 13.5, 13.0, 12.5, 12.0, 11.5,
               10.0, 6.5]
ax.plot(nigeria_lon, nigeria_lat, "k-", linewidth=1.0)
ax.fill(nigeria_lon, nigeria_lat, color="#f5f5f5", zorder=0)
ax.axhspan(4.0, 7.5, color=ZONE_COLORS["Coastal"], alpha=0.15, zorder=1)
ax.axhspan(7.5, 10.5, color=ZONE_COLORS["Guinea Savanna"], alpha=0.15, zorder=1)
ax.axhspan(10.5, 14.0, color=ZONE_COLORS["Sahel"], alpha=0.15, zorder=1)
for st in get_all_stations():
    color = ZONE_COLORS[st["zone"]]
    ax.plot(st["lon"], st["lat"], "o", color=color, markersize=5,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.annotate(st["city"], (st["lon"], st["lat"]),
                xytext=(4, 4), textcoords="offset points", fontsize=6, zorder=5)
ax.text(8, 12.5, "Sahel", fontsize=7, ha="center", style="italic",
        color=ZONE_COLORS["Sahel"], fontweight="bold")
ax.text(8, 9.0, "Guinea Savanna", fontsize=7, ha="center", style="italic",
        color=ZONE_COLORS["Guinea Savanna"], fontweight="bold")
ax.text(8, 6.0, "Coastal/Rainforest", fontsize=7, ha="center",
        style="italic", color=ZONE_COLORS["Coastal"], fontweight="bold")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_xlim(2, 15)
ax.set_ylim(3.5, 14.5)
ax.set_aspect("equal")
ax.set_title("(a) Study Area and Station Locations")
fig.tight_layout()
savefig(fig, "fig1_study_area")
print("  Saved fig1_study_area")

# Fig 2: Time Series with Posterior Change-Points
trace = hier_trace
fig, axes = plt.subplots(3, 1, figsize=IEEE_DOUBLE_COL_TALL, sharex=True)
for ax, zone in zip(axes, ZONE_ORDER):
    zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
    years = zdf["year"].values
    precip = zdf["annual_precip_mm"].values
    color = ZONE_COLORS[zone]
    ax.bar(years, precip, color=color, alpha=0.4, width=0.8)
    tau_samples = trace["posterior"]["tau_year"].sel(zone=zone).values.flatten()
    ax2 = ax.twinx()
    ax2.hist(tau_samples, bins=50, density=True, color="gray", alpha=0.5, zorder=3)
    ax2.set_ylabel("P(change-point)", fontsize=6)
    ax2.tick_params(labelsize=6)
    mu_b = float(trace["posterior"]["mu_before"].sel(zone=zone).values.mean())
    mu_a = float(trace["posterior"]["mu_after"].sel(zone=zone).values.mean())
    tau_med = float(np.median(tau_samples))
    ax.axhline(mu_b, xmax=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(mu_a, xmin=0.5, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
    ax.axvline(tau_med, color="red", linewidth=1.0, linestyle="-", alpha=0.8)
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title(f"{zone} (change-point: {tau_med:.0f})")
axes[-1].set_xlabel("Year")
fig.suptitle("Annual Precipitation with Bayesian Change-Point Estimates", fontsize=10, y=1.01)
fig.tight_layout()
savefig(fig, "fig2_changepoint_timeseries")
print("  Saved fig2_changepoint_timeseries")

# Fig 3: Trace and Rank Plots
try:
    axes_arr = az.plot_trace(trace, var_names=["tau_year", "mu_before", "mu_after", "steepness"],
                             compact=True, figsize=(7.16, 8))
    fig = axes_arr.ravel()[0].get_figure()
    fig.tight_layout()
    savefig(fig, "fig3_trace_plots")
    print("  Saved fig3_trace_plots")
except Exception as e:
    print(f"  fig3_trace_plots skipped: {e}")

try:
    axes_arr = az.plot_rank(trace, var_names=["tau_year", "mu_before", "mu_after"], figsize=(7.16, 5))
    fig = axes_arr.ravel()[0].get_figure()
    fig.tight_layout()
    savefig(fig, "fig3b_rank_plots")
    print("  Saved fig3b_rank_plots")
except Exception as e:
    print(f"  fig3b_rank_plots skipped: {e}")

# Fig 4: Forest Plot
try:
    fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE_COL)
    az.plot_forest(trace, var_names=["tau_year"], combined=True, prob=0.94, ax=axes[0])
    axes[0].set_title("Change-Point Year (94% HDI)")
    axes[0].set_xlabel("Year")
    az.plot_forest(trace, var_names=["shift_magnitude"], combined=True, prob=0.94, ax=axes[1])
    axes[1].set_title("Shift Magnitude (94% HDI)")
    axes[1].set_xlabel("mm/year")
    fig.tight_layout()
    savefig(fig, "fig4_forest")
    print("  Saved fig4_forest")
except Exception as e:
    print(f"  fig4_forest skipped: {e}")

# Fig 5: Posterior Predictive Checks
try:
    fig, axes = plt.subplots(1, 3, figsize=IEEE_DOUBLE_COL)
    for ax, zone in zip(axes, ZONE_ORDER):
        obs_key = f"obs_{zone}"
        pp_groups = list(hier_ppc.children) if hasattr(hier_ppc, 'children') else (list(hier_ppc.groups()) if hasattr(hier_ppc, 'groups') else [])
        if "posterior_predictive" in pp_groups:
            pp_data = hier_ppc["posterior_predictive"]
            if obs_key in pp_data:
                ax.hist(pp_data[obs_key].values.flatten(), bins=30, density=True,
                        alpha=0.5, color="blue", label="Predicted")
                zdf = annual_zone[annual_zone["zone"] == zone]
                ax.hist(zdf["annual_precip_mm"].values, bins=15, density=True,
                        alpha=0.5, color="red", label="Observed")
                ax.legend(fontsize=5)
        ax.set_title(zone)
        ax.set_xlabel("Precip (mm)")
    fig.suptitle("Posterior Predictive Checks", fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig5_ppc")
    print("  Saved fig5_ppc")
except Exception as e:
    print(f"  fig5_ppc skipped: {e}")

# Fig 6: Model Comparison
try:
    fig, axes = plt.subplots(1, 3, figsize=IEEE_DOUBLE_COL)
    for ax, zone in zip(axes, ZONE_ORDER):
        safe_zone = zone.replace(" ", "_")
        traces_dict = {}
        for mname in ["null", "one_cp", "two_cp"]:
            tp = TRACES_DIR / f"comparison_{safe_zone}_{mname}.nc"
            if tp.exists():
                traces_dict[mname] = az.from_netcdf(str(tp))
        if traces_dict:
            comp = az.compare(traces_dict, ic="loo")
            az.plot_compare(comp, ax=ax)
        ax.set_title(zone, fontsize=8)
    fig.suptitle("Model Comparison (LOO-IC)", fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig6_model_comparison")
    print("  Saved fig6_model_comparison")
except Exception as e:
    print(f"  fig6_model_comparison skipped: {e}")

checkpoint("All figures done")


# ═════════════════════════════════════════════════════════════════════════
# Final summary and zip
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print(f"End time: {time.strftime('%H:%M:%S')}")
print(f"\nTraces in: {TRACES_DIR}")
for f in sorted(TRACES_DIR.glob("*.nc")):
    print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"\nFigures in: {FIGURES_DIR}")
for f in sorted(FIGURES_DIR.glob("fig*")):
    print(f"  {f.name}")
print(f"\nOutput archived to /content/mcmc_output.zip")
print("Download with: colab download /content/mcmc_output.zip")
