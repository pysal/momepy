import os
import gdal

dir_with_csvs = r"/Users/martin/Downloads"
os.chdir(dir_with_csvs)


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]
csvfiles = find_csv_filenames(dir_with_csvs)
for fn in csvfiles:
    vrt_fn = fn.replace(".csv", ".vrt")
    lyr_name = fn.replace('.csv', '')
    out_tif = fn.replace('.csv', '.tiff')
    with open(vrt_fn, 'w') as fn_vrt:
        fn_vrt.write('<OGRVRTDataSource>\n')
        fn_vrt.write('\t<OGRVRTLayer name="%s">\n' % lyr_name)
        fn_vrt.write('\t\t<SrcDataSource>%s</SrcDataSource>\n' % fn)
        fn_vrt.write('\t\t<GeometryType>wkbPoint</GeometryType>\n')
        fn_vrt.write('\t\t<GeometryField encoding="PointFromColumns" x="Lon" y="Lat" z="Ref"/>\n')
        fn_vrt.write('\t</OGRVRTLayer>\n')
        fn_vrt.write('</OGRVRTDataSource>\n')

    gdal_cmd = 'gdal_grid -a invdist:power=2.0:smoothing=1.0 -zfield "Ref" -of GTiff -ot Float64 -l %s %s %s' % (lyr_name, vrt_fn, out_tif)

output = gdal.Grid('outcome3.tif', 'name.vrt', algorithm='invdist:power=2.0:smoothing=1.0')
