from .windows.download_tile_ui import DownloadThread
from .utils.raster_map import merge_tiles_bbox

def main(n,s,e,w,folder,zoom_level):
    # extent = {
    #         'n': 35.70,
    #         's': 35.60,
    #         'e': 52.00,
    #         'w': 51.90,
    #     }
    extent = {
        'n':n,
        's':s,
        'e':e,
        'w':w
    }
    print("extent['n'] ",extent['n'])
    print("extent ",extent)
    zoom = (zoom_level,zoom_level)#(self.min_z.value(), self.max_z.value())
    #folder = r'C:\Users\USER\Downloads\GUI Map\Map-App-main\data'
    fmt = "jpg"
    quality = 85
    
    DlThread = DownloadThread(extent, zoom, folder, fmt, quality)
    DlThread.run()


    zoom = zoom_level
    output_raster = folder + "/geotif.tif"
    format = "jpg"

    # Define your bounding box (example coordinates)

    merge_tiles_bbox(
        tile_folder=folder,
        output_path=output_raster,
        zoom=zoom,
        north_lat=extent['n'],
        south_lat=extent['s'],
        west_lon=extent['w'],
        east_lon=extent['e'],
        tile_size=256,
        format=format,
        compress_type='jpeg',
        jpeg_quality=85
    )
    