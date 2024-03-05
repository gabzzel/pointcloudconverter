# Point Cloud Converter

This basic command line interface combines several open source point cloud processing packages into 1 unified interface.
It supports the following formats:
- `.las` (and compressed `.laz`) reading & writing using [laspy](https://github.com/laspy/laspy)
- `.ply` (both text and binary) reading & writing using [plyfile](https://github.com/dranjan/python-plyfile)
- `.e57` reading & writing using [pye57](https://github.com/davidcaron/pye57)
- `.pts` reading & writing
- `.pcd` reading & writing using [pypcd4](https://github.com/MapIV/pypcd4)
- `potree` writing only using [potreeconverter](https://github.com/potree/PotreeConverter) (executable included in build)

## How to use
Execute using the command line with the following arguments:
- REQUIRED: A file path to an origin point cloud file (must be an absolute full path.)
- OPTIONAL: A destination file or folder path using `-destination` , `-dest` or `-d` followed by the desired path.
If not provided, the destination will be assumed to be the same folder as the origin. The filename will also be the same (except for the extension of course).
- OPTIONAL: A destination extension using `-extension`, `-ext` or `e`. Ignored if destination path is already a valid destination point cloud file path. Defaults to `.las` if not provided.
- OPTIONAL: `-unsafe` or `-u` to allow the application to overwrite folders and files. By default, the program does *not* overwrite files.

## Examples:
1. Providing only an origin. Extension defaults to `.las`.
Command: `C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57`
Converted file will be at `C:\Users\Someone\pointcloud-dummy.las`

2. Providing a destination file. Convert the `pointcloud-dummy.e57` file to a .las file.
Command: `C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57 -destination C:\Users\Someone\targetpointcloud.las`
Converted file will be at `C:\Users\Someone\targetpointcloud.las`

3. Providing a destination directory. Extension defaults to `.las`
Command: `C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57 -dest C:\Users\Someone\Downloads`
Converted file will be at `C:\Users\Someone\Downloads\pointcloud-dummy.las`

4. Providing an extension. Destination directory defaults to origin parent directory.
Command: `C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57 -ext .pcd`
Converted file will be at `C:\Users\Someone\pointcloud-dummy.pcd`

5. Providing both extension and destination directory.
Command: `C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57 -dest C:\Users\Someone\Downloads -ext .pts`
Converted file will be at `C:\Users\Someone\Downloads\pointcloud-dummy.pts`
