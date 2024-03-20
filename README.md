# Point Cloud Converter

This basic command line interface combines several open source point cloud processing packages into 1 unified interface.
It supports the following formats:
- `.las` (and compressed `.laz`) reading & writing using [laspy](https://github.com/laspy/laspy)
- `.ply` (both text and binary) reading & writing using [plyfile](https://github.com/dranjan/python-plyfile)
- `.e57` reading & writing using [pye57](https://github.com/davidcaron/pye57)
- `.pts` reading & writing (Warning: ASCII mode only)
- `.pcd` reading & writing using [pypcd4](https://github.com/MapIV/pypcd4)
- `potree` writing only using [potreeconverter](https://github.com/potree/PotreeConverter) (executable included in build)

## How to use
Execute using the command line with the following arguments:
- A file path to an origin point cloud file. Must be an absolute full path. (Required positional argument) 
- `-destination` (aliasses `-dest` and `-d`) : A destination file or folder path using  followed by the desired path.
If not provided, the destination directory will be assumed to be the same directory as the origin. The filename will also be the same (except for the extension of course).
- `-extension` (aliasses `-ext` and `e`) : A destination extension using . Ignored if destination path is already a valid destination point cloud file path. Defaults to `.las` if not provided.
- `-unsafe` (alias `-u`) : to allow the application to overwrite folders and files. By default, the program does *not* overwrite files or directories.
- `-verbose` (alias `-v`) : Specify a verbosity level, default is `2`.
  - `0` = No information is logged to the console.
  - `1` = Only basic information is printed to the console (e.g "Reading file complete.")
  - `2` = Prints nicely formatted messages and progress bars (while writing only).
  - `3` = Prints progress as a percentage to the console. Useful if you want to use the executable as is and read the progress from the standard output stream.

## Examples:
1. Providing only an origin. Extension defaults to `.las`.  
`C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57`  
Converted file will be at `C:\Users\Someone\pointcloud-dummy.las`

2. Providing a destination file. Convert the `pointcloud-dummy.e57` file to a .las file.   
`C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57 -destination C:\Users\Someone\targetpointcloud.las`  
Converted file will be at `C:\Users\Someone\targetpointcloud.las`  

3. Providing a destination directory. Extension defaults to `.las`  
`C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57 -dest C:\Users\Someone\Downloads`  
Converted file will be at `C:\Users\Someone\Downloads\pointcloud-dummy.las`  

4. Providing an extension. Destination directory defaults to origin parent directory.  
`C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57 -ext .pcd`  
Converted file will be at `C:\Users\Someone\pointcloud-dummy.pcd`  

5. Providing both extension and destination directory.  
`C:\Users\Someone\pointcloudconverter.exe C:\Users\Someone\pointcloud-dummy.e57 -dest C:\Users\Someone\Downloads -ext .pts`  
Converted file will be at `C:\Users\Someone\Downloads\pointcloud-dummy.pts`  
