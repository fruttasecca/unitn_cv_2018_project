# unitn_cv_2018_project
Hasty implementation of a top view people tracker for the computer vision course in unitn.

####Stuff in this reporistory:
- input video, video.mp4
- ouput video, result.mp4
- groung truth files 
- code (main.py,  feature_swarm.py, util.py)
- report on this project

####Requirements:
- numpy
- opencv
- the Rtree python package

In main.py you will obviousy find the main loop, feature_swarm
contains the tracking algorithm and util is where I have stuffed all the utility
stuff to avoid overly cluttering the code.

__To run the project just run main.py with python3, no arguments needed__;
it will output a file named output.mp4.

This is not a long term project or anything, do not expect tests or whatever.
