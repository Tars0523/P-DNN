# P-DNN

<br/>Tracking 1 person (Id 0), Using DNN(Darkent). Bounding Box height and Bounding Box centroid is using as P-control.

<br/> Input : Image from computer camera(VideoCaputre(0)) or USB Camera... extra

<br/> Subscribe : No subscribe Topic

<br/> Publish 1 : </cmd_vel> , command of linear velocity and angluar velocity ( coordinate is Z is up(sky not ground) and X is forward of car)

<br/> Publish 2 : </camera> , Image data using for Perecption, in this code there is no perception term, you can add other node for perception


```c
Ubuntu 20.04
ROS Noetic
>> cd ~/catkin_ws/src
>> git clone ~
>> cd ..
>> catkin_make
>> source ./devel/setup.bash
>> rosrun usb_cam opencv-capture.py
```
