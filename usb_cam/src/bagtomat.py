import rosbag

bag = rosbag.Bag('Clear_2_2023-04-12-14-32-16.bag')
for topic, msg, t in bag.read_messages():
    print(topic)
