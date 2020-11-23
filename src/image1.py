#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

class ColorRange:
	def __init__(self, a, b):
		self.min = a
		self.max = b

class DetectionTarget:
	def __init__(self, name, color_range):
		self.name = name
		self.range = color_range

detection_targets = (
	DetectionTarget('yellow', ColorRange((), ()))
)

class ObjectData2D:
	def __init__(self, view_data, img_pos):
		self.view_data = view_data
		self.img_pos = img_pos

		if img_pos is None:
			self.line = None

		else:
			pass
	
	def __bool__(self):
		return self.line is not None

class ObjectData3D:
	def __init__(self, position):
		self.position = position

class ViewData:
	def __init__(self, image):
		self.image = image

		self.detected_objects = dict()

		for detection_target in detection_targets:
			self.detect(detection_target)
	
	def detect(self, target):
		target_moments = cv2.moments(cv2.inRange(self.image, target.range.min, target.range.max))
		detected_object = ObjectData2D(self, np.array((target_moments['m01'] / target_moments['m00'], target_moments['m10'] / target_moments['m00'])))
		self.detected_objects[target.name] = detected_object
	
	def all_lines(self):
		return obj.line for obj in self.detected_objects.values() if obj

class SceneData:
	def __init__(self, x_view, y_view):
		self.x_view = x_view
		self.y_view = y_view

		self.detected_objects = dict()

		for detection_target in detection_targets:
			self.detect(detection_target)
	
	def detect(self, target):
		x_data = self.x_view.detected_objects[target.name]
		y_data = self.x_view.detected_objects[target.name]

		if not x_data and not y_data:
			detected_object = ObjectData3D(None)

		elif x_data and y_data:
			detected_object = ObjectData3D(x_data.line.get_average_closest_point(y_data.line))

		elif (x_data and not y_data) or (y_data and not x_data):
			known_line = (x_data or y_data).line
			other_lines = (x_data and y_data).view_data.all_lines()

			closest_line = None

			for line in other_lines:
				if closest_line is None or known_line.distance_to(closest_line) > known_line.distance_to(line):
					closest_line = line

			detected_object = ObjectData3D(known_line.get_average_closest_point(closest_line))

		return detected_object

class ImageConverter:

	# Defines publisher and subscriber
	def __init__(self):
		# initialize the node named image_processing
		rospy.init_node('image_processing', anonymous=True)
		# initialize a publisher to send images from camera1 to a topic named image_topic1
		self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
		# initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
		self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
		# initialize the bridge between openCV and ROS
		self.bridge = CvBridge()
		



	# Recieve data from camera 1, process it, and publish
	def callback1(self,data):
		# Recieve the image
		try:
			self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		
		# Uncomment if you want to save the image
		#cv2.imwrite('image_copy.png', cv_image)

		im1=cv2.imshow('window1', self.cv_image1)
		cv2.waitKey(1)
		# Publish the results
		try: 
			self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
		except CvBridgeError as e:
			print(e)

# call the class
def main(args):
	ic = ImageConverter()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
		main(sys.argv)


