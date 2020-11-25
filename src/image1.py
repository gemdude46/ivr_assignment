#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import math
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

def angle_between(u, v):
	''' Calculates the angle between two vectors in radians. '''
	return np.arccos(np.clip((u / np.linalg.norm(u)) @ (v / np.linalg.norm(v)), -1, 1))

class Line:
	''' Represents a line in 3D space. '''

	def __init__(self, p0, p1):
		''' Construct from two points. '''
		self.s = p0
		self.d = p1 - p0
		self.d = self.d / np.linalg.norm(self.d)
	
	def closest_points(self, other):
		'''Returns the closest point on this line to another, and the closest point on the other line to this one. '''
		w = self.s - other.s
		a = self.d @ self.d
		b = self.d @ other.d
		c = other.d @ other.d
		d = self.d @ w
		e = other.d @ w
		denom = a*c - b*b
		selflen = (b*e - c*d) / denom
		otherlen = (a*e - b*d) / denom
		return (self.s + self.d * selflen, other.s + other.d * otherlen)

	def distance_to(self, other):
		''' Returns the minimum distance to another line. '''
		a, b = self.closest_points(other)
		return np.linalg.norm(a - b)
	
	def get_average_closest_point(self, other):
		''' Returns the halfway point on the line segment that connects two lines at their closest aproach. '''
		a, b = self.closest_points(other)
		return (a + b) / 2

class ColorRange:
	''' Represents a range of colors used in target detection. '''

	def __init__(self, a, b):
		self.min = a
		self.max = b

class DetectionTarget:
	''' Represents a colored target to detect. '''

	def __init__(self, name, color_range, use_shape=False):
		self.name = name
		self.range = color_range
		self.use_shape = use_shape

detection_targets = (
	DetectionTarget('yellow', ColorRange((0,100,100), (10,150,150))),
	DetectionTarget('blue', ColorRange((100,0,0), (150,10,10))),
	DetectionTarget('green', ColorRange((0,100,0), (10,150,10))),
	DetectionTarget('red', ColorRange((0,0,100), (10,10,150))),
	DetectionTarget('target', ColorRange((0,50,100), (10,100,200)), True)
)

class CameraData:
	''' Stores data about a camera. '''

	def __init__(self, pos, facing, size):
		self.pos = pos
		self.facing = facing
		self.project_center = pos + facing
		self.size = size
	
	def line(self, img_pos):
		''' Casts a line from a point on the camera's image. '''
		right = np.cross(self.facing, np.array((0,0,1)))
		up = np.cross(right, self.facing)

		right = right / np.linalg.norm(right)
		up = up / np.linalg.norm(up)

		scaled = img_pos * self.size

		return Line(self.pos, self.project_center + right * scaled[0] + up * scaled[1])


camera1 = CameraData(np.array((18,0,0)), np.array((-1,0,0)), np.array((5/3, 5/3)))
camera2 = CameraData(np.array((0,-18,0)), np.array((0,1,0)), np.array((5/3, 5/3)))

class ObjectData2D:
	''' Data about an object in a single 2D view. '''

	def __init__(self, name, view_data, img_pos):
		self.name = name
		self.view_data = view_data
		self.img_pos = img_pos

		if img_pos is None:
			self.line = None
		else:
			self.line = view_data.camera_data.line(img_pos)
	
	def __bool__(self):
		return self.line is not None

	def __str__(self):
		return '{}: {}'.format(self.name, self.img_pos)

class ObjectData3D:
	''' Data about an object in 3D space, calculated from multiple views. '''

	def __init__(self, name, position):
		self.name = name
		self.position = position
	
	def __str__(self):
		return '{}: {}'.format(self.name, self.position)

class ViewData:
	''' Data from a single 2D viewpoint. '''

	def __init__(self, image, camera_data):
		self.image = image
		self.camera_data = camera_data

		self.detected_objects = dict()

		for detection_target in detection_targets:
			self.detect(detection_target)
	
	def detect(self, target):
		masked = cv2.inRange(self.image, target.range.min, target.range.max)

		if target.use_shape:
			detected_object = ObjectData2D(target.name, self, None)
			cv2.imshow('window1', masked)

		else:
			target_moments = cv2.moments(masked)
			if target_moments['m00'] < 1000:
				detected_object = ObjectData2D(target.name, self, None)
			else:
				detected_object = ObjectData2D(target.name, self, np.array((target_moments['m10'] / target_moments['m00'], target_moments['m01'] / target_moments['m00'])) / self.image.shape[:2] - np.array((0.5, 0.5)))
		
		self.detected_objects[target.name] = detected_object
	
	def all_lines(self):
		return (obj.line for obj in self.detected_objects.values() if obj)

	def __str__(self):
		return '\n'.join([str(o) for o in self.detected_objects.values()])

class SceneData:
	''' 3D data from multiple viewpoints. '''

	def __init__(self, x_view, y_view):
		self.x_view = x_view
		self.y_view = y_view

		self.detected_objects = dict()

		for detection_target in detection_targets:
			self.detect(detection_target)
	
	def detect(self, target):
		x_data = self.x_view.detected_objects[target.name]
		y_data = self.y_view.detected_objects[target.name]

		if not x_data and not y_data:
			detected_object = ObjectData3D(target.name, None)

		elif x_data and y_data:
			detected_object = ObjectData3D(target.name, x_data.line.get_average_closest_point(y_data.line))

		elif (x_data and not y_data) or (y_data and not x_data):
			known_line = (x_data or y_data).line
			other_lines = (x_data and y_data).view_data.all_lines()

			closest_line = None

			for line in other_lines:
				if closest_line is None or known_line.distance_to(closest_line) > known_line.distance_to(line):
					closest_line = line

			detected_object = ObjectData3D(target.name, known_line.get_average_closest_point(closest_line))

		self.detected_objects[target.name] = detected_object
	
	def __str__(self):
		return '\n'.join([str(self.x_view), str(self.y_view), *(str(o) for o in self.detected_objects.values())])

class ImageConverter:

	# Defines publisher and subscriber
	def __init__(self):
		# initialize the node named image_processing
		rospy.init_node('image_processing', anonymous=True)
		# initialize a publisher to send images from camera1 to a topic named image_topic1
		self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
		# initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
		self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
		# and the other camera
		self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
		# initialize the bridge between openCV and ROS
		self.bridge = CvBridge()
		# output for joint estimates
		self.est_2_pub = rospy.Publisher('/vision_estimates/joint2', Float64, queue_size=10)
		self.est_3_pub = rospy.Publisher('/vision_estimates/joint3', Float64, queue_size=10)
		self.est_4_pub = rospy.Publisher('/vision_estimates/joint4', Float64, queue_size=10)

		self.cv_image1 = self.cv_image2 = None



	# Recieve data from camera 1, process it, and publish
	def callback1(self,data):
		# Recieve the image
		try:
			self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		
		# Uncomment if you want to save the image
		#cv2.imwrite('image_copy.png', cv_image)

		#im1=cv2.imshow('window1', self.cv_image1)
		cv2.waitKey(1)
		# Publish the results
		try: 
			self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
		except CvBridgeError as e:
			print(e)

		if self.cv_image1 is not None and self.cv_image2 is not None:
			view1 = ViewData(self.cv_image1, camera1)
			view2 = ViewData(self.cv_image2, camera2)
			scene = SceneData(view1, view2)

			yellow = scene.detected_objects['yellow'].position
			blue = scene.detected_objects['blue'].position
			green = scene.detected_objects['green'].position
			red = scene.detected_objects['red'].position

			est_joint_2 = math.pi + math.atan2(*(green - blue)[1:])
			est_joint_3 = math.pi - math.atan2(*(green - blue)[::2])
			est_joint_4 = angle_between(green - blue, red - green)

			if est_joint_2 > math.pi: est_joint_2 -= 2 * math.pi
			if est_joint_3 > math.pi: est_joint_3 -= 2 * math.pi

			self.est_2_pub.publish(est_joint_2)
			self.est_3_pub.publish(est_joint_3)
			self.est_4_pub.publish(est_joint_4)

	
	def callback2(self, data):
		try:
			self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
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


