#Final Year Project Code by Tega Orogun C16518763 DT228/4


import cv2
import numpy as np
import os
import math


# I was foing to use this to track the balls movement but unfortunately couldn't see the ball when it was hit 
# def calculateDistance(point):  
# 	 print(point,"point passed")
# 	 k,l = point
# 	 k = int(k)
# 	 l = int(l)
# 	 dist = math.sqrt((k - (k-10))**2 + (l - (l-10))**2)  
# 	 print (dist)
# 	 #return dist   

# def inrectangle(point):
# 	#make distance formula 



try: 
	  
	# creating snapshot images folder 
	if not os.path.exists('images'): 
		os.makedirs('images') 
  
# if not created then raise error 
except OSError: 
	print ('Error: Creating directory for images')


cap = cv2.VideoCapture('golfarc2.mp4')

template_golf = cv2.imread('template3.jpg', cv2.IMREAD_GRAYSCALE)
w, h = template_golf.shape[::-1]
# print (template_golf.shape())

currentframe = 0 

colour_tracker = 0 

club_colour_tracker = 0

point_selected = 0
point = ()
previous_coords = np.array([[]])
allPointsSelected = False

hand_point =()
previous_coords2 = np.array([[]])

head_point =()
previous_coords3 = np.array([[]])

ball_point =()
previous_coords4 = np.array([[]])


club_selected = False

hand_counter = 0
head_counter = 0
ball_counter = 0

click_count = 0


#for the hand
upswing_draw_x = []

upswing_draw_y = []

upswing_drawin_x = []

upswing_drawin_y = []

downswing_draw_x = []

downswing_draw_y = []

downswing_drawin_x = []

downswing_drawin_y = []

#for the club

club_upswing_draw_x = []

club_upswing_draw_y = []

club_upswing_drawin_x = []

club_upswing_drawin_y = []

club_downswing_draw_x = []

club_downswing_draw_y = []

club_downswing_drawin_x = []

club_downswing_drawin_y = []

#ball tracking 
ball_track = []

ball_track_x = []

ball_track_y = []


upswing_draw = []

club_upswing = []
downswing_draw = []

head_track = []

#For comparisons
hand_comparison_draw = []

club_comparison_draw = []

head_comparison_draw = []


print("Click on points to track in the order they appear. Hands, Club, Head and the Ball.")



choice = input("Do you want to compare another person's swing?")

if choice == "No" or "no" :
	print("Okay.")
	
# while choice != "Yes" or choice != "yes" or choice != "No" or choice != "no":
# 	print("Wrong choice")
# 	choice = input("Do you want to compare another person's swing?")








# Create old frame
ret, frame1 = cap.read()

frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)







# Lucas kanade params
#This is used to in the optical flow method for detection 
lk_params = dict(winSize = (15, 15),
				 maxLevel = 4,
				 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

lk_params2 = dict(winSize = (15, 15),
				 maxLevel = 4,
				 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

lk_params3 = dict(winSize = (15, 15),
				 maxLevel = 4,
				 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

lk_params4 = dict(winSize = (15, 15),
				 maxLevel = 4,
				 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))











def no_of_clicks():
	global choice

	if click_count > 3:
		name = './images/frame' + '.jpg'

		cv2.imwrite(name, frame1)
		startofswing = cv2.imread("./images/frame.jpg")
		cv2.imshow("start of swing", startofswing)

		main_tracker()
	else:
		print("choose points")

	if choice == "Yes":
		writing_comparison()
	

def writing_comparison():
	#all these text files already exist from previous tasks I've done
	with open('hand_upswing.txt', "r") as fp:
	    for i in fp.readlines():
	        tmp = i.split(",")
	        try:
	            hand_comparison_draw.append((int(tmp[0]), int(tmp[1])))
	            #result.append((eval(tmp[0]), eval(tmp[1])))
	        except:pass

	for i in hand_comparison_draw:
		cv2.circle(frame1, tuple(i), 10, (55, 123, 55), -1)


	with open('club_upswing.txt', "r") as fp:
	    for i in fp.readlines():
	        tmp = i.split(",")
	        try:
	            club_comparison_draw.append((int(tmp[0]), int(tmp[1])))
	            #result.append((eval(tmp[0]), eval(tmp[1])))
	        except:pass

	for i in club_comparison_draw:
		cv2.circle(frame1, tuple(i), 10, (155, 123, 255), -1)

	with open('head_track.txt', "r") as fp:
	    for i in fp.readlines():
	        tmp = i.split(",")
	        try:
	            head_comparison_draw.append((int(tmp[0]), int(tmp[1])))
	            #result.append((eval(tmp[0]), eval(tmp[1])))
	        except:pass

	for i in head_comparison_draw:
		cv2.circle(frame1, tuple(i), 10, (55, 23, 255), -1)




#This is the main tracking function 
def main_tracker():
	global click_count, colour_tracker, frame1_gray, previous_coords, previous_coords2, previous_coords3, previous_coords4, club_colour_tracker
	print(ball_point, "checking")
	while cap.isOpened():
		print ("starting while loop")

		ret, frame2 = cap.read()

		#cv2.CV_CAP_PROP_POS_FRAMES
		# cap.set(1, currentframe)

		# currentframe = currentframe + 5
		
		

		frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


		
		
		if (point_selected == 1 or point_selected > 1):

			

			cv2.circle(frame2, point, 5, (0, 0, 255), 2)
				
			new_coords, status, error = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, previous_coords, None, **lk_params)

				


			previous_coords = new_coords
				
			x, y = new_coords.ravel()

			x = int(x)

			y = int(y)

			upswing_draw_x.append(x)

			upswing_drawin_x.append(x)

			downswing_draw_x.append(x)

			downswing_drawin_x.append(x)

			upswing_draw_y.append(y)

			if len(upswing_draw_x) > 2:

				for i in range(2, len(upswing_draw_x)):
				# print(upswing_draw_x[i-1], i, upswing_draw_x[i])
					
				# print("\n")
					if upswing_draw_x[i] <= upswing_draw_x[i-1] and upswing_draw_x[i] <= upswing_draw_x[i-2] and colour_tracker < 1:
						cv2.circle(frame2, (upswing_draw_x[i],upswing_draw_y[i]), 10 ,(255,0,0), -1)
						
					elif upswing_drawin_x[i] >= upswing_drawin_x[i-1] and upswing_drawin_x[i] >= upswing_drawin_x[i-2] and colour_tracker < 2:
						
						if colour_tracker < 1:
							print("changing to upswing in")
							colour_tracker = colour_tracker + 1
							# name = './images/frame' + '.jpg'
							# cv2.imwrite(name, frame2)	
							
						cv2.circle(frame2, (upswing_draw_x[i],upswing_draw_y[i]), 10 ,(255,0,0), -1)
						
					elif downswing_draw_x[i] <= downswing_draw_x[i-1] and  downswing_draw_x[i] <= downswing_draw_x[i-2] and colour_tracker < 3:
						
						if colour_tracker < 2:
							print("changing to downswing out")
							colour_tracker = colour_tracker + 1
							
						cv2.circle(frame2, (upswing_draw_x[i],upswing_draw_y[i]), 10 ,(0,255,0), -1)
						
					elif downswing_drawin_x[i] >= downswing_drawin_x[i-1] and downswing_drawin_x[i] >= downswing_drawin_x[i-2] and colour_tracker < 4:
						if colour_tracker < 3:
							colour_tracker = colour_tracker + 1
							name2 = './images/frame2' + '.jpg'
							topofswing = cv2.imread("./images/frame2.jpg")
							cv2.imshow("topofswing", topofswing)
							cv2.imwrite(name2, frame2)

						print("changing to downswing_drawin_x in")
						cv2.circle(frame2, (upswing_draw_x[i],upswing_draw_y[i]), 10 ,(0,255,0), -1)


					# else:
					# 	cv2.circle(frame2, (upswing_draw_x[i],upswing_draw_y[i]), 10 ,(0,0,255), -1)

				

			# upswing_draw = new_coords.ravel()


			upswing_draw.append((x,y))

			# with open('hand_upswing.txt','w') as f:
			# 	f.write( '\n'.join(' '.join(str(x) for x in tu) for tu in upswing_draw) )

			with open('hand_upswing2.txt', 'w') as f:
				for e in upswing_draw:
					f.write('%s\n' % ','.join(str(n) for n in e))

				



				




		if (point_selected == 2 or point_selected > 2):

			

			new_coords2, status2, error2 = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, previous_coords2, None, **lk_params2)

			previous_coords2 = new_coords2

			a, b = new_coords2.ravel()

			a = int(a)

			b = int(b)

			cv2.circle(frame2, (a,b), 5, (255, 0, 255), 2)

			club_upswing

			club_upswing_draw_x.append(a)

			club_upswing_draw_y.append(b)

			club_upswing_drawin_x.append(a)

			club_downswing_draw_x.append(a)

			club_downswing_drawin_x.append(a)


			if len(club_upswing_draw_x) > 2:

				for i in range(2, len(club_upswing_draw_x)):
				
					if club_upswing_draw_x[i] <= club_upswing_draw_x[i-1] and club_upswing_draw_x[i] <= club_upswing_draw_x[i-2] and club_colour_tracker < 1:
						cv2.circle(frame2, (club_upswing_draw_x[i],club_upswing_draw_y[i]), 10 ,(255,0,0), -1)
						
					elif club_upswing_drawin_x[i] >= club_upswing_drawin_x[i-1] and club_upswing_drawin_x[i] >= club_upswing_drawin_x[i-2] and club_colour_tracker < 2:
						
						if club_colour_tracker < 1:
							print("changing to upswing in")
							club_colour_tracker = club_colour_tracker + 1
							
						cv2.circle(frame2, (club_upswing_draw_x[i],club_upswing_draw_y[i]), 10 ,(255,0,0), -1)
						
					elif club_downswing_draw_x[i] <= club_downswing_draw_x[i-1] and  club_downswing_draw_x[i] <= club_downswing_draw_x[i-2] and club_colour_tracker < 3:
						
						if club_colour_tracker < 2:
							print("changing to downswing out")
							club_colour_tracker = club_colour_tracker + 1
							
						cv2.circle(frame2, (club_upswing_draw_x[i],club_upswing_draw_y[i]), 10 ,(0,255,0), -1)
						
					elif club_downswing_drawin_x[i] >= club_downswing_drawin_x[i-1] and club_downswing_drawin_x[i] >= club_downswing_drawin_x[i-2] and club_colour_tracker < 4:
						print("changing to downswing_drawin_x in")
						cv2.circle(frame2, (club_upswing_draw_x[i],club_upswing_draw_y[i]), 10 ,(0,255,0), -1)


			club_upswing.append((a,b))


			with open('club_upswing2.txt', 'w') as f:
				for e in club_upswing:
					f.write('%s\n' % ','.join(str(n) for n in e))


		

		
		if (point_selected == 3 or point_selected > 3):

			pass

			cv2.circle(frame2, head_point, 5, (255, 0, 0), 2)

			new_coords3, status3, error3 = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, previous_coords3, None, **lk_params3)

			

			previous_coords3 = new_coords3

			c, d = new_coords3.ravel()

			c = int(c)

			d = int(d)


			head_track.append((c,d))

			for i in head_track:
				cv2.circle(frame2, tuple(i), 10, (0, 0, 255), -1)


			with open('head_track2.txt', 'w') as f:
				for e in head_track:
					f.write('%s\n' % ','.join(str(n) for n in e))





		
		if (point_selected == 4 or point_selected > 4):

			pass

			frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
			



			#maks2 that kinda works
			lower_colour = np.array([180,180,180])
			upper_colour = np.array([255,255,255])

			# mask = cv2.inRange(frame2_hsv, lower_bound, upper_bound)

			# mask2 that kinda works 
			mask2 = cv2.inRange(frame2, lower_colour, upper_colour)

		# 	pass

			

			#cv2.circle(frame2, ball_point, 5, (255, 255, 255), -1)

			new_coords4, status4, error4 = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, previous_coords4, None, **lk_params4)

			previous_coords4 = new_coords4

			e, f = new_coords4.ravel()

			e = int(e)

			f = int(f)

			black = np.zeros((frame2.shape[0], frame2.shape[1], 3), np.uint8)

			black1 = cv2.rectangle(black,(e-20,f-20),(e+20,f+20),(255, 255, 255), -1)

			gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)

			_, b_mask = cv2.threshold(gray,127,255, 0)

			fin = cv2.bitwise_and(frame2_gray,frame2_gray, mask = b_mask)

			contoursblack, hierarchy = cv2.findContours(fin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			cv2.drawContours(frame2, contoursblack, -1, (0,255,0), 3)






			#mask2 that kinda works 
			kernel = np.ones((3, 3), np.uint8)

			# mask2 = cv2.erode(mask2, kernel, iterations = 1)

			# mask2 = cv2.erode(mask2, kernel, iterations = 1)

			closing = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)


			contours = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
					cv2.CHAIN_APPROX_SIMPLE)[0]
			contours.sort(key=lambda x:cv2.boundingRect(x)[0])

			array = []
			ii = 1
			
			for c in contours:
				# print(c, "whats c")
				(q,t),r = cv2.minEnclosingCircle(c)
				center = (int(q),int(t))
				q = int(q)
				t = int(f)
				r = int(r)
				if r >= 3 and r<=5 and q >= e-20 and q <= e+20 and t >= f-20 and t <= f + 20:
					cv2.circle(frame2,center,r,(0,255,0),2)
					array.append(center)
					











			_, threshed_vid = cv2.threshold(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
					200, 255, cv2.THRESH_BINARY)

			contours, hier = cv2.findContours(threshed_vid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		
			cv2.imshow("mask2", mask2)




			# lower_bound = np.array([0,0,10])
			# upper_bound = np.array([255,255,195])

			# image = frame2

			# mask = cv2.inRange(frame2, lower_bound, upper_bound)

			# mask = cv2.adaptiveThreshold(image_ori,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
			#             cv2.THRESH_BINARY_INV,33,2)

			

			# #Use erosion and dilation combination to eliminate false positives. 
			# #In this case the text Q0X could be identified as circles but it is not.
			# mask = cv2.erode(mask, kernel, iterations=6)
			# mask = cv2.dilate(mask, kernel, iterations=3)

			


			
			# cv2.imshow("preprocessed", mask)
			# cv2.imshow("frame", threshed_vid)

			

		# 	# for c in contours:
		# 	# 	# get the bounding rect
		# 	# 	t, l, w, h = cv2.boundingRect(c)

		# 	# 	t = int(e) 

		# 	# 	l = int(f)
		# 	# 	# draw a green rectangle to visualize the bounding rect
		# 	# 	cv2.rectangle(frame2, (t, l), (t+w, l+h), (0, 255, 0), 2)

		# 	print(new_coords4)


			ball_track.append((e,f))

			# golfball_circle = cv2.HoughCircles(threshed_vid, cv2.HOUGH_GRADIENT, 1, 80, param1 = 50, param2 = 18, minRadius = 0, maxRadius = 50)

			# print(len(golfball_circle), "hello4")
			# print(golfball_circle)

			# if golfball_circle is not None:
			# 	for (e, f, r) in golfball_circle[0]:
			
			# 		cv2.circle(frame2, (e, f), r, (0, 255, 0), 4)
			# 	# cv2.rectangle(frame2, (e - 5, f - 5), (e + 5, f + 5), (0, 128, 255), -1)

		# 	for i in ball_track:
		# 		cv2.circle(frame2, tuple(i), 10, (0, 0, 255), -1)

				
		cv2.imshow("Frame", frame2)

		frame1_gray = frame2_gray.copy()

		key = cv2.waitKey(1)

		# if click_count < 4:
		# 	cv2.waitKey()

		
		
		if key == 27:
			break

	# cap.release()
	# cv2.destroyAllWindows()


def mouse_click(event, x, y, flags, params):
	
	global click_count, point, hand_point, head_point, ball_point, point_selected, previous_coords, allPointsSelected, previous_coords2, previous_coords3, previous_coords4, club_selected, hand_counter, head_counter, ball_counter 
	
	if event == cv2.EVENT_LBUTTONDOWN:
		point_selected += 1
		hand_counter += 1
		head_counter += 1
		ball_counter += 1

		click_count = click_count + 1

		

		if not point and point_selected > 0:

			print("in point_selected")

			point = (x, y)
			
			previous_coords = np.array([[x, y]], dtype=np.float32)

			

			

		
		elif not hand_point and hand_counter > 1:
			
			print("in hand_counter")

			hand_point = (x, y)
			# point_selected += 1
			previous_coords2 = np.array([[x, y]], dtype=np.float32)

			# name2 = './images/frame2' + '.jpg'

			# cv2.imwrite(name2, frame2)

		

		elif not head_point and head_counter > 2:
			print("in head_counter")
			head_point = (x, y)
			# point_selected += 1
			previous_coords3 = np.array([[x, y]], dtype=np.float32)

			name3 = './images/frame3' + '.jpg'

			#cv2.imwrite(name3, frame2)

		

		elif not ball_point and ball_counter > 3:
			print("setting ball point", x, y)
			ball_point = (x, y)
			# point_selected += 1
			previous_coords4 = np.array([[x, y]], dtype=np.float32)

			#calculateDistance(ball_point)

			name4 = './images/frame4' + '.jpg'

			# cv2.imwrite(name4, frame2)

		else:
			print(ball_point, ball_counter, "everything false")

		no_of_clicks()

		
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_click)





cv2.imshow('Frame', frame1)
if click_count < 4:
	cv2.waitKey()

	


