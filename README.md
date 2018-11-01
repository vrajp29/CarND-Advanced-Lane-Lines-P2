## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



## Udacity Self-Driving Car Engineer Nanodegree

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

![mp4](result.mp4)
![png](output_7_1.png)

## Correct the Distortion:
    
Lets remove distortion from the images using camera calibbration matrix and distortion coefficients using chessboard

In this exercise, you'll use the OpenCV functions findChessboardCorners() and drawChessboardCorners() to automatically find and draw corners in an image of a chessboard pattern, 

![png](output_3_0.png)

![png](output_3_1.png)

## Remove distortion from images
There are two main steps to this process: use chessboard images to obtain image points and object points, and then use the OpenCV functions cv2.calibrateCamera() and cv2.undistort() to remove distortion from highway driving images

If you look around the edges of both original and undistorted images, you will observe that distortion is removed from original images


[png](output_4_0.png)

![png](output_4_1.png)



## Now lest perfrom the perspective transformation:

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective transform you’ll be most interested in is a bird’s-eye view transform that let’s us view a lane from above; this will be useful for calculating the lane curvature later on. 

Here we apply a perspective transform, choosing four source points manually

src = np.float32([[490, 482],[810, 482],
                 [1300, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0], 

I used cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(src, dst)

And the I used cv2.warpPerspective() to apply M and warp your image to a top-down view
warped = cv2.warpPerspective(undist, M, img_size)

![png](output_5_0.png)

![png](output_5_1.png)



## Lets see what happend when we apply Binary Thresholds

Here we try to apply different color space and identify which provides better lane lines and ignore the onces which give some noise

Used S channel from HLS color space

Used L channel from LUV color space

Used B channel from Lab color space

Create combination binary threshold which best highlights lane lines

![png](output_6_3.png)


## After finding combination of the binary threshold which gives the best result, lets try to Fit a polynomial to the lane line and find vehicle position and Radius of Curvature

### I used following steps to achieve this:
#### Line Finding Method: Peaks in a Histogram and finding the non-zero pixels around peaks using numpy.nonzero()
#### Fit the polynomial to the lane numpy.polyfit()
#### For each lane line measure radius of curvature       
#### Calculate the position of the vehicle
```
    center = abs(640 - ((rightx_int+leftx_int)/2))
    
```
I implemented this step in lines`fill_lane_lines()`.  Here is an example of my result on a test image:

![png](output_7_0.png)

![png](output_7_1.png)


Now we have estimation of radius of curvature and position of the vehicle 


## Lets create function process_video() which will process video frame by frame:
We cobine all the above code blocks with all averaging and fallback concepts. *Process_video()* is the final pipeline: Cell #108, `P2 - Advance Lane Finding Project.ipynd`

Here is [link to my video result](https://www.youtube.com/watch?v=pbDZBV0_m4M) You will be redirected to YouTube. 



## discussing problems / Issues I faced in implementation of this project

#### Gradient & Color Thresholding
* Time consuming experiment with gradient and color channnel thresholding.  
* In challenge video when car drives under the bridge it was difficult to find the lane lines
* Detecting lane lines in harder challenge was extremely hard as there were constant curves, fadeness, Shadow 

#### Bad Frames
The challenge video has a section where the car goes underneath a tunnel and no lanes are detected
To tackle this I had to resort to averaging over the well detected frames

Points of failure & Areas of Improvement
The pipeline seems to fail for the harder challenge video. 

#### Improvement:
I would take a smaller section to take the transform to take a better perspective transform
I would average over a smaller number of frames.

Additional option Challenge video: The pipeline did prity good job in detecting lane lines even though it having some hard time when car passed under a bridge or throught tunnel   
Here's the [link to my video Challenge output video](https://www.youtube.com/watch?v=xzmUQdYRtc0) You will be redirected to YouTube. 




