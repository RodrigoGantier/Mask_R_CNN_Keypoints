# Mask_R_CNN_Keypoints
The original code is from "https://github.com/matterport/Mask_RCNN" on Python 3, Keras, and TensorFlow.
The code recovers the work of "https://arxiv.org/abs/1703.06870" but does not include the of keypoints.  
So the work that ralice is, modifying the code to perform the detection of keypoints

### segmented image with the original code
<img src="pictures/segmented_image.png" width="400">

This image is one of the first tests, there is no mean average presition mAP
The resunts are far from the original paper

### Test Picture for keypoints
<img src="pictures/test_1.png" width="600">

### Network output
<img src="pictures/output_layer.png" width="600">

This picture reprecents the last layer in the betwork
