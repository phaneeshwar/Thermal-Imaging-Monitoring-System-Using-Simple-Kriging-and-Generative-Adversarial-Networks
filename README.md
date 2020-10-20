# Thermal Imaging Monitoring System Using Simple Kriging and Generative Adversarial Networks

# Abstract
The notion of this project is to replace the need of thermal cameras in various domains, by introducing few agile techniques such as Simple Kriging along with various approaches of deep learning neural networks. Due to limited access to hardware resources, this paper illustrates on building theoretical prototype that showers insights on generating thermal map using advance Kriging techniques, as well as briefly elaborate on deep learning technique such as Conditional Generative Adversarial Network (cGAN) that promote cognizance towards image-to-image translation to convert the generated heat map into an image. Nevertheless, this conceptual prototype may be extended to real time system that binds both concepts of Kriging and cGAN to arrive at fine thermal image generation fostering the elimination or replacement of usage of thermal cameras. Here, hard coded values have been inputted to Simple Kriging method and open source datasets have been employed to train and test GANs to translate RGB image to real time image.

# Simple- Kriging
Kriging is a method of spatial interpolation, which is
frequently used in mining, soil and environmental science.
This technique is one of the best approaches to estimate the
unsampled points in the given region of interest of a spatial
field, by utilizing discrete values obtained from different
locations of the workspace and determine the correlation
between the sample points in the region of interest. 

In case of simple kriging, a plane ‘W’, of dimension 100 x 100 is considered, as region of interest. Assuming there are four data known sampled points in the region of interest as shown in the fig.

![alt text]https://github.com/phaneeshwar/Msc-Project/blob/main/Img1.PNG



 
Fig. 16: Graphical visualization of points in a workspace

In this design, hard coded temperature data values were given as input. Simple kriging technique was implemented to determine temperatures at unknown locations. Eventually,   semi variance and semi variogram is plotted accordingly, shown in the fig. 17(b).

 
Fig.17 (a): Represents the scatter plot of temperature data points


 
Fig. 17 (b): Plot of Semi-Variogram
    The test point ‘K’ (fig. 16) is determined by evaluating its relation with other known points in the field. This is achieved by determining covariance matrix between the unsampled test point and every other known sampled points. 
The above two covariance matices assist in realizing weight matrix, which yields the dependancy of variable ‘K’ on other variables. With the help of weight matrix, the unknown test point is discovered. 
Similarly, every unknown point in the desired region of interest can be realized, to generate heat map of the region. And lastly the pixel values are assigned for different ranges of determined heat values. 


 
Fig. 18 (a)

 
Fig. 18 (b)


Output Heat map obtained :




