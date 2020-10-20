# Thermal Imaging Monitoring System Using Simple Kriging and Generative Adversarial Networks

# Abstract
The notion of this project is to replace the need of thermal cameras in various domains, by introducing few agile techniques such as Simple Kriging along with various approaches of deep learning neural networks. Due to limited access to hardware resources, this project illustrates on building theoretical prototype that showers insights on generating thermal map using advance Kriging techniques, as well as briefly elaborate on deep learning technique such as Conditional Generative Adversarial Network (cGAN) that promote cognizance towards image-to-image translation to convert the generated heat map into an image. Nevertheless, this conceptual prototype may be extended to real time system that binds both concepts of Kriging and cGAN to arrive at fine thermal image generation fostering the elimination or replacement of usage of thermal cameras. Here, hard coded values have been inputted to Simple Kriging method and open source datasets have been employed to train and test GANs to translate RGB image to real time image.

# Introduction
In the recent years, there has been enormous usage of thermal cameras in various domains. These sensors are primarily designed to determine range of surface temperatures, that can very efficiently monitor and control low and high temperature critical processes. With upswing in usage of thermal cameras, these have potentially become as one high cost commodity in industrial market. High investments are made to purchase these cameras for using them, as it is utilized for monitoring the flaws of different industrial components by the analysis of generated thermal map. It is also observed that, large number of applications require thermal sensors to extract information from certain regions of interest. For instances, medical applications, energy efficiency, automobiles etc. Thus, idea adopted in this paper is methodologically described to one of the efficient approach, to replace thermal sensors and/or cameras, that can generate heat map thermal image for monitoring, with the help of Kriging technique and Deep Learning Neural Networks. 
This project guides one in gathering some comprehensible understanding on working of Kriging techniques along with basics of Generative Adversarial Network (GAN). Various types of Kriging methods and its corresponding experimental equations are illustrated. It also presents various connecting neural network model that reside under GANs. Following this background section, experimental results are presented in section III, that demonstrate on various illustrations of generated heat maps from Kriging methods, along with some image-to-image translated output images. Finally concluding with inferences and future work.
Outputs of this hypothesis have been generated with the use of two open source datasets, Façade and Cityscapes, for training and testing deep learning neural network. While user defined values have been adopted for arriving at outputs from Kriging algorithms. 

# Simple- Kriging
Kriging is a method of spatial interpolation, which is frequently used in mining, soil and environmental science.This technique is one of the best approaches to estimate the
unsampled points in the given region of interest of a spatial field, by utilizing discrete values obtained from different locations of the workspace and determine the correlation between the sample points in the region of interest. 

In case of simple kriging, a plane ‘W’, of dimension 100 x 100 is considered, as region of interest. Assuming there are four data known sampled points in the region of interest as shown in the fig 1.

![alt text](https://github.com/phaneeshwar/Msc-Project/blob/main/Img1.PNG)

Fig 1: Graphical visualization of points in a workspace

In this design, hard coded temperature data values were given as input. Simple kriging technique was implemented to determine temperatures at unknown locations. Eventually,   semi variance and semi variogram is plotted accordingly, shown in the fig 2.

![alt text](https://github.com/phaneeshwar/Msc-Project/blob/main/Img2.PNG)

Fig 2: Represents the scatter plot of temperature data points


![alt text](https://github.com/phaneeshwar/Msc-Project/blob/main/Img3.PNG)

Fig. 3: Plot of Semi-Variogram

 The test point ‘K’ (fig. 1) is determined by evaluating its relation with other known points in the field. This is achieved by determining covariance matrix between the unsampled test point and every other known sampled points. 
The above two covariance matices assist in realizing weight matrix, which yields the dependancy of variable ‘K’ on other variables. With the help of weight matrix, the unknown test point is discovered. 
Similarly, every unknown point in the desired region of interest can be realized, to generate heat map of the region. And lastly the pixel values are assigned for different ranges of determined heat values. 

Output Heat map obtained :

![alt text](https://github.com/phaneeshwar/Msc-Project/blob/main/Image%204.PNG)

Fig. 4 Heat Map Obtained from Kriging


![alt text](https://github.com/phaneeshwar/Msc-Project/blob/main/Img5.PNG) 

Fig. 5 Heat Map Obtained from Kriging

# CGANs or Pix2Pix

Pix2Pix GANs networks are type of conditional GAN or cGAN in short, which involves output samples of generator is conditional for a given input. Pix2Pix is highly used to design general-purpose image-to-image translation. These networks are effective at image synthesis and overcome limitations of creating fake samples by normal GAN networks. Input of this network is correlated with additional information such as class labels, which are used to improve efficiency of the GAN models. 

Conditional GANs were applied and tested using variety of tasks and datasets. The dataset used in this paper for analysis is from Façade and Cityscapes. 
Façade dataset have 400 training image samples with dimensions of 512 x 256 and a batch size of 1. In case of cityscapes, there are 2975 training images with dimensions of 512 x 256 and a batch size of 10.Number of epochs chosen for Façade and cityscapes dataset is 150 and 200 epochs, respectively.

The cGAN network is programmed with aide of TensorFlow and Keras API using Google Colaboratory or in short, colab. It supports Tensor processing unit (TPU), Graphics Processing Unit (GPU). 12GB NVidia Tesla k80 GPU is utilized and this gradually increases the speed of training process that cannot be achieved with help of CPU.

![alt text](https://github.com/phaneeshwar/Msc-Project/blob/main/Img6.PNG)

![alt text](https://github.com/phaneeshwar/Msc-Project/blob/main/Img7.PNG)

![alt text](https://github.com/phaneeshwar/Msc-Project/blob/main/Img8.PNG)

# Conclusion

In this project, the heat map has been generated using Simple Kriging method. Currently, hard coded values were fed as input to design code used for simple kriging technique. Appropriate semi-variogram plot was determined, from which variations were analyzed such as sill, range, etc. 
Using these attributes suitable variogram model has been used to determine the covariance matrix for obtaining simple kriging weights. Appropriate heat map has been generated for given temperature dataset. 

Furthermore, conditional generative adversarial networks were used to perform image-to-image translation functional tasks. These networks learn loss functions from the given dataset and determine the output vector distribution. 
Results obtained from this network were satisfactory with minimal differences observed between the target and generated image. cGAN prototype constructed would perfectly work for RGB-to-Thermal Image dataset. 

As part of future work, this prototype can be extended to real time system, by continuously monitoring the temperature data in the given workspace and transmit the data to the data logger system. This data can be given as an input to the simple kriging program logic to generate heat maps for a fixed delay of time. 
In addition, generated heat map could be transferred to conditional generative adversarial networks to obtain thermal images of the workspace. 
In this way, the defined work space can be remotely monitored with less human effort.










