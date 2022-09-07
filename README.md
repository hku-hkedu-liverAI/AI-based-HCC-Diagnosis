# AI-based-HCC-Diagnosis:  This project is supported by HKU and HKEDU

# System requirements and operating system including version numbers
This project is based on Keras + Tensorflow, deployed on the platform with Centos whose version number is CentOS Linux release 7.7.1908 (Core). The hardware information of the server is as below.CPU: Intel(R) Xeon(R)  Gold 6146 CPU @ 3.20 GHz, 2 Thread(s) per core, 12 Core(s) per socket, 2 Socket(s)  ---> 48 CPU(s), GPU: 4 Tesla V100 32 GB, RAM Memory: 500 GB. Although the proposed model is trained on the HPC server, one can reduce the hardware requirement by reducing the resolution of input CT image volume. 

# The version of Keras (for more information via https://keras.io/getting_started/) is 2.2.4, the version of tensorflow-gpu (for more information via https://www.tensorflow.org/) is 1.15.0. 
To ensure the successful utilization of GPU computation power, one should install the correct version of GPU driver library first. As for us, the version of cudnn is 7.6.0 and the version of cudatoolkit is 10.1.168. Second, one can install tensorflow-gpu and Keras. In the following, one should install numpy, scipy, panda, opencv, pillow, pydicom, etc. 

# Installation Guide: Instruction + Typical Install time on a normal desktop computer
(1) Instrcution has desribed above, which can be summaried by: cudnn, cudatoolkit --> tensorflow-gpu, keras ---> other required libraries like numpy, scipy, and so on.
(2) Generally, the installation can be completed within two or four hours when the network connection has a good quality.
For easy installation, one can watch this video for more detailed information via 
https://www.youtube.com/watch?v=OEFKlRSd8Ic&ab_channel=JeffHeaton.

