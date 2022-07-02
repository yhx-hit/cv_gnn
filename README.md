# cv_gnn
A realization for classification of defocused ISAR images
# Abstract
Recently, researches on classification for inverse synthetic aperture radar (ISAR) images continue to deepen. However, the maneuvering and attitude adjustment of space targets will bring high-order terms to received echoes which cause defocus on ISAR images. The current classification models ignore the information of high-order terms containing in the relationship of real parts and imaginary parts of data. To this end, this letter proposes an end-to-end framework, called CV-GNN, specifically for the classification of defocused ISAR images under the condition of small samples. It models the features of real parts and imaginary parts of complex-valued (CV) images as graph information reasoning. Specifically, the deep relationship between them is mined to contribute to classification by complex-valued graph convolution. Moreover, the backpropagation process is derived in detail for updating weights and bias of the network. In addition, a few-shot dataset of defocused ISAR images is built for experiments, and ablation studies verify the effectiveness of complex-value and graph neural network. Compared with state-of-the-art methods, CV-GNN performs well in defocused image classification for each class of targets. 
![img](https://github.com/yhx-hit/cv_gnn/blob/main/heart.gif)
# Few-shot dataset
The self-built few-shot dataset of defocused ISAR images can be downloaded from  
https://drive.google.com/file/d/1ng5M1GpZlIVg6y_CLPbkFHqsEMLDRJsR/view?usp=sharing  
The dataset exists in the .mat format.  
# Read dataset
The ISAR image data is in the form of complex-valued. We can get the content by:  
```
  import h5py
  input_dict = h5py.File(dataset)
  img = input_dict['s3']
  img_real = img['real'].astype(np.float32)
  img_imag = img['imag'].astype(np.float32)
