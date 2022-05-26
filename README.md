# cv_gnn
A realization for classification of defocused ISAR images
# Few-shot dataset
The self-built AIR-MOT dataset can be downloaded from  
https://drive.google.com/file/d/1UgnevdqefRUNyzDvI8uMbr8s5-3msUYe/view?usp=sharing  
The dataset exists in the .mat format.  
# Read dataset
The ISAR image data is in the form of complex-valued. We can get the content by:  
```
  import h5py
  input_dict = h5py.File(dataset)
  img = input_dict['s3']
  img_real = img['real'].astype(np.float32)
  img_imag = img['imag'].astype(np.float32)
