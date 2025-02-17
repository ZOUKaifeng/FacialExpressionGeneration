# 3D Facial Expression Generator Based on Transformer VAE.

## 1. Dataset

We test our method on two commonly used facial expression datasets, [**CoMA**](https://coma.is.tue.mpg.de/) and [**BU-4DFE**](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html).

### Data preparation
68 landmarks have been manually sampled on a full mesh of the CoMA dataset. We need to do it only once, since all meshes come in a same topology. Frame length has been normalized as well by subsampling and linear interpolation.


## 2. Model
Our approach is divided into two steps. Firstly, a Transformer VAE is trained to perform the conditional generation of landmarks sequences. Then a landmark-guided mesh deformation (LGMD) model estimates vertex-wise displacements in a frame-by-frame manner, given the desired expression represented by the landmark set and a neutral face to apply the deformation. The overview of our method is shown in the figure below.

<img  src="Results/model1.png"  />
                                       


## 3. Results

### 3.1 Mesh results
The full mesh animation can be obtained by our landmark-driven 3D mesh deformation, based on a Radial Basis Function. Some of the results thus obtained are shown below: <br>
  
#### 3.1.1 Random sampling from Gaussian distribution

 <img  src="Results/gif/random_generation.gif"  /> 
 
#### 3.1.2 Diversity generated expressions

##### Diversity of "mouth open" expression
 <img  src="Results/gif/mouth_open.gif"  />  
 
##### Diversity of "baretheeth" expression
 <img  src="Results/gif/bareteeth_diversity.gif"  />  


 
  
    
#### 3.1.3 Noise removal
Since our model is based on VAE, it generates smoother sequences (right) compared to the original data (left).
  
   <img  src="Results/gif/noise_rem.gif"  /> 
   

### 3.2 Landmark videos for the visual comparison
The outputs of transformer VAE, a set of landmark sequences conditioned by the expression label, have been compared to those from other models for the qualitative jugements. We observed that the Action2Motion model (left) is data-hungry, thus given a small dataset (only a few hundred compared to several thousand in their original motion project) it is under-fitted and does not fully learn the expression behavior. For instance, the generated face does not come back to the neutral pose. The GRU-VAE model (middle) tends to produce less expressive faces. Our model produces expressive faces (comparable to Action2Motion) and successfully learned the expression behavior from a small dataset, i.e. It takes a good balance between the expressiveness in the generation and the size of the dataset it requires. <br> 

#### "Happy" expression

 <img  src="Results/gif/happy.gif"  />  
 
 
 #### "Surprise" expression

 <img  src="Results/gif/surprise.gif"  />  
 
    

## 4. Code
The code will be made available very soon!
