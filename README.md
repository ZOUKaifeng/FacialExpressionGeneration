# 3D Facial Expression Generator Based on Transformer VAE.

## 1. Dataset

We test our method on [**CoMA dataset**](https://coma.is.tue.mpg.de/) and [**BU-4DFE data**](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html).
## 2. Model
Our approach is divided into two steps. Firstly, a Transformer VAE is trained to perform the conditional generation of landmarks sequences. Then a landmark-guided mesh deformation (LGMD) model estimates vertex-wise displacements, which is used to deform a neutral face to the expected expression frame by frame.

<img  src="Results/model1.png"  />
                                       


### 3. Results

#### Diversity of mouth open
 <img  src="Results/diversity_mouth_open.jpg"  />  
 
#### Diversity of baretheeth
 <img  src="Results/diversity_bareteeth.jpg"  />  



#### Exaggeration
Since we add the displacements to the neutral face to deform it, an exaggerated expression can easily be obtained by multiplying the displacement with a constant value.
 <img  src="Results/exaggeration.jpg"  />

### Landmark videos for the visual comparison
The outputs of transformer VAE, a set of landmark sequences conditioned by the expression label, have been compared to those from other models for the qualitative jugements. <br> 
 <img  src="Results/gif/3DFacial_LM.gif"  />  
 
 
### Videos of rendered meshes 
The full mesh animation can be obtained by our landmark-driven 3D mesh deformation, based on a Radial Basis Function. Some of the results thus obtained are shown below: <br>
| Model|Happy  | Surprise |  Angry | 
| ------------- | ------------- | ------------- | -------------------|
| Ours |<img  src="Results/gif/trans/trans_Happy15.gif"  /> | <img src="Results/gif/trans/transSurprise_15.gif"  /> |  <img src="Results/gif/trans/trans_angry15.gif"  /> | 
| | | | |     
| Action2Motion |<img  src="Results/gif/action2motion/Happy15.gif"  /> | <img src="Results/gif/action2motion/Surprise15.gif"  /> |  <img src="Results/gif/action2motion/Angry15.gif"  /> | 
| | | | |  
|GRU-VAE|<img  src="Results/gif/gru/gru_happy15.gif"  /> | <img src="Results/gif/gru/Surprise15.gif"  /> |  <img src="Results/gif/gru/angry15.gif"  /> | 

<!--
### Rendered mesh results on <a href=https://coma.is.tue.mpg.de> COMA</a> dataset
Also has been developed is an autoencoder that can translate a landmark set to a full facial mesh.

 <img  src="Results/gif/BareTeeth.gif"  />  <img src="Results/gif/CheeksIn.gif"  />   <img src="Results/gif/HighSmile.gif"  />  
  
<img  src="Results/gif/LipsUp.gif"  />  <img src="Results/gif/MouthExtrem.gif"  />  <img src="Results/gif/MouthUp.gif"  /> 
-->
## 4. Supplementary material
Derivation of equation (3) from equation (2) in the paper is provided as below:
<img  src="Results/eq2toeq3.PNG"  />

Note: The expectation is about variable z, so a term containing p(c) can be unhinged as a constant which has no impact on the maxmization. 
