# FEGTOR
3D <b>F</b>acial <b>E</b>xpression <b>G</b>enerator with <b>Tr</b>ansformer-based conditional VAE.

## 1. Dataset
We use  [**BU face data**](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) as our dataset which consists of a total of 606 sequences of 83 manually labeled landmarks extracted from 3D facial scans. 6 basic expressions (anger, disgust, fear, happy, sad, and surprise) have been recorded for each of 101 subjects.

## 2. Model
Our model inherits its architecture from a <a href="https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf}"> conditional variational autoencoder </a>, with a Transformer-based encoder (left rounded box in the figure below) and a Transformer-based decoder (right rounded box). While based on the vanilla <a href="https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf"> Transformer </a>, we use two learnable tokens per expression label <i>c</i> denoted as <i>token<sub>&mu;</sub></i> and <i>token<sub>&Sigma;</sub></i> to feed into the encoder, rather than directly using the label value. We use the expression label to choose the corresponding tokens to be temporarily prepended with the encoded input sequence. 
The first two frames of the encoder output (corresponding to the tokens chosen as input) serve as the distribution parameters: mean <i>&mu;</i> and variance &Sigma;. Note that other encoder outputs are ignored, which implements a <a href="https://arxiv.org/pdf/2104.05670.pdf">time pooling</a>.

<img  src="Results/Face 3D.png"  />
                                       
## 3. Evaluation, results
### 3.1 Quantitative evaluation
We used <b>Fréchet Inception Distance</b> score to evaluate the quality of generation and the <b>classification accuracy</b> for the disentanglement, which is summarized in the table below. 

| Model type  | FID | Accuracy |
| ------------- | -------------  |-------------  |
| CondGRU| 101.5 |  16.69% |
| Action2motion| 33.3 | 60.89 % |
| GRU VAE | 28.7 |  71.07% |
| FEGTOR (ours) | 13.7 |  100% |

### 3.2 Qualitative evaluation
The outputs of transformer VAE, a set of landmark sequences conditioned by the expression label, have been compared to those from other models for the qualitative jugements. <br> 

### Landmark videos for the visual comparison

 <img  src="Results/gif/3DFacial_LM.gif"  />  
 
 
### Videos of rendered meshes 
The full mesh animation can be obtained by our landmark-driven 3D mesh deformation, based on a Radial Basis Function.<br> 
Some of the results thus obtained are shown below:
| Model|Happy  | Surprise |  Angry | 
| ------------- | ------------- | ------------- | -------------------|
| Ours |<img  src="Results/gif/trans/trans_Happy15.gif"  /> | <img src="Results/gif/trans/transSurprise_15.gif"  /> |  <img src="Results/gif/trans/trans_angry15.gif"  /> | 
| | | | |     
| Action2Motion |<img  src="Results/gif/action2motion/Happy15.gif"  /> | <img src="Results/gif/action2motion/Surprise15.gif"  /> |  <img src="Results/gif/action2motion/Angry15.gif"  /> | 
| | | | |  
|GRU-VAE|<img  src="Results/gif/gru/gru_happy15.gif"  /> | <img src="Results/gif/gru/Surprise15.gif"  /> |  <img src="Results/gif/gru/angry15.gif"  /> | 


### Rendered mesh results on COMA dataset
Also has been developed is an autoencoder that can translate a landmark set to a full facial mesh.

 <img  src="Results/gif/BareTeeth.gif"  />  <img src="Results/gif/CheeksIn.gif"  />   <img src="Results/gif/HighSmile.gif"  />  
  
<img  src="Results/gif/LipsUp.gif"  />  <img src="Results/gif/MouthExtrem.gif"  />  <img src="Results/gif/MouthUp.gif"  /> 

## 4. Supplementary material
Derivation of equation (3) from equation (2) in the paper is provided as below:
<img  src="Results/eq2toeq3.PNG"  />

Note: The expectation is about variable z, so a term containing p(c) can be unhinged as a constant which has no impact on the maxmization. 
