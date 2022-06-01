# FEGTOR
3D <b>F</b>acial <b>E</b>xpression <b>G</b>enerator with <b>Tr</b>ansformer-based conditional VAE.

## 1. Dataset
We use  [**BU face data**](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) as our dataset which consists of a total of 606 sequences of 83 manually labeled landmarks extracted from 3D facial scans. 6 basic expressions (anger, disgust, fear, happy, sad, and surprise) have been recorded for each of 101 subjects.

## 2. Model
Our model inherits its architecture from a <a href="https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf}"> conditional variational autoencoder </a>, with a Transformer-based encoder (left rounded box in the figure below) and a Transformer-based decoder (right rounded box). While based on the vanilla <a href="https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf"> Transformer </a>, we use two learnable tokens per expression label <i>c</i> denoted as <i>token<sub>&mu;</sub></i> and <i>token<sub>&Sigma;</sub></i> to feed into the encoder, rather than directly using the label value. We use the expression label to choose the corresponding tokens to be temporarily prepended with the encoded input sequence. 
The first two frames of the encoder output (corresponding to the tokens chosen as input) serve as the distribution parameters: mean <i>&mu;</i> and variance &Sigma;. Note that other encoder outputs are ignored, which implements a <a href="https://arxiv.org/pdf/2104.05670.pdf">time pooling</a>.

<img  src="Results/Face 3D.png"  />
                                       
## 3. Evaluation
1. FID score


| Model type  | FID | Accuracy |
| ------------- | -------------  |-------------  |
| CondGRU| 101.5 |  16.69% |
| Action2motion| 33.3 | 60.89 % |
| GRU VAE | 28.7 |  71.07% |
| Ours | 13.7 |  100% |
## 4. Results
The outputs of transformer VAE are a set of landmarks sequence.<br> 
The comparison is shown below:
### Example           

 <img  src="Results/gif/3DFacial_LM.gif"  />  
 
 
 
### Rendered Example 
We apply the expression based on landmarks on a neutral mesh by ICP and RBF.<br> 
The results of three methods are shown below for comparison:
| Model|Happy  | Surprise |  Angry | 
| ------------- | ------------- | ------------- | -------------------|
| Ours |<img  src="Results/gif/trans/trans_Happy15.gif"  /> | <img src="Results/gif/trans/transSurprise_15.gif"  /> |  <img src="Results/gif/trans/trans_angry15.gif"  /> | 
| | | | |     
| Action2Motion |<img  src="Results/gif/action2motion/Happy15.gif"  /> | <img src="Results/gif/action2motion/Surprise15.gif"  /> |  <img src="Results/gif/action2motion/Angry15.gif"  /> | 
| | | | |  
|GRU-VAE|<img  src="Results/gif/gru/gru_happy15.gif"  /> | <img src="Results/gif/gru/Surprise15.gif"  /> |  <img src="Results/gif/gru/angry15.gif"  /> | 


Results on COMA dataset. We developed a autoencoder to transform landmark to mesh.

 <img  src="Results/gif/BareTeeth.gif"  />  <img src="Results/gif/CheeksIn.gif"  />   <img src="Results/gif/HighSmile.gif"  />  
  
<img  src="Results/gif/LipsUp.gif"  />  <img src="Results/gif/MouthExtrem.gif"  />  <img src="Results/gif/MouthUp.gif"  /> 

## 5. Supplementary material
Proof of equation (2) to equation (3) in the paper.
<img  src="Results/eq2toeq3.PNG"  />

PS: The expectation is about variable z, so a term containing p(c) can be unhinged as a constant which has no impact on maxmizing. 
