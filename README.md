# FEGTOR
3D Facial expression generator with Transformer-based conditional VAE.

## 1. Dataset
Our dataset is generated from the  [**BU face data**](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) .<br> It consists of a total of 606 sequences of 83 manually labeled landmarks extracted from 3D facial scans. 6 basic expressions (anger, disgust, fear, happy, sad, and surprise) have been recorded for each of 101 subjects.

## 2. Model
<img  src="Results/Face 3D.png"  />

                                       
## 3. Evaluation
1. FID score


| Model type  | FID | Accuracy |
| ------------- | -------------  |
| CondGRU| 101.5 |  16.69% |
| Action2motion| 33.3 | 60.89 % |
| GRU VAE | 28.7 |  71.07% |
| Ours | 13.7 |  100% |
## 4. Results
The outputs of transformer VAE are a set of landmarks sequence.<br> 
The comparison is showed below:
### Example           
| Happy  | Surprise |  Angry | 
| ------------- | ------------- | ------------- | 
| <img  src="Results/happy.gif"  /> | <img src="Results/surprise.gif"  /> |  <img src="Results/angry.gif"  /> |   
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
