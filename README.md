# FEGTOR
Towards accurate, high quality, conditioned 3D face sequences generation.

## 1. Dataset
 [**BU face data**](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) .<br>


## 2. Model
<img  src="Results/Face 3D.png"  />

## 3. Results
Our results are based on 83 landmarks which should be reconstructed to deform 3D mesh
### Example           
| Happy  | Surprise |  Angry | 
| ------------- | ------------- | ------------- | 
| <img  src="Results/happy.gif"  /> | <img src="Results/surprise.gif"  /> |  <img src="Results/angry.gif"  /> |   
### Rendered Example           
| Happy  | Surprise |  Angry | 
| ------------- | ------------- | ------------- | 
| <img  src="Results/gif/trans/trans_Happy15.gif"  /> | <img src="Results/gif/trans/transSurprise_15.gif"  /> |  <img src="Results/gif/trans/trans_angry15.gif"  /> | 
                                              
## 4. Evaluation
1. FID score


| Model type  | FID |
| ------------- | -------------  |
| CondGRU| 101.5 |
| Action2motion| 33.3 |
| GRU VAE | 28.7 |
| ours | 13.7 |
