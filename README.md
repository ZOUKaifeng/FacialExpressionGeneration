# DisenFace
Towards accurate, high quality, conditioned 3D face sequences generation.

## 1. Dataset
Our dataset consist of [**BU face data**](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) .<br>


## 2. Model
<img  src="Results/Face 3D.png"  />

## 3. Results
Our results are based on 83 landmarks which will be reconstructed to 3D mesh
### Example           
| Happy  | Surprise |  Angry | 
| ------------- | ------------- | ------------- | 
| <img  src="Results/happy.gif"  /> | <img src="Results/surprise.gif"  /> |  <img src="Results/angry.gif"  /> |   
### Rendered Example           
| Happy  | Surprise |  Angry | 
| ------------- | ------------- | ------------- | 
| <img  src="Results/gif/trans/trans_Happy15.gif"  /> | <img src="Results/gif/trans/transSurprise.gif"  /> |  <img src="Results/gif/trans/trans_angry15.gif"  /> | 
                                              
## 4. Evaluation
1. FID score


| Model type | Angry  | Disgust |  Fear | Happy | Sad | Surprise |
| ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |
| Transformer VAE| $62.71\pm11.34$ | $67.77\pm7.69$ | $59.54\pm4.28$ | $58.92\pm7.24$ | $\pm$ | $68.37\pm7.83$ |
| LSTM VAE| $\pm$ | $\pm$ | $\pm$ | $\pm$ | $\pm$ | $\pm$ |
|Ours | $\pm$ | $\pm$ | $\pm$ | $\pm$ | $\pm$ | $\pm$ |
