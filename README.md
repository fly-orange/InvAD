This is the code of "Detecting Both Seen and Unseen Anomalies in Time Series" (being updated)

# Rebuttal 
We have provided the following point-by-point responses to the comments from all reviewers.
- [Rebuttal](#rebuttal)
  - [Response to Review # ATa7](#response-to-review--ata7)
  - [Response to Review # YiQY](#response-to-review--yiqy)
  - [Response to Review # wYcy](#response-to-review--wycy)
  - [Response to Review # eudm](#response-to-review--eudm)
  - [Response to Review # NKTq](#response-to-review--nktq)



## Response to Review \# ATa7
Thank you for your review and insightful comments. 

**Q1. Why the proposed embedding can be decomposed into ID and OOD features?** 
First, ID and OOD features are two features defined by us for different use. ID features are expected distinguish between ID normal and ID anomaly samples. OOD features are expected to distinguish between ID samples and OOD samples. 
Second, we produce them through a training strategy. We force the OOD features of training samples (ID samples) to be a constant tensor, like in DeepSVDD. This brings two benefits: (1) For those unseen samples, the OOD features are expected to be distant from the constant value, which can be used to distinguish them from ID samples. (2) OOD features of training samples is constant means that more information of training samples can be stored in ID features for classification.


**Q2. The purpose of Invertible block is unclear.**
The purpose of the using invertible blocks as our feature encoder is summarized as follows: (1) We aim to produce two kinds of features and ensure no information loss (important for subtle anomaly detection), the two-way architecture and invertbility of INN satisfy these two demainds.
(2) INN can provide a framework that integrates close-set classification (classfication head in the forward process), SVDD (constant ouput in the forward process), reconstruction (back process) and bridge their disadvantages. Specifically, an additional feature can be used to detect OOD samples, which will escape the close-set classification. For reconstruction, regular autoencoder requires crafted design of decoder, i.e, a decoder does not necessarily reconstruct inputs even though the encoder has extracted good features. INN-based framework can guarantee theoretcially perfect reconstruction only if the forward process extracted sufficient features. For SVDD, previous method suffer from mode collapse, where a trivial constant mapping is learnt. INN-based framework can avoid learning a constant mapping that is non-invertible. 

**Q3. How does InvAD perform under different labelled data sizes**
![Few-shot-settings](/pictures/few_shot.png)
<caption>Fig.1 Performance under different size of used labeled data  </caption>

We supplement an experiment to  testify the performance of our method under different size of labelled size. Results presented in Fig.1 show that our method consistently surpasses other two baselines. Furthermore, the AUROC decrease only about 14.2\% on average with only 10\% labeled data compared to all labeled data. It demonstrate that our method is data-effcient.


**Q4. Why unsupervised and weakly supervised setting use different dataset, would proposed method consistently oustand on all datasets?**
The selection of datasets under different settings is based on the availability of anomalies in the training stage. UCR are only tested under the unsupervised settings because it contains no anomalies in the training stage, so can not form a weakly-supervised settings. EMG, SMD, ASD, Credit Card, WaQ, PSM are tested under the weakly supervised settings because they contain anomalies in the training stage and are labelled. Our method can use labels to guide the training process, so we compare it with other weakly-supervised methods and unsupervised methods. Of course, we have added a unsupervised version of our method on these datasets. Although it is inferior to our weakly-supervised version, it still outperform GANF, which is SOTA unsupervised method on these methods. The results also demonstrate that introducing anomaly labels indeed benefit the detection. 

|        |  EMG   | SMD  | ASD | 
| :----: | :----:  |:----:|:----:  |
| GANF  |  8.7/59.1/15.1/63.9/8.2   |  3.5/45.5/6.5/58.2/3.3 |  32.1/100.0/48.6/75.1/32.1    |      
| InvAD(u)  | 14.1/34.8/20.1/65.3/11.2   |  33.3/58.3/42.4/62.3/10.5  | 100.0/50.0/66.7/75.0/29.0  |
| InvAD(w)  | 80.4/80.4/80.4/92.3/67.2  |  61.7/85.3/71.6/89.5/55.8  | 69.2/61.4/65.1/90.0/44.5 |  
|        |  CreditCard   | WaQ  | PSM | 
| GANF  | 0.3/1.5/0.5/72.6/0.3  |  11.0/1.3/2.4/77.5/1.2  | 55.1/57.1/56.1/69.7/39.0     |
| InvAD(u) | 2.0/25.0/3.7/70.2/1.2  | 4.2/41.0/7.7/89.7/4.0  | 60.0/80.3/68.7/79.2/52.3   | 
| InvAD(w) | 37.7/45.4/41.2/92/2/25.9  | 70.1/83.9/76.4/98.8/61.8  | 67.9/79.8/76.3/84.6/58.0  |  
<caption>Table 1. Performance (Precisoon/recall/F1/AUROC/IOU) of unsupervised version of InvAD on EMG, SMD, ASD, CreditCard, WaQ and PSM.</caption>

## Response to Review \# YiQY
Thank you for your review and insightful comments.

**Q1. The evaluation metrics in the field of time series anomaly detection**
I agree with you that the evaluation for TSAD field is not a consensus. The reported metrics is selected based on considerations as follows. (1) For UCR, because there exists only one anomaly in each subdataset, we choose the time window with the highest anomaly score and use affliated F1 to measure the releation between predicted anomly segment and ground truth anomaly segment from the event-perspective. (2) For SAD, EPSY, RS, NATOPS and CT, because they only contain series-level labels, we only make series predictions, and directly use F1 and AUROC without considering point and segment. For EMG, SMD, ASD, Credit Card, WaQ, and PSM, we use point-wise F1, precision and recall because they cannot overestimate the method. Although they may underestimate the method, in our opinion, point-wise predictions is sometimes meaningful, detailed in Q5.
Considering that segment-wise evaluation is sometimes meanningful, we add a metric, namely, AUC of the F1(PA%K) as previous works [1] do, which considers the trade-off between F1 and ill-posed PA. 


|        |  EMG   | SMD  | ASD | 
| :----: | :----:  |:----:|:----:  |
| Input  |  0.373/0.139/0.216   | 0.613/0.174/0.253  |  0.419/0.040/0.068    |      
| Random  | 0.109/0.098/0.108   |  0.058/0.056/0.058  | 0.012/0.012/0.012  |
| RIAE  |  0.231/0.179/0.215  |  0.884/0.221/0.296  | 0.628/0.072/0.119  | 
| InvAE  |  0.877/0.793/0.824 |  0.931/0.687/0.737  | 0.702/0.656/0.674  | 
|        |  CreditCard   | WaQ  | PSM | 
| Input | 0.003/0.003/0.003  | 0.093/0.064/0.071  |  0.552/0.552/0.552    |
| Random | 0.003/0.003/0.003  |  0.225/0.023/0.042 | 0.552/0.552/0.552   | 
| RAIE  | 0.116/0.116/0.116  |  0.406/0.353/0.367 | 0.615/0.568/0.610  |
| InvAE | 0.416/0.415/0.416  |  0.864/0.756/0.777  |0.803/0.763/0.793  | 
<caption>Table 1. Performance (F1/PA, F1/PW, AUC_F1/PA%K) of simple baselines</caption>

**Q2. Simple baselines and the comparison with them**
We have added three simple baselines, "input itself", "random score", "radom initialized AE scores" [2] to evaluate each metrics on datasets. Table 1 present the results. First, point adjustemt(PA) can brings about a great improvement in F1 even for simple baselines, showing the overestimation of this strategy. Second, F1 without PA strategy of these simple baselines remains under 36\% on EMG, SMD, ASD, CreditCard and WaQ, showing that the anomalies in these datasets are not easy to spot. The metrics on PSM is relatively high because it contains a high proportion of anomalies (around 27\%), which are easy to guess by random methods. However, our method outperforms these simple baselines, indicating more complicated anomalies can be identified.   

**Q3. The table about the dataset information**
We have added the detailed table in the main text.

**Q4. Is it fair for this papaer to use flawed time series AD datasets such as SMD**
First, considering the insights in [3], we do not use SWAT, WADI, SMAP, MSL which have proven to have several flaws.  For SMD, we follows [4], and filter 13 sub datasets which have determined flaws, and use the remaining 15 sub dataset. In summary, we realize the evaluation of this field remains diabated, and try to validate the proposed method on as many as possible datasets, including the latest UCR dataset, datasets with clear anomaly definitions (SAD,EPSY, RS, NATOPS, CT), and some previous used datasets after inspection (EMG, SMD, ASD, CreditCard, WaQ, PSM). If there exist any new and flawless datasets, we are willing to validate our method. 


**Q5. Can the point-wise F1 reflect the characteristics of time series anomalies?**
Yes, point-wise F1 may not reflect the anomlous fragment and may underestimate the method. However, we suppose that compared to overestimation, understeimation is tolerable because point-wise F1 sometimes counts in realistic. For example, in CreditCard dataset, it is important to figure which orders(points) are anomalous in the time series. In EMG dataset, locate the specific anomalous points may help interpretation, which is essential in safety-critical applications. In summary, point-wise F1 is a strict but sometimes meaningful metric, so we use it. 
Of course, we agree that segment-wise F1 could also help in some circumstances where anomaly segmentation is not very important. To overcome overestimation, we consider the adjusted PA%K strategy and use its AUC to evaluate the method. The results are presented in Table 1. Performance of more baselines will be added.

[1] 2024_ICDE_Unraveling the ‘Anomaly’ in Time Series Anomaly
Detection: A Self-supervised Tri-domain Solution

[2] 2022_AAAI_Towards a Rigorous Evaluation of Time-series Anomaly Detection

[3] 2021_TKDE_Current Time Series Anomaly Detection Benchamarks Are Flawed And Are Creating the Illusion of Progress

[4] 2023_TMLR_TimeSeAD: Benchmarking Deep Multivariate Time-Series
Anomaly Detection

## Response to Review \# wYcy

Thank you for your review and insightful comments.

**Q1. Reproducibility**
The code is given at .

**Q2. Limited experimentation**
We conduct experiments under 3 settings and 12 datasets, including UCR synthetic dataset (containing 250 subdatasets) and other 11 real-world datasets.

**Q3. Reference list**
Thank you for your reminder. We will revise the reference list.

**Q4. Fig.1 is not informative**
We aim to show that features used for supervised and unsupervised settings are expected to exhibit different characteristics(diverse and centralized), which can not achieved by a one-way feature encoder. This encourages the need of two-way encoder. We will make it clear.

**Q5. ID/OOD to seen/unseen**
ID samples denote those samples drawn from the same distribution as the training data, i.e, they are ‘seen’ in the training stage. OOD samples denote those samples drawn from a different distribution, i.e, they are ‘unseen’ in the training stage. This usage has appeared in previous work [1].

**Q6. Tells ID or OOD**
The method can tell whether the anomaly is ID or OOD. Our method gives classification scores (to detect ID anomalies) and reconstruction scores (to detect OOD anomalies), respectively. We have draw the distribution of these two different scores in Fig. 6. It can be seen that ID anomalies have higher classification scores while OOD anomalies have higher reconstruction scores.

**Q7. In line 342, it says $R^{T \times d}$. Is it T or L ??**
It is $T$, the length of the time windows.
$L$ is the length of total datasets. We use sliding window technique to divide total dataset into several time windows, and feed time windows to our model. 

**Q8. I don't see the relation between being invertible and the jacobian.**
Yes. An invertible $f$ must satisfy that the determinant of its Jacobian matrix is nonequal to zero [3]. For a function $f$ formulated as Eq.(1), its Jacobian determinant is always greater than zero. Furthermore, we can easily find its inverted $g$ formulated as Eq.(3).

**Q9. $L_{rec}$ In Fig.2**
Yes, $L_{rec}$ is not computed in the testing flow. What we want to express is that reconstruction error between reconstructed and original features is used as a part of anomaly scores in the testing flow. We will clarify it.

**Q10. The input embedding layer**
We use the polular time series input embedding layer [3], containing a convolutional embedding and a positional embedding. This layer is trained along with other parts without pre-training. 

**Q11. what is $\sigma()$**
We apologize for the typo and have corrected it.

**Q12. Why and how fix the OOD values to a constant value**
OOD values are expected to exhibit discrepancy between ID (training) and OOD samples. **To realize the goal, we draw the inspiration from the one of the most classical AD algorithm, DeepSVDD [4]**. In DeepSVDD, all training samples are pulled close to a predefined hypersphere center in the feature space, and those unseen samples will be distant from the center in the testing stage. Similarly, we fix the OOD features of ID samples to a predefined constant tensor and expect OOD features of unseen samples to be distant from the constant value. Similar to DeepSVDD, we alsp explore the selection of constant tensor. 
We select 10 random constant value and find that the variance of F1 is smaller than 2.7\%. Furthermore, we adopt optimized and freeze strategy as previous SVDD methods, and find that the result are close to different constant values. This demonstrate that InvAD is more stable than SVDD. This can be explained that SVDD are prone to mode collapse where the learnt function is a constant mapping, while the invertibility of InvAD guarantees the learnt function must not be a constant mapping. 

|        |  SAD   | EPI  | RS | NATOPS | CT 
| :----: | :----:  |:----:|:----:  |:----:  |:----: |
| constant  | 54.4 $\pm$ 0.3   |  90.8 $\pm$ 2.5  |  81.8 $\pm$ 1.1    |  83.9 $\pm$ 1.7     |  71.2 $\pm$ 2.7   |
| optimized | 54.2  | 89.2   | 81.5   |  82.1   |  71.0   |
| freezed   | 54.2  | 89.6   | 82.8  |  83.3   |  72.1   |
<caption>Table.1 Performance under different strategies for the constant tensor</caption>

[1] 2023\_KDD\_Deep Weakly-supervised anomaly detection

[2] 2015\_ICLR\_NICE: non-linear independent component estimation

[3] 2023\_ICLR\_A time series is worth 64 worlds: long term forecasting with transformers

[4] 2018\_ICML\_Deep one-class classification

## Response to Review \# eudm
Thank you for your review and insightful comments.

**Q1. Why $h$ and $z$ come to represent ID and OOD representations, respectively?**
We apologize for your confusion. It should be clarified that it is our training strategy rather than the feature encoder that make the two inputs to represent "ID representations" and "OOD representations". We explain this point from three aspects:
(1) What are ID and OOD representations?
"ID representation" $h$ is defined as a representation that is used to distinguish between seen normal and abnormal samples. "OOD representation" $z$ is the representation that is used to distinguish between ID samples and OOD samples. 
(2) How to make $z$ distinguish between ID samples and OOD samples?
Insprired from the classical DeepSVDD, which pulls all training samples close to a predefined hypersphere center in the representation space, we fix the $z$ of all training samples to a predefined constant tensor through loss in Eq.(7). In the testing stage, $z$ of OOD samples will be distant from the constant tensor and their distance along with the backward reconstruction can acts as the OOD anomaly score. 
(3) How to make $h$ classify ID normal and abnormal samples?
$z$ of ID samples is constant tensor means that more information of training samples can be encoded in $h$, which are used distinguishe ID normal and abnormal samples by supervised learning. 
From the perspective of open-set classification, $h$ and $z$ can be seen as two kind of features to recognize seen class and unseen class, respectively.


**Q2. I do not understand the motivation for using INN**
We use INN for the follow reasons. 
(1) As mentioned in Q1, two kinds of representation are expected to be produced without information loss. The two-way architecture and invertibility of INN satisfy the demand perfectly. Therefore, we explore INN's application for our problem, which has never been explored. 
(2) **INN provides a framework that integrates close-set classification, reconstruction and SVDD, and mitigate their disadvantages.** For reconstruction, regular autoencoder requires crafted design of decoder, i.e, a decoder does not necessarily reconstruct inputs even though the encoder has extracted good features. INN-based framework can guarantee theoretcially perfect reconstruction only if the forward process extracted sufficient features. For SVDD, previous method suffer from mode collapse, where a trivial constant mapping is learnt. INN-based framework can avoid learning a constant mapping that is non-invertible.
Additionally, we conduct a ablation study that use regular NN(LSTM and Transformer) in Table 1. The results show that the their perormance are all inferior to InvAD.
|        |  SAD   | EPI  | RS | NATOPS | CT 
| :----: | :----:  |:----:|:----:  |:----:  |:----: |
| INN    | 54.7  | 90.6   | 82.0   |  84.1   |  71.3   |
| LSTM-AE-based   | 53.7  | 68.9   | 72.2   |  66.2   |  63.3   |
| Transformer-based   | 51.3  | 65.6   | 72.0   |  65.5   |  64.9   |
<caption>Table 1. Comparison with regular NN -based method</caption>

(3) I agree that INN tends to be multi-layered, but INN could be trained using BP without stroing activations, which will save large computaional memory [1]. Furthermore, to reduce the model size and inference time, we explore the simplest TCN as the function in INN blocks. The model size has been smaller than most of existing time series anomaly detection methods in Fig.7(e).


**Q3. How is the input to the INN divided?**
Each input $x \in \mathcal{R}^{T \times D}$  is divided channel wise into $h \in \mathcal{R}^{T \times D_1}$ and $z \in \mathcal{R}^{T \times D_2}$. In implementation, we use $D_1 = D_2 = \frac{D}{2}$. We present the peformance under different division in Table 2. It can be seen small dimension for the ID feature $h$ will lead to performance decrease, indicating that ID features should incorporate sufficient information for classfication. On the other hand, a over small $z$ will also lead to that OOD features of all samples cannot be encoded in a small-dimension constant vetcor. Therefore, half division is a good selection.
|        |  SAD   | EPI  | RS | NATOPS | CT 
| :----: | :----:  |:----:|:----:  |:----:  |:----: |
| 8    | 31.8  | 69.3   | 67.0   |  74.8   |  53.0   |
| 16   | 32.1  | 80.3   | 71.7   |  77.6   |  66.3   |
| 24   | 48.2  | 80.3   | 73.7   |  78.3   |  64.9   |
| 32   | 54.7  | 90.6   | 82.0   |  84.1   |  74.1   |
| 40   | 52.2  | 91.3   | 86.9   |  80.3   |  74.2   |
| 48   | 53.3  | 90.6   | 82.1   |  80.0   |  73.0   |
| 56   | 52.6  | 87.8   | 77.9   |  80.0   |  73.6   |
<caption>Table 2. Performance under different dimension of ID representations </caption>

**Q4. Are there any results evaluating the sensitivity of hyperalpha in equation (1)?**
We present the F1 score under different $\alpha$ as in Table 3 and find that the performance decrease when $\alpha$ is above 0.2. This is because a too high $\alpha$ will make the sigmoid's activations close to zero, which will lead to small or even vanishing gradient.
|        |  SAD   | EPI  | RS | NATOPS | CT 
| :----: | :----:  |:----:|:----:  |:----:  |:----: |
| 0.1   | 54.3 |  89.2  |  82.6    |  79.5  |  72.4  |
| 0.2   | 54.7  | 90.6   | 82.0   |  84.1   |  71.3  |
| 0.6   | 46.2  | 79.4   | 69.9  |  74.0   |  67.5   |
| 0.8   | 40.5  | 78.0   | 68.0  |  83.3   |  63.8   |
| 1.0   | 39.7  | 76.9   | 67.1  |  83.3   |  56.6   |
<caption>Table 3. Peformance under different $\alpha$ </caption>


**Q5. The proposed method cannot be used for unsupervised learning?**
Our method can be used for unsupervised learning when the classfication head and classfication loss are removed. Under this setting, the second term of Eq.(14) acts as the anomaly score, which can be seen as the sum of one-class classfication-based anomaly score and reconstruction-based anomaly scores. In fact, this is what we do in the evaluation on UCR dataset where no labels are provided. Even though, our method can achieve the highest accuracy on UCR datasets. In summary, we explore the application of INN in the time series anomaly detection field, taking adantage its two-way and invertibility property to address different settings.

[1] 2017\_ICLR\_The Reversible Residual Network: Backpropagation Without Storing Activations


## Response to Review \# NKTq

Thank you for your review and insightful comments.

**Q1. Valuation of computation time is missing. Since the input dimension cannot be changed by a reversible neural network, the computation time may be high for data with a large number of dimensions**
Yes, the input dimension can not be changed by INN, so we first transform original signals with high-dimension into an embedding with fixed dimension by a popular input embedding layer in time series analysis. The INN are applied to this embedding. We report the training and inference time for each iteration (batch size = 128, num block = 16) in Table 1. It can be seen that INN will not lead to overhigh training or inference time.
|        |  SAD   | EPI  | RS | NATOPS | CT 
| :----: | :----:  |:----:|:----:  |:----:  |:----: |
| InvAD    | 0.255/0.088  | 0.384/0.181   | 0.390/0.193   |  0.363/0.174   |  0.345/0.069 |
| COCA   | 0.198/0.078  | 0.313/0.122   | 0.260/0.226   |  0.310/0.176   |  0.332/0.135   |
| NCAD   | 0.124/0.079  | 0.343/0.151   | 0.265/0.268   |  0.273/0.161   |  0.326/0.127   |
<caption>Table 1. Training and inference time (s) for each iteration</caption>

**Q2. There are many typos**
We apologize for the typos and have corrected them.

**Q3. Hyperparameter search of comparison methods may narrow the gap with the proposed method.**
Thank you for your suggetions. We are conducting hyperparameters search baselines and will update their results.

**Q4. Is there a possibility that good features cannot be obtained due to architectural limitations caused by the invertible nature of the neural network? Also, can the proposed method be used effectively with high-dimensional data?**
First, to mitigate the arcitectural limitation the capability of the neural network, INN tends to have multi-layers to represent as complex as possible non-linear function. In our paper, we test the performance of different number of blocks and find that when the more than 10 blocks could lead to a good performance.
Second, we have considered the circumstance of high-dimension data, so we first transform original data in a embedding with fixed dimension and apply INN to the embedding. Furthermore, the input embedding layer we use is a popular in time series analysis (such as PatchTST, Dcdetector, GPT4TS, etc.), which consists of a convolutional embedding layer and the positional embedding.

**Q5. This method divides the first input data of the invertible block into two components, does this division affect the results? Also, how should the predefined tensor be determined? Does this value change the performance?**
First, we conduct experiments under different division when total embedding is 64, and present result in Table 2. It can be seen that a too small high-dimension of $h$ or $z$ will result in performace decrease due to insufficient information to represent ID and OOD features. The half division is realtively good, and we use this strategy. 
|        |  SAD   | EPI  | RS | NATOPS | CT 
| :----: | :----:  |:----:|:----:  |:----:  |:----: |
| 8    | 31.8  | 69.3   | 67.0   |  74.8   |  53.0   |
| 16   | 32.1  | 80.3   | 71.7   |  77.6   |  66.3   |
| 24   | 48.2  | 80.3   | 73.7   |  78.3   |  64.9   |
| 32   | 54.7  | 90.6   | 82.0   |  84.1   |  74.1   |
| 40   | 52.2  | 91.3   | 86.9   |  80.3   |  74.2   |
| 48   | 53.3  | 90.6   | 82.1   |  80.0   |  73.0   |
| 56   | 52.6  | 87.8   | 77.9   |  80.0   |  73.6   |
<caption>Table 2. Performance under different dimension of ID features</caption>

Second, we try 10 different constant values random sampled from -1 to 1 and report their performance as follows. It can be seen the variance of F1 is below 2.7\%. Additionally, we adopt "optimize" strategy and "optimize and freeze" strategy like DeepSVDD and find that the performance is still stable. This is because that DeepSVDD is prone to mode collapse due to the learnt trivial constant mapping which is non-invertible, so it requires optimized and freeze strategy. In contrast, our method can mathematically avoid the circunstance by learning a invertible mapping.

|        |  SAD   | EPI  | RS | NATOPS | CT 
| :----: | :----:  |:----:|:----:  |:----:  |:----: |
| constant  | 54.4 $\pm$ 0.3   |  90.8 $\pm$ 2.5  |  81.8 $\pm$ 1.1    |  83.9 $\pm$ 1.7     |  71.2 $\pm$ 2.7   |
| optimized | 54.2  | 89.2   | 81.5   |  82.1   |  71.0   |
| freezed   | 54.2  | 89.6   | 82.8  |  83.3   |  72.1   |


**Q6. Wouldn't equations 7 and 8 have a similar effect?**
Yes, the two equations have a similar effect but are all needed. I agree that the reconstruction loss is zero when all OOD features are c. However, this is a extremely ideal cirmunstances. In practice, OOD features can not be always $c$ even though they are very close to it. **Under this circumstance, a smaller distance will not guarantee the smaller reconstruction error because the network is not linear.** Therefore, a backward loss (Eq.8) is still needed to ensure the reconstruction. Due to this reason, the joint optimization of these two losses is widely accepted in other INN-based applications.