from layout_tools import *
import urllib.parse
import re

def convert(text):
    def toimage(x):
        if x[1] and x[-2] == r'$':
            x = x[2:-2]
            img = r"\n<img src='https://math.vercel.app?from={}&color=black' style='display: block; margin: 0.5em auto;'>\n".format(urllib.parse.quote_plus(x))
            return img
        else:
            x = x[1:-1]
            return r'![](https://math.vercel.app?from={}&color=black)'.format(urllib.parse.quote_plus(x))
    return re.sub(r'\${2}([^$]+)\${2}|\$(.+?)\$', lambda x: toimage(x.group()), text)

text_background='''

## Public Datasets 

We identified and collected 18 datasets proposed in the past decades in the literature containing 1887 time series with labeled anomalies. Specifically, each point in every time series is labeled as normal or abnormal. 
The following table summarizes relevant characteristics of the datasets, including their size and length, as well as statistics about the anomalies. The first 8 datasets originally contained univariate time series, 
whereas the remaining 10 datasets originally contained multivariate time series that we converted into univariate time series. Specifically, we run our AD methods on each dimension separately, and we keep those dimensions where at least one method achieves AUC-ROC>0.8. 

Even though some of these datasets are publicly available (e.g., in code repositories), we could not identify works performing evaluations in a large portion of them. The main reason is the laborious task of identifying and collecting datasets across different communities 
and, subsequently, processing and formatting the datasets to bring them in a unified format. For some datasets, complicated documentation describes the collection process and instructions for extracting anomalies. In other cases, the lack of documentation hinders the 
process of utilizing the datasets. We relieve the community from this task and provide datasets in a unified format with the scripts for extracting anomalies from the original data. 

Briefly, the benchmark includes the following datasets:

* __Dodgers__ ({} time series): 
 
 This dataset is a loop sensor data for the Glendale on-ramp for the 101 North freeway in Los Angeles and the anomalies represent unusual traffic after a Dodgers game.
* __ECG__ ({} time series): 
 
 It is a standard electrocardiogram dataset and the anomalies represent ventricular premature contractions. 
* __IOPS__ ({} time series): 
 
 This is a dataset with performance indicators that reflect the scale, quality of web services, and health status of a machine.
* __KDD21__ ({} time series): 
 
 THis dataset is a composite dataset released in a recent SIGKDD 2021 competition.
* __MGAB__ ({} time series): 
 
 This is composed of Mackey-Glass time series with non-trivial anomalies. Mackey-Glass time series exhibit chaotic behavior that is difficult for the human eye to distinguish.
* __NAB__ ({} time series): 
 
 It is composed of labeled real-world and artificial time series including AWS server metrics, online advertisement clicking rates, real time traffic data, and a collection of Twitter mentions of large publicly-traded companies.
* __NASA-SMAP and NASA-MSL__ ({} time series): 
 
 These dataset are two real spacecraft telemetry data with anomalies from Soil Moisture Active Passive (SMAP) satellite and Curiosity Rover on Mars (MSL). We only keep the first data dimension that presents the continuous data, and we omit the remaining dimensions with binary data. 
* __SensorScope__ ({} time series): 
  
 This is a collection of environmental data, such as temperature, humidity, and solar radiation, collected from a typical tiered sensor measurement system.
* __Yahoo__ ({} time series): 
 
 It is a dataset published by Yahoo labs consisting of real and synthetic time series based on the real production traffic to some of the Yahoo production systems.
* __Daphnet__ ({} time series): 
 
 This contains the annotated readings of 3 acceleration sensors at the hip and leg of Parkinson's disease patients that experience freezing of gait (FoG) during walking tasks.
* __GHL__ ({} time series): 
 
 It is a Gasoil Heating Loop Dataset and contains the status of 3 reservoirs such as the temperature and level. Anomalies indicate changes in max temperature or pump frequency.
* __Genesis__ ({} time series): 
 
 This is a portable pick-and-place demonstrator which uses an air tank to supply all the gripping and storage units.
* __MITDB__ ({} time series): 
 
 This dataset contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979.
* __OPPORTUNITY (OPP)__ ({} time series): 
 
 It is a dataset devised to benchmark human activity recognition algorithms (e.g., classification, automatic data segmentation, sensor fusion, and feature extraction). The dataset comprises the readings of motion sensors recorded while users executed typical daily activities.
* __Occupancy__ ({} time series): 
 
 This contains experimental data used for binary classification (room occupancy) from temperature, humidity, light, and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.
* __SMD__ (Server Machine Dataset) ({} time series):
 
 It is a 5-week-long dataset collected from a large Internet company. This dataset contains 3 groups of entities from 28 different machines.
* __SVDB__ ({} time series): 
 
 This dataset includes half-hour ECG recordings chosen to supplement the examples of supraventricular arrhythmias in the MIT-BIH Arrhythmia Database.

'''.format(
	len(df.loc[df['dataset'] == 'Dodgers']),
	len(df.loc[df['dataset'] == 'ECG']),
	len(df.loc[df['dataset'] == 'IOPS']),
	len(df.loc[df['dataset'] == 'MGAB']),
	len(df.loc[df['dataset'] == 'NAB']),
	len(df.loc[df['dataset'] == 'NASA_SMAP']),
	len(df.loc[df['dataset'] == 'NASA_MSL']),
	len(df.loc[df['dataset'] == 'SensorScope']),
	len(df.loc[df['dataset'] == 'YAHOO']),
	len(df.loc[df['dataset'] == 'KDD21']),
	len(df.loc[df['dataset'] == 'Daphnet']),
	len(df.loc[df['dataset'] == 'Genesis']),
	len(df.loc[df['dataset'] == 'GHL']),
	len(df.loc[df['dataset'] == 'MITDB']),
	len(df.loc[df['dataset'] == 'Occupancy']),
	len(df.loc[df['dataset'] == 'OPPORTUNITY']),
	len(df.loc[df['dataset'] == 'SMD']),
	len(df.loc[df['dataset'] == 'SVDB']),
	)




background_method = '''

## Anomaly Detection Methods

For the initial evaluation we consider the following strong baselines. 

* __Isolation Forest (IForest)__: 
 
 This method constructs the binary tree based on the space splitting and the nodes with shorter path lengths to the root are more likely to be anomalies. 
* __The Local Outlier Factor (LOF)__: 
 
 This method computes the ratio of the neighboring density to the local density. 
* __The Histogram-based Outlier Score (HBOS)__: 
 
 This method constructs a histogram for the data and the inverse of the height of the bin is used as the outlier score of the data point. 
* __Matrix Profile (MP)__: 
 
 This method calculates as anomaly the subsequence with the most significant 1-NN distance. 
* __NORMA__: 
 
 This method identifies the normal pattern based on clustering and calculates each point's effective distance to the normal pattern. 
* __Principal Component Analysis (PCA)__: 
 
 This method projects data to a lower-dimensional hyperplane, and data points with a significant distance from this plane can be identified as outliers. 
* __Autoencoder (AE)__: 
 
 This method projects data to the lower-dimensional latent space and reconstructs the data, and outliers are expected to have more evident reconstruction deviation. 
* __LSTM-AD__: 
 
 This method build a non-linear relationship between current and previous time series (using Long-Short-Term-Memory cells), and the outliers are detected by the deviation between the predicted and actual values 
* __Polynomial Approximation (POLY)__: 
 
 This method build a non-linear relationship between current and previous time series (using polynomial decomposition), and the outliers are detected by the deviation between the predicted and actual values
* __CNN__: 
 
 This method build a non-linear relationship between current and previous time series (using convolutional Neural Network), and the outliers are detected by the deviation between the predicted and actual values. 
* __One-class Support Vector Machines (OCSVM)__: 
 
 This method fits the dataset to find the normal data's boundary.
'''

background_method_param = '''

### Parameters

We set the parameters as follows. 

* For Isolation Forest we consider two variants: IForest1 indicates the IForest model without a sliding window, 
which is suited to the global point outliers, whereas IForest considers the sliding window variant. 
For IForest/IForest1, we use the default 100 base estimators in the tree 
ensemble. For LOF, we follow the default setting in this model and we use 20 as the number of neighbors.

* For MP we set the window as the period of the time series, estimated using the autocorrelation function. We use 
the same period estimation for all methods requiring to set a a sliding window. 

* For NORMA, we follow the default parameter settings in the paper. Similarly to MP, the pattern length is 
estimated with the autocorrelation function. We set the normal model of length to be 3*pattern length and sample 40% of the data without overlapping. 

* For PCA, we use 10 principal components. 

* For POLY, the best model is selected from the following settings. 
The power of polynomial fitted to the data is 0 or 3. The length of the window to be predicted is 20 or the period of the series. 

* LSTM-AD, AE, CNN, and OCSVM are semi-supervised algorithms that require anomaly-free training data. In our test, 
only KDD21, NASA-SMAP, and NASA-MSL contain anomaly-free training data. For the other datasets, 
we train the models on the initial regions of the time series. 
Specifically, the training ratio for YAHOO is 30% and for the remaining datasets is 10%. 
We expect a low-density anomalies (< 5%) in the training dataset would not affect the result. 
However, we note that for some datasets with higher contamination ratios the results 
for the semi-supervised algorithms could likely further improved. We also highlight that several of the methods 
into consideration require minimal tuning and the default values reported in the corresponding papers or codes 
we rely on work well across datasets.

* For OCSVM, we set the upper bound on the fraction of training errors to be 0.05. 

* For LSTM-AD, we use the following parameters: 
two LSTM layers with units=50, then a Dense layer with units=1. loss='mse', optimizer='adam', 
validation split ratio=0.15, batch size=64, epochs = 50, patience = 5. 

* For AE, the best model is selected 
from three MLP-based Autoencoders with the architectures given by (32,16,8,16,32), (32,8,32), (32,16,32). 
Activation function is ReLU. Each number indicates the units for the corresponding Dense layer. 
Then one Dense layer with units = the length of the input, validation split ratio=0.15, batch size=64, 
epochs=100, patience=5, optimizer='adam', loss='mse'. 

* Finally, for CNN, we use three Convolutional Blocks 
(filters=8,16,32, kernel size=2, strides=1) with Max Pooling (pool size=2) and ReLU. Then one Dense layer 
with units=64, then one Dropout layer with rate=0.2, then one dense layer with units=1. loss='mse', optimizer='adam', 
validation split ratio=0.15, batch size=64, epochs = 100, patience = 5. 

'''




background_notation=r'''

## Evaluation Measures

We define here the evaluation measures used in the demonstration.
We first introduce formal notations useful for the rest of the demo. Then, we review in detail previously proposed evaluation measures for time-series AD methods. 
We review notations for the time series and anomaly score sequence.

***

### Time Series: 

A time series $ T \in \mathbb{R}^n $ is a sequence of
real-valued numbers $ T_i\in\mathbb{R} $ $ [T_1,T_2,...,T_n] $, where
$ n=|T| $ is the length of $ T $, and $ T_i $ is the $i^{th}$ point of $ T $. We
are typically interested in local regions of the time series, known as
subsequences. A subsequence $ T_{i,\ell} \in \mathbb{R}^\ell $ of a time
series $ T $ is a continuous subset of the values of $ T $ of length $ \ell $
starting at position $ i $. Formally,
$ T_{i,\ell} = [T_i, T_{i+1},...,T_{i+\ell-1}] $. 

***

### Anomaly Score Sequence:

For a time series $T \in \mathbb{R}^n$, an AD method $A$
returns an anomaly score sequence $S_T$. For point-based approaches
(i.e., methods that return a score for each point of $T$), we have
$S_T \in \mathbb{R}^n$. For range-based approaches (i.e., methods that
return a score for each subsequence of a given length $\ell$), we have
$S_T \in \mathbb{R}^{n-\ell}$. Overall, for range-based (or
subsequence-based) approaches, we define
$S_T = [{S_T}_1,{S_T}_2,...,{S_T}_{n-\ell}]$ with ${S_T}_i \in [0,1]$.

***

### Background Evaluation Measures

We present previously proposed quality measures for evaluating the
accuracy of an AD method given its anomaly score. We first discuss
threshold-based and, then, threshold-independent measures.

#### Threshold-based AD Evaluation Measures


The anomaly score $S_T$ produced by an AD method $A$ highlights the
parts of the time series $T$ considered as abnormal. The highest values
in the anomaly score correspond to the most abnormal points.
Threshold-based measures require to set a threshold to mark each point
as an anomaly or not. Usually, this threshold is set to
$\mu(S_T) + \alpha*\sigma(S_T)$, with $\alpha$ set to
3, where $\mu(S_T)$ is the mean and $\sigma(S_T)$
is the standard deviation $S_T$. Given a threshold $Thres$, we compute
the $pred \in \{0,1\}^n$ as follows:


$\forall i \in [1,|S_T|],$

$pred_i = 0, \text{if: } {S_T}_i < Thres $

$pred_i = 1, \text{if: } {S_T}_i \geq Thres $


Threshold-based measures compare $pred$ to $label \in \{0,1\}^n$, which
indicates the true (human provided) labeled anomalies. Given the
Identity vector $I=[1,1,...,1]$, the points detected as anomalies or not
fall into the following four categories:

-   **True Positive (TP)**: Number of points that have been correctly
    identified as anomalies. Formally: $TP = label^\top \cdot pred$.

-   **True Negative (TN)**: Number of points that have been correctly
    identified as normal. Formally:
    $TN = (I-label)^\top \cdot (I-pred)$.

-   **False Positive (FP)**: Number of points that have been wrongly
    identified as anomalies. Formally: $FP = (I-label)^\top \cdot pred$.

-   **False Negative (FN)**: Number of points that have been wrongly
    identified as normal. Formally: $FN = label^\top \cdot (I-pred)$.

Given these four categories, several quality measures have been proposed
to assess the accuracy of AD methods. 

**Precision:** We define Precision
(or positive predictive value) as the number correctly identified
anomalies over the total number of points detected as anomalies by the
method: $Precision = \frac{TP}{TP+FP}$

**Recall:** We define Recall (or True Positive Rate
(TPR), $tpr$) as the number of correctly identified anomalies over all
anomalies: $Recall = \frac{TP}{TP+FN}$

**False Positive Rate (FPR):** A supplemental measure to the Recall is
the FPR, $fpr$, defined as the number of points wrongly identified as
anomalies over the total number of normal points:
$fpr = \frac{FP}{FP+TN}$

**F-Score:** Precision and Recall evaluate two
different aspects of the AD quality. A measure that combines these two
aspects is the harmonic mean $F_{\beta}$, with non-negative real values
for $\beta$: $F_{\beta} = \frac{(1+\beta^2)*Precision*Recall}{\beta^2*Precision+Recall}$ 

Usually, $\beta$ is set to 1, balancing the importance
between Precision and Recall. In this demo, $F_1$ is referred to as F
or F-score. 

**Precision@k:** All previous measures require an anomaly
score threshold to be computed. An alternative approach is to measure
the Precision using a subset of anomalies corresponding to the $k$
highest value in the anomaly score $S_T$. This is equivalent to setting
the threshold such that only the $k$ highest values are retrieved.


To address the shortcomings of the point-based quality measures, a
range-based definition was recently proposed, extending the mathematical
models of the traditional Precision and Recall.
This definition considers several factors: (i) whether a subsequence is
detected or not (ExistenceReward or ER); (ii) how many points in the
subsequence are detected (OverlapReward or OR); (iii) which part of the
subsequence is detected (position-dependent weight function); and (iv)
how many fragmented regions correspond to one real subsequence outlier
(CardinalityFactor or CF). Formally, we define $R=\{R_1,...R_{N_r}\}$ as
the set of anomaly ranges, with
$R_k=\{pos_i,pos_{i+1}, ..., pos_{i+j}\}$ and
$\forall pos \in R_k, label_{pos} = 1$, and $P=\{P_1,...P_{N_p}\}$ as
the set of predicted anomaly ranges, with
$P_k=\{pos_i,pos_{i+1}, ..., pos_{i+j}\}$ and
$\forall pos \in R_k, pred_{pos} = 1$. Then, we define ER, OR, and CF as
follows:

* $ER(R_i,P)$ is defined as follows:

 $ER(R_i,P) = 1, \text{if } \sum_{j=1}^{N_p} |R_i \cap P_j| \geq 1$

 $ER(R_i,P) = 0, \text{otherwise}$

* $CF(R_i,P)$ is defined as follows:

 $CF(R_i,P) = 1, \text{if } \exists P_i \in P, |R_i \cap P_i| \geq 1$

 $CF(R_i,P) = \gamma(R_i,P), \text{otherwise}$

* $OR(R_i,P)$ is defined as follows:

 $OR(R_i,P) = CF(R_i,P)*\sum_{j=1}^{N_p} \omega(R_i,R_i \cap P_j, \delta)$


The $\gamma(),\omega()$, and $\delta()$ are tunable functions that
capture the cardinality, size, and position of the overlap respectively.
The default parameters are set to $\gamma()=1,\delta()=1$ and $\omega()$
to the overlap ratio covered by the predicted anomaly
range. 

* **Rprecision:** Based on the above, we define:


 $Rprecision(R,P) = \frac{\sum_{i=1}^{N_p} Rprecision_s(R,P_i)}{N_p}$

 $Rprecision_s(R,P_i) = CF(P_i,R)*\sum_{j=1}^{N_r} \omega(P_i,P_i \cap R_j, \delta)$

* **Rrecall:** Based on the above, we define:

 $Rrecall(R,P) = \frac{\sum_{i=1}^{N_r} Rrecall_s(R_i,P)}{N_r}$

 $Rrecall_s(R_i,P) = \alpha*ER(R_i,P) + (1-\alpha)*OR(R_i,P)$


 The parameter $\alpha$ is user defined. The default value is $\alpha=0$. 

* **R-F-score (RF):** As described previously, the F-score combines Precision and Recall.
Similarly, we define $RF_{\beta}$, with non-negative real values for
$\beta$ as follows:

 $RF_{\beta} = \frac{(1+\beta^2)*Rprecision*Rrecall}{\beta^2*Rprecision+Rrecall}$

 As before, $\beta$ is set to 1. In this demo, $RF_1$ is referred to as RF-score.

***

### Threshold-independent AD Evaluation Measures

Until now, we introduced accuracy measures requiring to threshold the
produced anomaly score of AD methods. However, the accuracy values vary
significantly when the threshold changes. In order to evaluate a method
holistically using its corresponding anomaly score, two measures from
the AUC family of measures are used. 

* **AUC-ROC:** The
Area Under the Receiver Operating Characteristics curve (AUC-ROC) is
defined as the area under the curve corresponding to TPR on the y-axis
and FPR on the x-axis when we vary the anomaly score threshold. The area
under the curve is computed using the trapezoidal rule. For that
purpose, we define $Th$ as an ordered set of thresholds between 0 and 1.
Formally, we have $Th=[Th_0,Th_1,...Th_N]$ with
$0=Th_0<Th_1<...<Th_N=1$. Therefore, $AUC\text{-}ROC$ is defined as
follows: 

 $AUC\text{-}ROC = \frac{1}{2}\sum_{k=1}^{N} \Delta^{k}_{TPR}*\Delta^{k}_{FPR}$

 with:

 $\Delta^{k}_{FPR} = FPR(Th_{k})-FPR(Th_{k-1})$

 $\Delta^{k}_{TPR} = TPR(Th_{k-1})+TPR(Th_{k})$

* **AUC-PR:** The Area
Under the Precision-Recall curve (AUC-PR) is defined as the area under
the curve corresponding to the Recall on the x-axis and Precision on the
y-axis when we vary the anomaly score threshold. As before, the area
under the curve is computed using the trapezoidal rule. Thus, we define
AUC-PR: 

 $AUC\text{-}PR = \frac{1}{2}\sum_{k=1}^{N} \Delta^{k}_{Precision}*\Delta^{k}_{Recall}$

 with:

 $\Delta^{k}_{Recall} = Recall(Th_{k})-Recall(Th_{k-1})$

 $\Delta^{k}_{Precision} = Precision(Th_{k-1})+Precision(Th_{k})$


 A simpler alternative to approximate the area under
 the curve is to compute the average Precision of the PR curve:
 In this demo, we use the above equation to approximate AUC-PR.


'''



references_text='''


# References

***

Here are the references on which our papers and demonstration are built.

1. [n.d.]. http://iops.ai/dataset_detail/?id=10.

2. Charu C. Aggarwal. 2017. Outlier Analysis (2 ed.). Springer International
Publishing. https://doi.org/10.1007/978-3-319-47578-3

3. Subutai Ahmad, Alexander Lavin, Scott Purdy, and Zuha Agha. 2017. Unsu-
pervised real-time anomaly detection for streaming data. Neurocomputing 262
(2017), 134–147. https://doi.org/10.1016/j.neucom.2017.04.070

4. Arvind Arasu, Mitch Cherniack, Eduardo Galvez, David Maier, Anurag S Maskey,
Esther Ryvkina, Michael Stonebraker, and Richard Tibbetts. 2004. Linear road: a
stream data management benchmark. In Proceedings of the Thirtieth international
conference on Very large data bases-Volume 30. 480–491.

5. Marc Bachlin, Meir Plotnik, Daniel Roggen, Inbal Maidan, Jeffrey M. Hausdorff,
Nir Giladi, and Gerhard Troster. 2010. Wearable Assistant for Parkinson’s
Disease Patients With the Freezing of Gait Symptom. IEEE Transactions on
Information Technology in Biomedicine 14, 2 (2010), 436–446. https://doi.org/10.
1109/TITB.2009.2036165

6. Anthony Bagnall, Hoang Anh Dau, Jason Lines, Michael Flynn, James Large,
Aaron Bostrom, Paul Southam, and Eamonn Keogh. 2018. The UEA multivariate
time series classification archive, 2018. arXiv preprint arXiv:1811.00075 (2018).

7. Anthony Bagnall, Jason Lines, Aaron Bostrom, James Large, and Eamonn Keogh. 2017. The great time series classification bake off: a review and experimental
evaluation of recent algorithmic advances. Data mining and knowledge discovery
31, 3 (2017), 606–660.

8. Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang,
Danny Gutfreund, Joshua Tenenbaum, and Boris Katz. 2019. Objectnet: A large-
scale bias-controlled dataset for pushing the limits of object recognition models.
(2019).

9. Pawel Benecki, Szymon Piechaczek, Daniel Kostrzewa, and Jakub Nalepa. 2021.
Detecting Anomalies in Spacecraft Telemetry Using Evolutionary Thresholding
and LSTMs. In Proceedings of the Genetic and Evolutionary Computation Con-
ference Companion (Lille, France) (GECCO ’21). Association for Computing Ma-
chinery, New York, NY, USA, 143–144. https://doi.org/10.1145/3449726.3459411

10. Ana Maria Bianco, M Garcia Ben, EJ Martinez, and Vıctor J Yohai. 2001. Outlier
detection in regression models with arima errors using robust estimates. Journal
of Forecasting 20, 8 (2001), 565–579.

11. Ane Blázquez-García, Angel Conde, Usue Mori, and Jose A Lozano. 2021. A
Review on outlier/Anomaly Detection in Time Series Data. ACM Computing
Surveys (CSUR) 54, 3 (2021), 1–33.

12. Peter Bodik, Armando Fox, Michael J Franklin, Michael I Jordan, and David A
Patterson. 2010. Characterizing, modeling, and generating workload spikes for
stateful services. In Proceedings of the 1st ACM symposium on Cloud computing. 241–252.

13. Paul Boniol, Michele Linardi, Federico Roncallo, and Themis Palpanas. 2020.
Automated Anomaly Detection in Large Sequences. In 36th IEEE International
Conference on Data Engineering, ICDE 2020, Dallas, TX, USA, April 20-24, 2020.

14. Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas, Mohammed
Meftah, and Emmanuel Remy. 2021. Unsupervised and scalable subsequence
anomaly detection in large data series. The VLDB Journal (2021).

15. Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas, Mohammed
Meftah, and Emmanuel Remy. 2021. Unsupervised and scalable subsequence
anomaly detection in large data series. The VLDB Journal (March 2021). https:
//doi.org/10.1007/s00778-021-00655-8

16. Paul Boniol and Themis Palpanas. 2020. Series2Graph: Graph-based Subse-
quence Anomaly Detection for Time Series. PVLDB 13, 11 (2020).

17. Paul Boniol, John Paparrizos, Themis Palpanas, and Michael J Franklin. 2021.
SAND: streaming subsequence anomaly detection.

18. Loic Bontemps, James McDermott, Nhien-An Le-Khac, et al. 2016. Collective
anomaly detection based on long short-term memory recurrent neural networks.
In International Conference on Future Data and Security Engineering. Springer, 141–152.

19. Mohammad Braei and Sebastian Wagner. 2020. Anomaly detection in univariate
time-series: A survey on the state-of-the-art. arXiv preprint arXiv:2004.00433 (2020).

20. Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 2000.
LOF: Identifying Density-based Local Outliers. In SIGMOD.

21. Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 2000.
LOF: identifying density-based local outliers. ACM SIGMOD Record 29, 2 (May 2000), 93–104. https://doi.org/10.1145/335191.335388

22. Yingyi Bu, Oscar Tat-Wing Leung, Ada Wai-Chee Fu, Eamonn J. Keogh, Jian Pei,
and Sam Meshkin. 2007. WAT: Finding Top-K Discords in Time Series Database.
In SIAM.

23. Luis M. Candanedo and Véronique Feldheim. 2016. Accurate occupancy de-
tection of an office room from light, temperature, humidity and CO2 measure-
ments using statistical learning models. Energy and Buildings 112 (2016), 28–39.
https://doi.org/10.1016/j.enbuild.2015.11.071

24. Raghavendra Chalapathy and Sanjay Chawla. 2019. Deep learning for anomaly
detection: A survey. arXiv preprint arXiv:1901.03407 (2019).

25. Cody Coleman, Daniel Kang, Deepak Narayanan, Luigi Nardi, Tian Zhao, Jian
Zhang, Peter Bailis, Kunle Olukotun, Chris Ré, and Matei Zaharia. 2019. Analysis
of dawnbench, a time-to-accuracy machine learning performance benchmark.
ACM SIGOPS Operating Systems Review 53, 1 (2019), 14–25.

26. Brian F Cooper, Adam Silberstein, Erwin Tam, Raghu Ramakrishnan, and Russell
Sears. 2010. Benchmarking cloud serving systems with YCSB. In Proceedings of
the 1st ACM symposium on Cloud computing. 143–154.

27. Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan
Zhu, Shaghayegh Gharghabi, Chotirat Ann Ratanamahatana, Yanping, Bing
Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen, Gustavo Batista,
and Hexagon-ML. 2018. The UCR Time Series Classification Archive. https:
//www.cs.ucr.edu/~eamonn/time_series_data_2018/.

28. Janez Demšar. 2006. Statistical comparisons of classifiers over multiple data
sets. The Journal of Machine Learning Research 7 (2006), 1–30.

29. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009.
Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on
computer vision and pattern recognition. Ieee, 248–255.

30. Dheeru Dua, Casey Graff, et al. 2017. UCI machine learning repository. (2017).

31. Andrew F. Emmott, Shubhomoy Das, Thomas Dietterich, Alan Fern, and Weng-
Keen Wong. 2013. Systematic construction of anomaly detection benchmarks
from real data. In Proceedings of the ACM SIGKDD Workshop on Outlier Detection
and Description (ODD ’13). Association for Computing Machinery, New York,
NY, USA, 16–21. https://doi.org/10.1145/2500853.2500858

32. Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar,
and Pierre-Alain Muller. 2019. Deep learning for time series classification: a
review. Data mining and knowledge discovery 33, 4 (2019), 917–963.

33. Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier,
Daniel F Schmidt, Jonathan Weber, Geoffrey I Webb, Lhassane Idoumghar, Pierre-
Alain Muller, and François Petitjean. 2020. Inceptiontime: Finding alexnet for
time series classification. Data Mining and Knowledge Discovery 34, 6 (2020), 1936–1962.

34. Pavel Filonov, Andrey Lavrentyev, and Artem Vorontsov. 2016. Multivariate
Industrial Time Series with Cyber-Attack Simulation: Fault Detection Using an
LSTM-based Predictive Data Model. arXiv:1612.06676 [cs.LG]

35. Vincent Fortuin, Matthias Hüser, Francesco Locatello, Heiko Strathmann, and
Gunnar Rätsch. 2018. Som-vae: Interpretable discrete representation learning
on time series. arXiv preprint arXiv:1806.02199 (2018).

36. Anthony J Fox. 1972. Outliers in time series. Journal of the Royal Statistical
Society: Series B (Methodological) 34, 3 (1972), 350–363.

37. Milton Friedman. 1937. The use of ranks to avoid the assumption of normality
implicit in the analysis of variance. J. Amer. Statist. Assoc. 32 (1937), 675–701.

38. Ada Wai-Chee Fu, Oscar Tat-Wing Leung, Eamonn J. Keogh, and Jessica Lin. 2006. Finding Time Series Discords Based on Haar Transform. In ADMA.

39. Sam George. 2019 (accessed August 15, 2020). IoT Signals report: IoT’s promise
will be unlocked by addressing skills shortage, complexity and security. https:
//blogs.microsoft.com/blog/2019/07/30/.

40. Ahmad Ghazal, Tilmann Rabl, Minqing Hu, Francois Raab, Meikel Poess, Alain
Crolotte, and Hans-Arno Jacobsen. 2013. Bigbench: Towards an industry stan-
dard benchmark for big data analytics. In Proceedings of the 2013 ACM SIGMOD
international conference on Management of data. 1197–1208.

41. Markus Goldstein and Andreas Dengel. 2012. Histogram-based outlier score
(hbos): A fast unsupervised anomaly detection algorithm. KI-2012: poster and
demo track 9 (2012).

42. Jim Gray. 1993. The benchmark handbook for database and transasction systems.
Mergan Kaufmann, San Mateo (1993).

43. Scott David Greenwald. 1990. Improved detection and classification of arrhythmias
in noise-corrupted electrocardiograms using contextual information. Thesis. Mas-
sachusetts Institute of Technology. https://dspace.mit.edu/handle/1721.1/29206
Accepted: 2005-10-07T20:45:22Z.

44. Manish Gupta, Jing Gao, Charu C Aggarwal, and Jiawei Han. 2013. Outlier
detection for temporal data: A survey. IEEE Transactions on Knowledge and data
Engineering 26, 9 (2013), 2250–2267.

45. Junfeng He, Sanjiv Kumar, and Shih-Fu Chang. 2012. On the difficulty of
nearest neighbor search. In Proceedings of the 29th International Coference on
International Conference on Machine Learning (ICML’12). Omnipress, Madison,
WI, USA, 41–48.

46. Mark Hung. 2017. Leading the iot, gartner insights on how to lead in a connected
world. Gartner Research (2017), 1–29.

47. Alexander Ihler, Jon Hutchins, and Padhraic Smyth. 2006. Adaptive Event
Detection with Time-Varying Poisson Processes. In Proceedings of the 12th ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining
(Philadelphia, PA, USA) (KDD ’06). Association for Computing Machinery, New
York, NY, USA, 207–216. https://doi.org/10.1145/1150402.1150428

48. Vincent Jacob, Fei Song, Arnaud Stiegler, Bijan Rad, Yanlei Diao, and Nesime
Tatbul. 2020. Exathlon: A Benchmark for Explainable Anomaly Detection over
Time Series. arXiv preprint arXiv:2010.05073 (2020).

49. E. Keogh, T. Dutta Roy, U. Naik, and A Agrawal. [n.d.]. Multi-dataset Time-
Series Anomaly Detection Competition 2021, https://compete.hexagon-ml.com/
practice/competition/39/.

50. Eamonn Keogh, Stefano Lonardi, Chotirat Ann Ratanamahatana, Li Wei, Sang-
Hee Lee, and John Handley. 2007. Compression-based data mining of sequential
data. Data Mining and Knowledge Discovery (2007).

51. M. Kontaki, A. Gounaris, A. N. Papadopoulos, K. Tsichlas, and Y. Manolopoulos. 2011. Continuous monitoring of distance-based outliers over data streams. In 2011 IEEE 27th International Conference on Data Engineering. 135–146. https:
//doi.org/10.1109/ICDE.2011.5767923

52. Kwei-Herng Lai, Daochen Zha, Junjie Xu, Yue Zhao, Guanchu Wang, and Xia
Hu. 2021. Revisiting Time Series Outlier Detection: Definitions and Benchmarks.
In NeurIPS Track on Datasets and Benchmarks.

53. N. Laptev, S. Amizadeh, and Y. Billawala. 2015. S5 - A Labeled Anomaly Detection
Dataset, version 1.0(16M). https://webscope.sandbox.yahoo.com/catalog.php?
datatype=s&did=70

54. Tae Jun Lee, Justin Gottschlich, Nesime Tatbul, Eric Metcalf, and Stan Zdonik. 2018. Greenhouse: A zero-positive machine learning system for time-series
anomaly detection. arXiv preprint arXiv:1801.03168 (2018).

55. Zhi Li, Hong Ma, and Yongbing Mei. 2007. A Unifying Method for Outlier
and Change Detection from Data Streams Based on Local Polynomial Fitting.
In Advances in Knowledge Discovery and Data Mining (Lecture Notes in Com-
puter Science), Zhi-Hua Zhou, Hang Li, and Qiang Yang (Eds.). Springer, Berlin,
Heidelberg, 150–161. https://doi.org/10.1007/978-3-540-71701-0_17

56. Michele Linardi, Yan Zhu, Themis Palpanas, and Eamonn J. Keogh. 2020. Matrix
Profile Goes MAD: Variable-Length Motif And Discord Discovery in Data Series.
In DAMI.

57. Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. 2008. Isolation Forest. In ICDM
(ICDM).

58. Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. 2008. Isolation Forest. In 2008 Eighth IEEE International Conference on Data Mining. 413–422. https:
//doi.org/10.1109/ICDM.2008.17 ISSN: 2374-8486.

59. Yubao Liu, Xiuwei Chen, and Fei Wang. 2009. Efficient Detection of Discords
for Time Series Stream. Advances in Data and Web Management (2009).

60. Haoran Ma, Benyamin Ghojogh, Maria N. Samad, Dongyu Zheng, and Mark
Crowley. 2020. Isolation Mondrian Forest for Batch and Online Anomaly Detec-
tion. arXiv:2003.03692 [cs.LG]

61. Spyros Makridakis and Michele Hibon. 2000. The M3-Competition: results,
conclusions and implications. International journal of forecasting 16, 4 (2000), 451–476.

62. Spyros Makridakis and Evangelos Spiliotis. 2021. The M5 Competition and the
Future of Human Expertise in Forecasting. Foresight: The International Journal
of Applied Forecasting 60 (2021).

63. Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. 2018. The
M4 Competition: Results, findings, conclusion and way forward. International
Journal of Forecasting 34, 4 (2018), 802–808.

64. Pankaj Malhotra, Lovekesh Vig, Gautam Shroff, and Puneet Agarwal. 2015. Long
Short Term Memory Networks for Anomaly Detection in Time Series. (2015).

65. Pankaj Malhotra, L. Vig, Gautam M. Shroff, and Puneet Agarwal. 2015. Long
Short Term Memory Networks for Anomaly Detection in Time Series. In ESANN.

66. G.B. Moody and R.G. Mark. 2001. The impact of the MIT-BIH Arrhythmia
Database. IEEE Engineering in Medicine and Biology Magazine 20, 3 (2001), 45–50. https://doi.org/10.1109/51.932724

67. George B Moody and Roger G Mark. 1992. MIT-BIH Arrhythmia Database.
https://doi.org/10.13026/C2F305

68. M. Munir, S. A. Siddiqui, A. Dengel, and S. Ahmed. 2019. DeepAnT: A Deep
Learning Approach for Unsupervised Anomaly Detection in Time Series. IEEE
Access 7 (2019), 1991–2005. https://doi.org/10.1109/ACCESS.2018.2886457

69. Raghunath Othayoth Nambiar, Matthew Lanken, Nicholas Wakou, Forrest Car-
man, and Michael Majdalany. 2009. Transaction Processing Performance Council
(TPC): twenty years later–a look back, a look ahead. In Technology Conference
on Performance Evaluation and Benchmarking. Springer, 1–10.

70. Peter Nemenyi. 1963. Distribution-free Multiple Comparisons. Ph.D. Dissertation.
Princeton University.

71. Irene CL Ng and Susan YL Wakenshaw. 2017. The Internet-of-Things: Review
and research directions. International Journal of Research in Marketing 34, 1
(2017), 3–21.

72. ES Page. 1957. On problems in which a change in a parameter occurs at an
unknown point. Biometrika 44, 1/2 (1957), 248–252.

73. Spiros Papadimitriou, Hiroyuki Kitagawa, Phillip B Gibbons, and Christos Falout-
sos. 2003. Loci: Fast outlier detection using the local correlation integral. In Pro-
ceedings 19th international conference on data engineering (Cat. No. 03CH37405).
IEEE, 315–326.

74. John Paparrizos and Luis Gravano. 2016. k-Shape: Efficient and Accurate
Clustering of Time Series. ACM SIGMOD Record 45, 1 (June 2016), 69–76.
https://doi.org/10.1145/2949741.2949758

75. John Paparrizos, Chunwei Liu, Aaron J Elmore, and Michael J Franklin. 2020.
Debunking four long-standing misconceptions of time-series distance measures.
In Proceedings of the 2020 ACM SIGMOD International Conference on Management
of Data. 1887–1905.

76. Charlotte Pelletier, Geoffrey I Webb, and François Petitjean. 2019. Temporal
convolutional neural network for the classification of satellite image time series.
Remote Sensing 11, 5 (2019), 523.

77. Daniel Peña and Ruey S Tsay. 2021. Statistical Learning for Big Dependent Data.
John Wiley & Sons.

78. Tilmann Rabl, Christoph Brücke, Philipp Härtling, Stella Stars, Rodrigo Escobar
Palacios, Hamesh Patel, Satyam Srivastava, Christoph Boden, Jens Meiners, and
Sebastian Schelter. 2019. ADABench-Towards an Industry Standard Benchmark
for Advanced Analytics. In Technology Conference on Performance Evaluation and Benchmarking. Springer, 47–63.

79. Daniel Roggen, Alberto Calatroni, Mirco Rossi, Thomas Holleczek, Kilian Förster,
Gerhard Tröster, Paul Lukowicz, David Bannach, Gerald Pirkl, Alois Ferscha,
Jakob Doppler, Clemens Holzmann, Marc Kurz, Gerald Holl, Ricardo Chavar-
riaga, Hesam Sagha, Hamidreza Bayati, Marco Creatura, and José del R. Millàn. 2010. Collecting complex activity datasets in highly rich networked sensor
environments. In 2010 Seventh International Conference on Networked Sensing
Systems (INSS). 233–240. https://doi.org/10.1109/INSS.2010.5573462 

80. Mayu Sakurada and Takehisa Yairi. 2014. Anomaly detection using autoencoders
with nonlinear dimensionality reduction. In Proceedings of the MLSDA 2014 2nd
workshop on machine learning for sensory data analysis. 4–11. 

81. Mayu Sakurada and Takehisa Yairi. 2014. Anomaly Detection Using Autoen-
coders with Nonlinear Dimensionality Reduction. In Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis (Gold Coast,
Australia QLD, Australia) (MLSDA’14). Association for Computing Machinery,
New York, NY, USA, 4–11. https://doi.org/10.1145/2689746.2689747 

82. Bernhard Schölkopf, Robert Williamson, Alex Smola, John Shawe-Taylor, and
John Platt. 1999. Support vector method for novelty detection. In Proceedings
of the 12th International Conference on Neural Information Processing Systems
(NIPS’99). MIT Press, Cambridge, MA, USA, 582–588.

83. Pavel Senin, Jessica Lin, Xing Wang, Tim Oates, Sunil Gandhi, Arnold P. Boedi-
hardjo, Crystal Chen, and Susan Frankenstein. 2015. Time series anomaly
discovery with grammar-based compression. In EDBT.

84. Nidhi Singh and Craig Olinsky. 2017. Demystifying Numenta anomaly bench-
mark. In 2017 International Joint Conference on Neural Networks (IJCNN). IEEE, 1570–1577.

85. Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. 2019. Robust
Anomaly Detection for Multivariate Time Series through Stochastic Recurrent
Neural Network. In Proceedings of the 25th ACM SIGKDD International Conference
on Knowledge Discovery amp; Data Mining (Anchorage, AK, USA) (KDD ’19).
Association for Computing Machinery, New York, NY, USA, 2828–2837. https:
//doi.org/10.1145/3292500.3330672

86. Sharmila Subramaniam, Themis Palpanas, Dimitris Papadopoulos, Vana Kaloger-
aki, and Dimitrios Gunopulos. 2006. Online Outlier Detection in Sensor Data
Using Non-Parametric Models. In VLDB 2006. 187–198.

87. Nesime Tatbul, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. 2018. Precision and recall for time series. In Proceedings of the 32nd International
Conference on Neural Information Processing Systems. 1924–1934.

88. Nesime Tatbul, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. 2018. Precision and Recall for Time Series. In Advances in Neural Information
Processing Systems, Vol. 31. Curran Associates, Inc. https://proceedings.neurips.
cc/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html

89. Markus Thill, Wolfgang Konen, and Thomas Bäck. 2020. MarkusThill/MGAB:
The Mackey-Glass Anomaly Benchmark, https://doi.org/10.5281/zenodo.3762385.
https://doi.org/10.5281/zenodo.3762385

90. Luan Tran, Liyue Fan, and Cyrus Shahabi. 2016. Distance-Based Outlier Detec-
tion in Data Streams. Proc. VLDB Endow. 9, 12 (Aug. 2016), 1089–1100.

91. Ruey S Tsay. 1988. Outliers, level shifts, and variance changes in time series.
Journal of forecasting 7, 1 (1988), 1–20.

92. Ruey S Tsay and Rong Chen. 2018. Nonlinear time series analysis. Vol. 891. John
Wiley & Sons.

93. Ruey S Tsay, Daniel Pena, and Alan E Pankratz. 2000. Outliers in multivariate
time series. Biometrika 87, 4 (2000), 789–804.

94. Alexander von Birgelen and Oliver Niggemann. 2018. Anomaly Detection and
Localization for Cyber-Physical Production Systems with Self-Organizing Maps.
Springer Berlin Heidelberg, Berlin, Heidelberg, 55–71. https://doi.org/10.1007/978-3-662-57805-6_4

95. Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and
Samuel R Bowman. 2018. GLUE: A multi-task benchmark and analysis platform
for natural language understanding. arXiv preprint arXiv:1804.07461 (2018).

96. Frank Wilcoxon. 1945. Individual comparisons by ranking methods. Biometrics
Bulletin (1945), 80–83.

97. Frank Wilcoxon. 1945. Individual Comparisons by Ranking Methods. Biometrics
Bulletin 1, 6 (1945), 80–83. http://www.jstor.org/stable/3001968

98. Renjie Wu and Eamonn J Keogh. 2020. Current Time Series Anomaly Detection
Benchmarks are Flawed and are Creating the Illusion of Progress. arXiv preprint
arXiv:2009.13807 (2020).

99. Yuan Yao, Abhishek Sharma, Leana Golubchik, and Ramesh Govindan. 2010.
Online anomaly detection for sensor systems: A simple and efficient approach.
Performance Evaluation 67, 11 (2010), 1059–1075. https://doi.org/10.1016/j.peva.
2010.08.018 Performance 2010.

100. Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei
Ding, Hoang Anh Dau, Zachary Zimmerman, Diego Furtado Silva, Abdullah
Mueen, and Eamonn Keogh. 2018. Time series joins, motifs, discords and
shapelets: a unifying view that exploits the matrix profile. Data Mining and
Knowledge Discovery 32, 1 (Jan. 2018), 83–123. https://doi.org/10.1007/s10618-017-0519-9

101. Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei
Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, and Eamonn J.
Keogh. 2016. Matrix Profile I: All Pairs Similarity Joins for Time Series. In ICDM 

102. Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu,
Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, and Nitesh V Chawla. 2019.
A deep neural network for unsupervised anomaly detection and diagnosis in
multivariate time series data. In Proceedings of the AAAI Conference on Artificial
Intelligence, Vol. 33. 1409–1416

***


'''




text_info_page_1 = r'''
## Global Anomaly Detection Methods Evaluation

We report in the following table the average accuracy for each method on each time series (we display the boxplot of the table as well).
You may filter the table with the following parameters:

- **Dataset**: You can filter the table by dataset. The default value is 'ALL' (all datasets).
- **Measure**: You can update the accuracy measure. The default value is 'AUC_PR'.
- **Anomaly Type**: You can filter the table based on the anomaly type. The default value is 'ALL' (both point and sequence anomalies).
- **Time Series Type**: You can filter the table based on the time series type. The default value is 'ALL' (both single and multiple anomalies in the time series).


By clicking on a given row of the table, the time series, as well as the anomaly scores, will appear below it.

'''

text_info_page_2 = r'''
In this frame, you can select any pair of methods
and perform a detailed comparison. After choosing two methods,
the GUI displays a scatter plot, in which each point
corresponds to a time series. The x- and y-axes correspond to the
accuracy of the two selected methods. The color of the scatter
points depends on the dataset to which the corresponding time
series belongs. Moreover, the GUI displays a box plot
that corresponds to the overall performance of the two methods.

You can then filter the scatter plot by:
- **Dataset**: You can filter the table by dataset. The default value is 'ALL' (all datasets).
- **Measure**: You can update the accuracy measure. The default value is 'AUC_PR'.
- **Anomaly Type**: You can filter the table based on the anomaly type. The default value is 'ALL' (both point and sequence anomalies).
- **Time Series Type**: You can filter the table based on the time series type. The default value is 'ALL' (both single and multiple anomalies in the time series).

Finally, you can click on any scatter point, and the corresponding time
series will be displayed, along with the anomaly score of the two
selected methods.
'''

text_info_page_3 = r'''

## Global Accuracy Measures Evaluation

We analyze the sensitivity of different approaches quantitatively to different factors: 
(i) lag, (ii) noise, and (iii) normal/abnormal ratio. 
As already mentioned, these factors are realistic. For instance, lag can be either introduced 
by the anomaly detection methods (such as methods that produce a score per subsequences are 
only high at the beginning of abnormal subsequences) or by human labeling approximation. 
Furthermore, even though lag and noises are injected, an optimal evaluation metric should 
not vary significantly. Therefore, we aim to measure the variance of the different evaluation 
measures when we vary the lag, noise, and the normal/abnormal ratio. Thus, we proceed as follows:


- For each anomaly detection method, we first compute the anomaly score on a given time series.
- We then inject either lag $l$, noise $n$ or change the normal/abnormal ratio $r$. For 10 different values of $ l \in [-0.25*\ell,0.25*\ell]$, $n \in [-0.05*(max(S_T)-min(S_T)),0.05*(max(S_T)-min(S_T))]$ and $r \in [0.01,0.2]$, we compute the 13 different evaluation measures.
- For each evaluation measure and each AD methods, we compute the standard deviation of the ten different values. 
- We compute the average standard deviation (across all AD methods) for the 13 different AD quality measures. 
- We compute the average standard deviation for every time series in each dataset.
- We compute the average standard deviation for every dataset. You can choose in the first dropdown wich dataset you want to analyze. ALL corresponds to the avertage on all time series of all datasets


The final boxplot/barplot corresponds to the sensitivity to lag/noise or ratio of a given evaluation measures on either ALL or one specific dataset.

## Create your own experiment

In this section, you are free to choose the settings of the previous global experiment. 
You can choose which time series to use, which AD methods, and wich sensitivity (lag, noise or ratio) to measure. 
The evolution of AD measures values will be displayed when applied on the chosen time series and the chosen AD method.


'''






