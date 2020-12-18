# Deep learning-based classification of the mouse estrous cycle stages, K. Sano and etal, 2020, Scientific Reports.
https://www.nature.com/articles/s41598-020-68611-0

# SECREIT
Mouse Estrous Cycle Estimation

### Prerequisites

- Python 3.6
- numpy 1.19.1
- opencv-python 4.1.1
- keras (tensorflow backend)== 2.2.4 (NOT 2.3)
- tensorflow 2.2.0

## Overview
![Overview](https://github.com/SanoKyohei/Secreit/blob/master/Example/Overview.png)  

## Model AUC (Performance)
#### SECREIT correctly classified competitive to two professionals
#### diestrus stage (D): 0.982 
#### estrus stage (E): 0.979
#### proestrus stage (P): 0.962

## Weight parameter
https://opac.ll.chiba-u.jp/da/curator/108041/weights.hdf5

## Dataset
### Example of file name
 "D_a_14_e1_Auto2_train.png" 
 <br> ①　D: Estrous Stage. When the estrous stage is intermediate class, we described it using "_", such as "D_E".
 <br> ②　a_14_e1: image id
 <br> ③　Auto2: Name of Experiment. Auto2 and Auto3 was used for trainig. Auto1 was for validation and Auto4 is for test. 
### Dataset download
<br> https://opac.ll.chiba-u.jp/da/curator/108041/D_part1.zip  <br> 
<br> https://opac.ll.chiba-u.jp/da/curator/108041/D_part2.zip <br> 
<br> https://opac.ll.chiba-u.jp/da/curator/108041/D_part3.zip  <br>
<br> https://opac.ll.chiba-u.jp/da/curator/108041/E.zip <br>
<br> https://opac.ll.chiba-u.jp/da/curator/108041/P.zip <br>



