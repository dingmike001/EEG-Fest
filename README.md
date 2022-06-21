# EEG-Fest
This repo contains a source code for EEG-Fest.The EEG-Fest model, as shown in Figure, is composed of a feature extraction module, a feature relation module, and a decision module.   
![framework](https://github.com/dingmike001/EEG-Fest/blob/main/img/Framework.png?raw=true)

# Dataset Information
## SEED-VIG
The dataset was collected in a simulated driving system. 23 subjects take part in the experiment. Each trial in the dataset includes EEG signals of a subject in the whole experiment process. The experiment duration is 2 hours. 17 channels' EEG signals are recorded from every subject and sampled at 200Hz. This dataset uses PERCLOS as labels. Eye closing time is recorded every 8 seconds interval time to calculate PERCLOS. A threshold of 0.7 is used to indicate if a subject is drowsy.[1] We divide each trail into 885 samples, each sample contains 8-second length of EEG data.  
![perclos](https://github.com/dingmike001/EEG-Fest/blob/main/img/perclos.png?raw=true)

## Dataset from Sustained-Attention Driving Task (SADT)
27 subjects' EEG data are included in this dataset. Every subject was asked to drive in a simulated driving system to keep the virtual car in the center of the lane. The virtual environment produces lane-departure events randomly making the car drift from the original path. The moment the subject makes the response to the drift car is recorded as a counter-steering event. The time between the beginning of the lane-departure event and the start of the counter-steering event is recorded as react time[2]. We can get the drowsiness index by inputting react time into equation. It indicates the subject is drowsy if the drowsiness index is near 1[3].32 channels' EEG signals are recorded from every subject. Moving average filter with a window length of 10 is used to smooth the drowsiness index. Every sample contains 3-second length EEG data, which starts from the lane-departure event. The sample rate is 500Hz.  
![drowsy_index](https://github.com/dingmike001/EEG-Fest/blob/main/img/index.png?raw=true)

## SEED
This dataset is not collected in an experiment under driving conditions. 15 subjects in the experiment were required to watch different movie clips, which lead to subjects' positive or negative emotions. 62 channels' EEG data were collected from every subject and used to analyze the subject's emotion change caused by movie clips\cite{zheng2015investigating}. In our work, we use the SEED dataset as Non-Driving EEG data input into the EEG-Fest to see if the EEG-Fest could distinguish it from driving EEG data. Its sample rate is 200Hz.

Since EEG data includes artefact information of subjects' muscular motion or ocular motion, we need to remove these artefacts to avoid abnormal results caused by them. The Automatic Artifact Removal (AAR) plug-in for EEGLAB\cite{eeglab2022}, a MATLAB toolbox, is used to do such work.

## Reference
[1] Wei-Long Zheng and Bao-Liang Lu. A multimodal approach to estimating vigilance using eeg and forehead eog. Journal of neural engineering, 14(2):026017, 2017.
[2] Zehong Cao, Chun-Hsiang Chuang, Jung-Kai King, and Chin-Teng Lin. Multi-channel eeg recordings during a sustained-attention driving task. Scientific data, 6(1):1–8, 2019.
[3] Dongrui Wu, Vernon J Lawhern, Stephen Gordon, Brent J Lance, and Chin-Teng Lin. Driver drowsiness estimation from eeg signals using online weighted adaptation regularization for regression (owarr). IEEE Transactions on Fuzzy Systems, 25(6):1522–1535, 2016.
