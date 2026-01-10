# TMFN-Text-Guided-Mamba-Fusion-Network-for-Efficient-Multimodal-Sentiment-Analysis
TMFN is a fusion network for efficient multimodal sentiment analysis, it consists a parameter-efficient adapter, a mamba-based single-layer encoder, a text-guided cross-attention, and employ text modalities as a semantic anchor in subsequent fusion stage. The architecture of TMFN is as follows：
![img.png](overall_architecture.png)


The link of the TMFN's paper is as follows:


The next part is a readme that tell you how to use our code to achieve the TMFN, if you have any question pls contact me.


# 1. Explian of the program construction
```text
TMFN/
├── configs/                # Save the hyperparameter settings
│   └── default.yaml        # Change the hyperparameters here
├── models/                 # The core of the TMFN
│   ├── cross_attn_encoder.py  # The fusion module
│   └── multimodal.py       # Full construction of the model
├── utils/                  # Utility scripts for data processing
│   ├── extract_audio.py
│   ├── extract_audio_ur_funny.py
│   ├── extract_video.py
│   ├── Extract_video_ur_funny.py
│   ├── generate_label4urfunny.py
│   ├── metricsTop.py
│   └── plot.py
├── dataloader.py           # Data loading logic
├── trainer.py              # Model training logic
├── run_experiment.py       # Main procedure entrance
├── requirements.txt        # Dependency list
└── README.md
```


# 2. Dataset download
If you do not have CMU-MOSEI, CH-SIMS, and UR-FUNNY dataset, you could donwload them as follows link:


CMU-MOSEI: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git


CH-SIMS: https://github.com/thuiar/MMSA.git


UR-FUNNY: https://github.com/ROC-HCI/UR-FUNNY.git


# 3. Environment setting
You could see the environment of the TMFN in the "requirements.text".

And install them by ``` pip install requirements.txt ```

# 4. Data preprocess
You have to preprocess the raw data to split the video and visual feature.


For CMU-MOSEI, CH-SIMS, You could use "extract_audio.py" and "extract_video.py" to process the raw data.


For UR-FUNNY, it consists of two steps. Firstly, you will extract the feature by "extract_audio_ur_funny.py" and "extract_video_ur_funny.py". Then, you have to rebuild the label of the ur-funny dataset through "generate_label4urfunny.py"


# Please attention: befor you run the code, please set and check the path of your dataset location.
# 5. 
