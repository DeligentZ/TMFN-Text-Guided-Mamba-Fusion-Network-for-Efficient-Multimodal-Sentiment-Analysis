# TMFN-Text-Guided-Mamba-Fusion-Network-for-Efficient-Multimodal-Sentiment-Analysis
TMFN is a fusion network for efficient multimodal sentiment analysis, it consists a parameter-efficient adapter, a mamba-based single-layer encoder, a text-guided cross-attention, and employ text modalities as a semantic anchor in subsequent fusion stage.

The link of the TMFN's paper is as follows:

The next part is a readme that tell you how to use our code to achieve the TMFN, if you have any question pls contact me.

___

1. Explian of the program construction
    TMFN/
    ├── configs/                       # save the hyperparameter's setting. 
    │   └── default.yaml               # change the hyperparameters at here.
    ├── models/                        # the core of the TMFN
    │   ├── cross_attn_encoder.py      # the fusion module
    │   └── multimodal.py              # all the construction of the model
    ├── utils/                         
    │   ├── extract_audio.py
    │   ├── extract_audio_ur_funny.py
    │   ├── extact_video.py
    │   ├── Extract_video_ur_funny.py
    │   ├── generate_label4urfunny.py
    │   ├── metricsTop.py
    │   └── plot.py
    ├── dataloader.py                 
    ├── requirements.txt              # requirements list
    ├── trainer.py                    
    ├── run_experiment.py             # main procedure entrance
    └── README.md
   
