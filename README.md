# TMFN-Text-Guided-Mamba-Fusion-Network-for-Efficient-Multimodal-Sentiment-Analysis
TMFN is a fusion network for efficient multimodal sentiment analysis, it consists a parameter-efficient adapter, a mamba-based single-layer encoder, a text-guided cross-attention, and employ text modalities as a semantic anchor in subsequent fusion stage. The architecture of TMFN is as follows：
![the overall architecture of the TMFN](Overall architecture.pdf)
The link of the TMFN's paper is as follows:

The next part is a readme that tell you how to use our code to achieve the TMFN, if you have any question pls contact me.

___
1. Explian of the program construction
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
