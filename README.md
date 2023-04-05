# Continual Learning with Imagination for Facial Expression Recognition

This is a Keras-Tensorflow implementation for the [Continual Learning with Imagination for Facial Expression Recognition](https://ieeexplore.ieee.org/document/9320226) paper published at the 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020). 
The paper proposes a novel framework for Continual Learning (CL)-based personalised Facial Expression Recognition. 


## Installation and Data Requirements

### Dependencies
You can install all dependencies using the following:
```
pip install -r requirements.txt
```

### Data
Data processing protocols can be found in the paper. You can request access to the different datasets as follows:

RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

MMI: https://mmifacedb.eu/

BAUM1: https://archive.ics.uci.edu/ml/datasets/BAUM-1

Once downloaded, the datasets should be placed at the following path:
```
data/
```

### Trained Models

The trained Conditional Adversarial Auto-Encoder (CAAE)-based Imagination model can be found [here](https://drive.google.com/drive/folders/1QFH_Ymq3EjQYWdbrKKktVSIwzDcXlhX8?usp=sharing). Once downloaded they should be placed at the following path:
```
Models/Trained_Model/
```

CLIFER training also uses the VGG-Face descriptor for a customised loss function. The VGG-Face descriptor (mat) can be accessed [here](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/). Once downloaded it should placed at the following path:
```
Models/VGG/
```

## Abstract
Current Facial Expression Recognition (FER) approaches tend to be insensitive to individual differences in expression and interaction contexts. They are unable to adapt to the dynamics of real-world environments where data is only available incrementally, acquired by the system during interactions. In this paper, we propose a novel continual learning framework with imagination for FER (CLIFER) that (i) implements imagination to simulate expression data for particular subjects and integrates it with (ii) a complementary learning-based dual-memory (episodic and semantic) model, to augment person-specific learning. The framework is evaluated on its ability to remember previously seen classes as well as on generalising to yet unseen classes, resulting in high F1-scores for multiple FER datasets: RAVDESS (episodic: F1=0.98 ± 0.01, semantic: F1=0.75 ± 0.01), MMI (episodic: F1=0.75 ± 0.07, semantic: F1=0.46 ± 0.04) and BAUM-I (episodic: F1=0.87 ± 0.05, semantic: F1=0.51 ± 0.04).

## Citation

```
@INPROCEEDINGS{Churamani2020CLIFER,  
  author		= {N. {Churamani} and H. {Gunes}},  
  booktitle		= {Proceedings of the 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG)},   
  title			= {{CLIFER: Continual Learning with Imagination for Facial Expression Recognition}},   
  year			= {2020},  
  publisher		= {IEEE},
  pages			= {322--328},  
  doi			= {10.1109/FG47880.2020.00110}
}
```

## Acknowledgement
**Funding:** The work of Nikhil Churamani and Hatice Gunes is supported by EPSRC under grants ref: EP/R513180/1 and EP/R030782/1, respectively. The authors also thank German I. Parisi for his insights on the GDM model.
