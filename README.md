Emotion Prediction using Facial Expressions


Emotion state prediction using facial expressions is one of the subcomponents to be addressed in 
this project, with applications ranging from improving human-computer interaction to assisting 
in mental health diagnosis. However, accurately predicting an individual's emotional state solely 
from facial expressions remains a difficult challenge. To address this, the study intends to 
develop a robust face emotion prediction model using deep learning techniques.
Another challenge that can be identified is enhancing accuracy by considering an individual's 
gender when predicting emotional states from facial expressions. The main key concern 
regarding Gender-Related Expression Variability is that individuals of different genders may 
exhibit variations in their facial expressions when experiencing the same emotion.
Hence, examining the impact of gender variations in Facial Expression Recognition (FER) 
training can lead to an improved understanding of the significance of specific facial regions. This 
knowledge can then be applied to the creation of improved models, ultimately affecting FERbased applications.
Moreover, deep learning models require high-quality and diverse datasets. However, research in 
this field often relies on existing datasets, which may not always have a balanced representation 
of gender. Hence, the challenge of finding separate datasets for males and females is a critical 
issue.


Objective:
Developed a predictive model which utilizes facial expressions to predict a individual's emotional 
state. The model should take a multimodal approach, taking into account the user's age and 
gender as extra considerations.
- Developed a model to predict emotional state from facial expressions without considering 
        gender.
- Developed separate models using gender-specific datasets for males and females.
- Experiment with gender-based and general model development using two technologies.
- Analyzing results do model comparison to identify the impact of gender for facial expression 
        recognition.




To identify the impact of gender variations in Facial Expression Recognition (FER) training, 
in this study is structured around two primary methodologies, initially develop a model for predicting 
emotional states from facial expressions without considering gender. And then develop two 
models as male FER and female FER using gender-specific datasets for males and females.



• Facial expression recognition without considering gender.

This approach involves training the model using the standard fer2013 and CK+ image dataset, 
which includes both female and male images. The dataset was initially loaded, and preprocessing 
steps were implemented to enhance its suitability for training. During the preprocessing phase, 
pixel value normalization and label encoding for emotion categories were carried out. 
Additionally, class weights and data augmentation techniques were applied to address class 
imbalances effectively. This prepared the dataset for subsequent model training.
Model training is conducted using two deep learning technologies: Convolutional Neural 
Network (CNN) and Residual Neural Network (ResNet)


  
Facial expression recognition based on gender.

In the second approach, training involves considering gender as a variable. During this phase, the 
FER2013 and CK+ datasets are partitioned into two distinct subsets, manually separated based 
on instances of males and females. The preprocessing and other steps before model training in 
this approach are the same as those in the general model. This approach also conducting model 
training using two deep learning technologies: Convolutional Neural Network (CNN) and 
Residual Neural Network (ResNet).
And then, from the results obtained from the two approaches, a comparative analysis of the 
accuracies of the models will be conducted. The study will subsequently analyze the results and 
perform model comparisons to identify the impact of gender on facial expression recognition.
Furthermore, conducting experiments with gender-based and general model development using 
two technologies, namely Convolutional Neural Networks (CNN) and Residual Networks 
(ResNet), is expected to produce more accurate results for the comparative analysis. In addition, 
these experiments will facilitate a technology comparison to evaluate the effectiveness and 
performance of predicting emotions from facial expressions.

 overview diagram of gender based facial expression recognition :


  <img width="449" alt="image" src="https://github.com/kaveendhya/Emotion-prediction-from-Facial-Expression/assets/88360228/a7954df4-ecd0-4db7-9a84-3c614fea59e4">


Result:

•	Comparative analysis of gender influence

![image](https://github.com/kaveendhya/Emotion-prediction-from-Facial-Expression/assets/88360228/0730140b-a41c-474e-bff9-94235f220f48)


Analyzing results of CNN model, average accuracy of gender-based model (61.96%.) is higher than the general model which does not consider the gender (57.74%). Applying the ResNet, average accuracy of gender-based model (66.76%.) is also higher than the general model which does not consider gender (66.06%).
In comparison, the outcomes from both the CNN and ResNet methodologies highlight that gender-based considerations play a significant role in the accuracy of facial expression prediction models.

