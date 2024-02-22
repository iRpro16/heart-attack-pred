# heart-attack-pred
***
## Software and Tool Requirements
1. [iRpro16](https://github.com/iRpro16/heart-attack-pred)
2. [Streamlit](https://www.streamlit.io)
3. [Kaggle](https://www.kaggle.com)
4. [VSCodeIDF](https://code.visualstudio.com)
5. [GitCLI](https://git-scm.com)

## What I have learned
This project allowed to have learned the process of machine learning projects better.
At first, I tried to use neural networks with softmax regression. It scored very well on
both training and testing data, but failed to generalize well. So, I learned that with
tabular data, it is better to stick with tradional ML classification techniques. Whereas,
for this project I used XGBClassifier.

Another issue this project has shed light on was the importance of a balanced dataset.
My model has been hyperparameter tuned and scored well on metrics, but I had eventually
come to the conclusion that my dataset was quite imbalanced. I had many examples of 
"typical angina" and "non-anginal pain", but both "atypical angina" and "asymptomatic" 
classes were minority groups.

This is when I had to use the "class_weights" paramater to add heavier weights on 
minority classes. This helped the model classify a bit better. Needless to say, 
this was quite the learning curve as I continue to code using OOP and classes
for more structured code and re-usability.