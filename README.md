# Distaster Response Classification App

### Udacity Data Pipelines Course
The following project was performed in the course of one of required submissions during the Udacity Data Scientist Nanodegree Programm.
***The last commit `8be106c2e694774ecc8e06c6f8cc52b4512ceb83` 
contains files used to deploy the final application to the web app provider `Heroku`***
***- https://disaster-response-project-jn.herokuapp.com/***

***If you want to run the application locally please download the zip file of commit `9b8f917319e847b411a09053181f23134ac15ae4.` 
The link to the commit:
https://github.com/JacekNiedzielski/Distaster_Response_Pipeline/tree/9b8f917319e847b411a09053181f23134ac15ae4
Then proceed with chapter `Required libraries and local installation instructions`***

#### The aim of this project is to provide a predictive tool beeing able to classify the real world messages (just like these posted on social media or provided in news) into disaster categories. Such a tool could potentially function as a core part of more sophisticated concept, where the application automatically informs appropriate services about the particular disaster event/events.

#### Application developed in this repository is based on the machine learning algorithm of decision tree beeing embeded in the sckit-learn multiouptut classifier. I have tried multiple different algorithms of supervised learning classifiers, but it occured, that the tree gives the best recall results for the case of an actually present disaster event (its True value). In another words, the decision tree used in this project, intercepts relatively highest number of true disasters, so that the possibility of not providing help for somebody who actually needs it is minimised. The tradeoff for that is depecited in lower precision of the tool. The tree has been embeded in the multioutputclassifier, because of the possibility of multiple category choices. However it could perform also as standalone (decision tree is capable of performing multioutput classification)

#### The tokenization and lemmatization of messages which have been used to train the model has been performed on the basis of natural language processing libraries called `nltk.` 

#### The data set used to train and evaluate the model has been provided by figure8. The set comprises of ~26k messages of different lengths, which have been classified into 36 categories. Since the data set came as highly inbalanced, I performed some class weight adjustments during the model training. However these measures could increase the performance only to the particular level. For better scoring one would need to incorporate more input data, especially for undersampled categories.

### More detailed information and insights regarding data processing and machine learning techniques used during this project can be found in the following files:
- Machine Learning Pipeline.ipynb
- ETL_Pipeline.ipynb
##### Both files can be found under the following commit: htps://github.com/JacekNiedzielski/Distaster_Response_Pipeline/tree/9b8f917319e847b411a09053181f23134ac15ae4


### Required libraries and local installation instructions

The code has been written in python 3.9.4

Furthermore, you will need the following libraries:
- json
- pandas
- plotly
- nltk
- flask
- joblib
- sqlalchemy
- sklearn
- gunicorn

After you will have your libraries installed, you will need to perform the following to process the data used for the application, train the machine learning model and run the application locally. 

1. Download the zip file of the commit https://github.com/JacekNiedzielski/Distaster_Response_Pipeline/tree/9b8f917319e847b411a09053181f23134ac15ae4
2. Unpack the files and open the terminal window in the main folder.
3. Type the following commands to execute data processing (ETL Pipeline): `python process_data.py messages.csv categories.csv disaster_database.db`
4. Type the following commands to execute model's training (ML Pipeline): `python train_classifier.py disaster_database.db class.pkl`
5. Type the following command to run the application:                     `python run.py`
6. Type localhost:3001 in the omnibox of your browser


### How to use the app? 

Firstly - it is important to get know with categories' occurences within the training data set. As I mentioned before - I had tried to balance the classes, but still one can assumme lower peformance when trying to classify messages refering to the topics which occured very seldom. Please compare the picture below:

![image](https://user-images.githubusercontent.com/64994740/139306599-707b4dbc-6852-4388-a074-b23e834a3f09.png)


After this short glance please type the message you want to classify. The machine learning algorithm will output the reference categories. Below the example output:
![image](https://user-images.githubusercontent.com/64994740/139308951-71eef84b-6bb1-412a-951d-4ac8e8a0f966.png)




### Acknowledgements

Figure8 - data set <br/>
Udacity - templates and mentoring





Feel free to use the provided code!


