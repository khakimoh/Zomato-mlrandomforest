# Zomato-MLRandomForest-Article
Predict the user rating of Zomato restaurant with machine learning from Random Forest classification data - final project bachelor.

Social media provides a wealth of information for decision-making, especially for finding restaurants in unfamiliar areas. This study leverages Zomato restaurant data to unveil latent structures and hidden patterns within the data that influence customer ratings. By incorporating machine learning, we create a personalized recommendation system to guide users towards their ideal dining experience. To overcome the limitations of raw data, we employ a multi-step approach. Clustering algorithms unveil hidden patterns (latent structures) in user preferences and restaurant attributes. These structures improve the accuracy of our classification models, addressing the challenge of complex relationships between seemingly unrelated data points. 
Our findings highlight the effectiveness of the Random Forest classification algorithm. Applied after the multi-step approach that unveils latent structures, it achieves a remarkable 90% accuracy rate. This success is demonstrably linked to uncovering hidden patterns in user preferences and restaurant attributes through clustering. These structures allow the Random Forest model to make more precise classifications, ultimately leading to a superior recommendation system. Notably, most errors involve misclassifications between similar restaurant types, which is acceptable in this context due to the inherent overlap in user preferences for these categories.


### Table content: The steps will generally be as follows:
- Extract and convert data to standard CSV
- Normalize many data
- The information is divided into two segment
- Clustering feature with Hierarchical method unsupervised 
- Clustering feature with K-means method unsupervised
- Set label with CoEfficient
- Merging clusters and sticking the label to data about delivery and classy ambiance
- Over sampling
- Feature selection
- Compare methods
- Random Forest Classifier
- Classification data for predicting user ratting
- Multiple ReClassification
- Multi-label Classification
- Multi-Class Classification
- Improve The Model
- Predict a new restaurant score
- Analysis functionality method be run