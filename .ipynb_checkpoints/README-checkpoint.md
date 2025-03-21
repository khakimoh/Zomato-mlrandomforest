# Zomato-MLRandomForest-Article
Predict the user rating of Zomato restaurant with machine learning from Random Forest classification data - final project bachelor.

Social networks have facilitated access to a wealth of information, which can provide valuable insights for decision-making. The study objective is to evaluate the impact of various restaurant aspects on overall customer ratings. By utilizing Zomato restaurant data, the research aids users in identifying their preferred dining establishments in unfamiliar locales. The integration of machine learning techniques is highlighted in this paper to help restaurants improve customer service by prioritizing influential parameters. For this purpose, data preprocessing is implemented to normalize numerical data and remove redundant and outlier data. Then, an augmented dataset is created through data clustering to uncover latent data structures and improve classification quality. In addition, oversampling is employed to increase the number of samples. In order to identify attributes with the strongest correlations, feature selection methods are implemented. After all, some machine learning algorithms are utilized to classify data. Notably, the Random Forest algorithm achieves an accuracy of 71%, with most detection errors occurring when the target class is misclassified as an adjacent class.
Given the similarity between neighboring classes in user recommendations, this misclassification can be interpreted as valid, resulting in a remarkable 97% accuracy improvement.
In contrast, the similarity between neighboring classes is high in user recommendations. The area under the Receiver Operating Characteristic curve guarantees the algorithm's high power and reliability.

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

