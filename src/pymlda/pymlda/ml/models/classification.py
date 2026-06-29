from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

class ClassifierFactory:
    
    @staticmethod
    def get(model_name="svm"):
        
        if model_name == "svm":
            return SVC(kernel='linear', C=100, decision_function_shape='ovo')
        
        if model_name == "knn":
            return KNeighborsClassifier(n_neighbors=3)
        
        if model_name == "rf":
            return RandomForestClassifier()
        
        if model_name == "dt":
            return DecisionTreeClassifier()
        
        if model_name == "nb":
            return GaussianNB()
        
        if model_name == "xgb":
            return xgb.XGBClassifier(objective='multi:softmax')
        
        raise ValueError("Model not supported")