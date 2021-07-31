from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot
import numpy as np

scores1 = [0.8440944881889764, 0.8204724409448819, 0.8598425196850393, 0.8881889763779528, 0.8850393700787401,
           0.8866141732283465, 0.8976377952755905, 0.8944881889763779, 0.8582677165354331, 0.8818897637795275]


scores2 = [0.9432314410480349, 0.9461426491994177, 0.9374090247452693, 0.9243085880640466, 0.9694323144104804,
           0.9374090247452693, 0.9213973799126638, 0.9387755102040817, 0.9110787172011662, 0.9402332361516035]

print(np.mean(scores1), np.std(scores1))
print(np.mean(scores2), np.std(scores2))



pyplot.boxplot([scores1, scores2], labels=['Decision Tree', '1D CNN'], showmeans=True)
pyplot.xlabel("Model")
pyplot.ylabel("Accuracy")
pyplot.show()