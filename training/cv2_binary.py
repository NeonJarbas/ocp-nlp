from ovos_utils.log import LOG
from ocp_nlp.classify import BinaryPlaybackClassifier

s_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_sentences_v0.csv"

LOG.set_level("DEBUG")
# download datasets from https://github.com/NeonJarbas/OCP-dataset

o = BinaryPlaybackClassifier()  # english only Accuracy 0.99
o.find_best_Perceptron(s_path)


# o.search_best(s_path)
# Accuracy: 0.9869315226246796
#               precision    recall  f1-score   support
#
#          OCP       0.99      0.98      0.98      4839
#        other       0.99      1.00      0.99      8657
#
#     accuracy                           0.99     13496
#    macro avg       0.99      0.99      0.99     13496
# weighted avg       0.99      0.99      0.99     13496
#o.load()

preds = o.predict(["play a song", "play my morning jams",
                   "i want to watch the matrix",
                   "tell me a joke", "who are you", "you suck"])
print(preds)  # ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']
