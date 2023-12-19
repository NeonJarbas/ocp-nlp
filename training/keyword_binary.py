from ovos_utils.log import LOG
from ocp_nlp.classify import KeywordBinaryPlaybackClassifier

ents_csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_entities_v0.csv"
s_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_sentences_v0.csv"
t_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_playback_type_v0.csv"
csv_small_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_small_v0.csv"
csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"
csv_big_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_big_v0.csv"

LOG.set_level("DEBUG")
# download datasets from https://github.com/NeonJarbas/OCP-dataset

o = KeywordBinaryPlaybackClassifier()  # lang agnostic
#o.find_best_Perceptron(s_path)
#  {'alpha': 0.0001, 'early_stopping': True, 'l1_ratio': 0.15, 'penalty': 'l1'}
# Accuracy: 0.8845729092225357
#               precision    recall  f1-score   support
#
#          OCP       0.82      0.87      0.85      4839
#        other       0.93      0.89      0.91      8657
#
#     accuracy                           0.89     13496
#    macro avg       0.88      0.88      0.88     13496
# weighted avg       0.89      0.89      0.89     13496

o.find_best_MLP(s_path)
# Best parameters found:
#  {'solver': 'adam', 'learning_rate': 'constant',
#  'hidden_layer_sizes': (50, 50, 50), 'early_stopping': False,
#  'alpha': 0.0001, 'activation': 'tanh'}
# Accuracy: 0.9331252157522353
#               precision    recall  f1-score   support
#
#          OCP       0.94      0.90      0.92      4839
#        other       0.94      0.97      0.96      8657
#
#     accuracy                           0.94     13496
#    macro avg       0.94      0.93      0.94     13496
# weighted avg       0.94      0.94      0.94     13496



o.load()

preds = o.predict(["play a song", "play my morning jams",
                   "i want to watch the matrix",
                   "tell me a joke", "who are you", "you suck"])
print(preds)  # ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']
