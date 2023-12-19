from ovos_utils.log import LOG
from ocp_nlp.classify import KeywordPlaybackTypeClassifier

ents_csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_entities_v0.csv"
s_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_sentences_v0.csv"
t_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_playback_type_v0.csv"
csv_small_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_small_v0.csv"
csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"
csv_big_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_big_v0.csv"

LOG.set_level("DEBUG")
# download datasets from https://github.com/NeonJarbas/OCP-dataset

# keyword biased classifier
clf = KeywordPlaybackTypeClassifier()  # lang agnostic

clf.find_best_MLP(t_path)
# Best parameters found:
#  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (10, 30, 10),
#  'early_stopping': False, 'alpha': 0.005, 'activation': 'tanh'}
# Accuracy: 0.9128171600286484
#               precision    recall  f1-score   support
#
#        audio       0.92      0.93      0.92      2586
#     external       0.90      0.87      0.88       240
#        video       0.95      0.94      0.94      3327
#
#     accuracy                           0.93      6153
#    macro avg       0.92      0.91      0.92      6153
# weighted avg       0.93      0.93      0.93      6153




clf.load()

# klownevilus is an unknown entity
labels = clf.predict_labels(["play klownevilus"])[0]
print(clf.predict(["play klownevilus"])[0])
old = labels["audio"]
old2 = labels["video"]

# probability increases for movie
clf.register_entity("movie_name", ["klownevilus"])

labels2 = clf.predict_labels(["play klownevilus"])[0]
print(clf.predict(["play klownevilus"])[0])
n = labels2["audio"]
n2 = labels2["video"]

print("bias changed audio confidence: ", old, "to", n)
print("bias changed video confidence: ", old2, "to", n2)
# bias changed movie confidence:    to
# bias changed music confidence:   to

assert n > old
assert n2 < old2

from pprint import pprint

pprint(labels)
pprint(labels2)