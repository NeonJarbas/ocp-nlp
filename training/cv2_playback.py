from ovos_utils.log import LOG
from ocp_nlp.classify import PlaybackTypeClassifier

t_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_playback_type_v0.csv"

LOG.set_level("DEBUG")
# download datasets from https://github.com/NeonJarbas/OCP-dataset


# basic text only classifier
clf1 = PlaybackTypeClassifier()
#clf1.search_best(t_path)
# Accuracy: 0.8629571152149005
#               precision    recall  f1-score   support
#
#        audio       0.92      0.94      0.93      2581
#     external       0.97      0.70      0.82       240
#        video       0.94      0.94      0.94      3328
#
#     accuracy                           0.93      6149
#    macro avg       0.94      0.86      0.90      6149
# weighted avg       0.93      0.93      0.93      6149
#
clf1.load()

label, confidence = clf1.predict(["play internet radio"], probability=True)[0]
print(label, confidence)  # audio 0.9364632959879458

label, confidence = clf1.predict(["watch kill bill"], probability=True)[0]
print(label, confidence)  # video 0.9077102214166061
