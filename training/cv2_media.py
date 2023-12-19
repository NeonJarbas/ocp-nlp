from ovos_utils.log import LOG
from ocp_nlp.classify import MediaTypeClassifier

ents_csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_entities_v0.csv"
s_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_sentences_v0.csv"
t_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_playback_type_v0.csv"
csv_small_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_small_v0.csv"
csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"
csv_big_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_big_v0.csv"

LOG.set_level("DEBUG")
# download datasets from https://github.com/NeonJarbas/OCP-dataset


# basic text only classifier
clf1 = MediaTypeClassifier()
# clf1.search_best(csv_path, test_csv_path=csv_big_path)

# CalibratedClassifierCV(estimator=Perceptron())
# Accuracy: 0.881016339627877
#               precision    recall  f1-score   support
#
#           ad       0.99      0.95      0.97      1200
#        adult       0.90      0.79      0.84       672
#   adult_asmr       1.00      0.94      0.97      1139
#        anime       0.96      0.59      0.73       976
#         asmr       1.00      0.93      0.97       769
#        audio       0.94      0.98      0.96      1032
#    audiobook       1.00      0.96      0.98      1175
#          bts       0.99      0.98      0.99       754
#     bw_movie       1.00      1.00      1.00       832
#      cartoon       0.82      0.89      0.85       904
#        comic       0.98      0.87      0.92       635
#  documentary       0.77      0.76      0.76      1004
#         game       0.97      0.97      0.97      1194
#       hentai       0.85      0.65      0.74       910
# iot_playback       1.00      0.95      0.98      1129
#        movie       0.70      0.99      0.82      1200
#        music       0.67      1.00      0.80      1200
#         news       0.99      0.98      0.98       877
#      podcast       0.81      0.87      0.84      1196
#        radio       0.97      0.88      0.92       593
#  radio_drama       0.78      0.81      0.80      1145
#       series       0.84      0.95      0.89      1188
#   short_film       0.96      0.61      0.75       355
# silent_movie       1.00      0.93      0.96       942
#      trailer       1.00      0.82      0.90       499
#   tv_channel       0.99      0.77      0.87      1173
#        video       0.86      0.95      0.90      1093
#
#     accuracy                           0.89     25786
#    macro avg       0.92      0.88      0.89     25786
# weighted avg       0.91      0.89      0.89     25786
#

clf1.load()


label, confidence = clf1.predict(["play metallica"], probability=True)[0]
print(label, confidence)  # music 0.177293105471099

label, confidence = clf1.predict(["play internet radio"], probability=True)[0]
print(label, confidence)  # radio 0.7595566197852865

label, confidence = clf1.predict(["watch kill bill"], probability=True)[0]
print(label, confidence)  # movie 0.14057431626235914
