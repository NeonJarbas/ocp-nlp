from ovos_utils.log import LOG
from ocp_nlp.classify import KeywordMediaTypeClassifier

ents_csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_entities_v0.csv"
s_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_sentences_v0.csv"
t_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_playback_type_v0.csv"
csv_small_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_small_v0.csv"
csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"
csv_big_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_big_v0.csv"

LOG.set_level("DEBUG")
# download datasets from https://github.com/NeonJarbas/OCP-dataset

# keyword biased classifier
clf = KeywordMediaTypeClassifier()  # lang agnostic
clf.find_best_MLP(csv_path, test_csv_path=csv_big_path)
#clf.search_best(csv_path, test_csv_path=csv_big_path)

# Accuracy: 0.88880505374442
#               precision    recall  f1-score   support
#
#           ad       0.97      0.62      0.76      1200
#        adult       0.75      0.92      0.83       672
#   adult_asmr       1.00      0.89      0.94      1139
#        anime       0.98      0.80      0.88       976
#         asmr       0.99      0.92      0.95       769
#        audio       0.92      0.94      0.93      1032
#    audiobook       0.99      0.97      0.98      1175
#          bts       0.99      0.96      0.98       754
#     bw_movie       0.99      0.95      0.97       832
#      cartoon       0.84      0.88      0.86       904
#        comic       0.90      0.92      0.91       635
#  documentary       0.92      0.91      0.91      1004
#         game       0.99      0.92      0.95      1194
#       hentai       0.94      0.96      0.95       910
# iot_playback       0.99      0.87      0.93      1129
#        movie       0.64      0.94      0.76      1200
#        music       0.67      0.95      0.78      1200
#         news       0.96      0.99      0.97       877
#      podcast       0.87      0.93      0.90      1196
#        radio       0.89      0.96      0.92       593
#  radio_drama       0.96      0.95      0.96      1145
#       series       0.86      0.85      0.85      1188
#   short_film       0.79      0.61      0.69       355
# silent_movie       0.94      0.92      0.93       942
#      trailer       0.92      0.76      0.84       499
#   tv_channel       1.00      0.84      0.91      1173
#        video       0.84      0.86      0.85      1093
#
#     accuracy                           0.89     25786
#    macro avg       0.91      0.89      0.89     25786
# weighted avg       0.91      0.89      0.90     25786

clf.load()


# klownevilus is an unknown entity
labels = clf.predict_labels(["play klownevilus"])[0]
print(clf.predict(["play klownevilus"])[0])
old = labels["movie"]
old2 = labels["music"]

# probability increases for movie
clf.register_entity("movie_name", ["klownevilus"])

labels2 = clf.predict_labels(["play klownevilus"])[0]
print(clf.predict(["play klownevilus"])[0])
n = labels2["movie"]
n2 = labels2["music"]

print("bias changed movie confidence: ", old, "to", n)
print("bias changed music confidence: ", old2, "to", n2)
# bias changed movie confidence:  0.039498435675704546 to 0.22248210180998731
# bias changed music confidence:  0.45780906930410603 to 0.108106704335338

assert n > old
assert n2 < old2

from pprint import pprint

pprint(labels)
pprint(labels2)