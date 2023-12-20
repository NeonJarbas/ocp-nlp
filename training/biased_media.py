from ovos_utils.log import LOG

from ocp_nlp.classify import BiasedMediaTypeClassifier
from ocp_nlp.testing import validate_podcast_bias_properties, validate_movie_bias_properties

LOG.set_level("DEBUG")

# download datasets from https://github.com/NeonJarbas/OCP-dataset


# each sentence template used once
csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"

# use N templates per label, where N is the label with less templates
csv_small_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_small_v0.csv"

# use N templates per label, where N is the label with more templates
# (repeat templates with different entity slots replaced)
csv_big_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_big_v0.csv"

N_SEARCHES = 5

# iterate over possible feature extractors and
# find best MLP classifier hyperparams + features combo
accuracies = {}

top_a = 0.9645927759546236  # I manually track this across runs
# when I find very good models


def find_best_mlp(feats, n_searches=N_SEARCHES):
    global accuracies, top_a
    # keyword biased classifier
    for i in range(n_searches):
        clf = BiasedMediaTypeClassifier(enabled_features=feats)  # lang agnostic
        acc, best_params, report = clf.find_best_MLP(csv_path, test_csv_path=csv_big_path)
        print("params:", best_params)
        print("balanced accuracy:", acc)
        validated = validate_movie_bias_properties(clf) and \
                    validate_podcast_bias_properties(clf)  # and \
        # validate_audiobook_bias_properties(clf)

        if not validated:
            print(f"{feats}_{i} failed validation, does not exhibit wanted bias properties")

        accuracies[f"{feats}_{i}"] = (feats, acc, best_params, report, validated, clf)
        if validated and acc > top_a:
            top_a = acc
            clf.save()


# commented out combos experimentally determined to consistently not pass the validation function
# NOTE: re-running the training might yield different results
# the randomized search does not cover all combos of MLP hyperparams
feats = [
    # ["keyword"],  # music -> audio | movie 0.042  -> 0.11 | music: 0.395 -> 0.028
    # ["bias"],
    # ["media_type"],
    ["keyword", "media_type"],  # music -> movie | movie 0.08  -> 0.3 | music: 0.13 -> 0.09 | audio: 0.05 -> 0.04
    # ["bias", "media_type"], # music -> music | movie 0.019  -> 0.024 | music: 0.53 -> 0.50 | audio: 0.003 -> 0.07
    # ["playback_type", "media_type"],  # music (no bias features)
    # ["keyword", "playback_type"], # music -> audio | movie 0.042  -> 0.040 | music: 0.29 -> 0.22 | audio: 0.17 -> 0.40
    # ["bias", "playback_type"], # audiobook -> documentary | movie 0.031 -> 0.028 | music 0.1 -> 0.2 | audio 0.02 -> 0.05
    # ["keyword", "bias"], # music -> audio | movie 0.043  -> 0.047 | music: 0.31 -> 0.21 | audio: 0.05 -> 0.50
    # ["keyword", "bias", "playback_type"], # music -> music | movie 0.03  -> 0.04 | music: 0.49 -> 0.48 | audio: 0.05 -> 0.12
    # ["keyword", "bias", "media_type"],
    # ["keyword", "bias", "media_type", "playback_type"],
]

for f in feats:
    print(f"finding best MLP for features: {f}")
    find_best_mlp(f)

# finding best MLP for features: ['keyword', 'media_type']
#
# params: {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (40, 116, 71), 'early_stopping': False, 'alpha': 0.0095, 'activation': 'identity'}
# balanced accuracy: 0.9651471749886363
#  movie_name bias changed label from:  music to movie
#  movie_name bias changed movie confidence:  0.07450303964064282 to 0.6499682813495347
#  movie_name bias changed music confidence:  0.34475078739682347 to 0.026004178096136525
#  movie_name bias changed audio confidence:  0.028622865102784813 to 0.0036721641710850115
#  movie_name bias changed audiobook confidence:  0.09921019109982641 to 0.009410656236843642
#  movie_name bias changed podcast confidence:  0.04367621972507546 to 0.015012670933577462
#    movie_name + soundtrack changed label from:  movie to music
#    movie_name + soundtrack changed movie confidence:  0.6499682813495347 to 0.003381156869436225
#    movie_name + soundtrack changed music confidence:  0.026004178096136525 to 0.963769040333419
#  podcast_name bias changed label from:  music to podcast
#  podcast_name bias changed movie confidence:  0.07450303964064282 to 0.0034325443933823536
#  podcast_name bias changed music confidence:  0.34475078739682347 to 0.003210353336975118
#  podcast_name bias changed audio confidence:  0.028622865102784813 to 0.0006612933925238789
#  podcast_name bias changed audiobook confidence:  0.09921019109982641 to 0.0008115191868721189
#  podcast_name bias changed podcast confidence:  0.04367621972507546 to 0.9636212995007216
#
# params: {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (127, 49),
#           'early_stopping': True, 'alpha': 0.028, 'activation': 'tanh'}
# balanced accuracy: 0.9566519288526771
#  movie_name bias changed label from:  music to movie
#  movie_name bias changed movie confidence:  0.0633890203520851 to 0.22339232650891527
#  movie_name bias changed music confidence:  0.24868707241063204 to 0.12043473733073021
#  movie_name bias changed audio confidence:  0.0696497926824917 to 0.10693427736761976
#  movie_name bias changed audiobook confidence:  0.04169723833663541 to 0.023016234557916265
#  movie_name bias changed podcast confidence:  0.07048184741298479 to 0.05869906754956972
#    MOVIE BIAS FAILED: audio label conf decreased -0.03728448468512806 - less than 0.1
# ['keyword', 'media_type']_1 failed validation, does not exhibit wanted bias properties
#
# params: {'solver': 'lbfgs', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (157,),
#           'early_stopping': True, 'alpha': 0.05, 'activation': 'tanh'}
# balanced accuracy: 0.9646386664944414
#     FAILED: default label is audiobook
# ['keyword', 'media_type']_3 failed validation, does not exhibit wanted bias properties
#
# params: {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (126, 132, 35),
#           'early_stopping': True, 'alpha': 0.07, 'activation': 'identity'}
# balanced accuracy: 0.9566606061205506
#  movie_name bias changed label from:  music to movie
#  movie_name bias changed movie confidence:  0.05805086675286882 to 0.27580789143797574
#  movie_name bias changed music confidence:  0.35605479590759004 to 0.12973418832934047
#  movie_name bias changed audio confidence:  0.04075724625251508 to 0.08337360357164889
#  movie_name bias changed audiobook confidence:  0.035414771023945565 to 0.01968639757171898
#  movie_name bias changed podcast confidence:  0.0671729044004774 to 0.06643791584319528
#
# params: {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (127, 36),
#           'early_stopping': False, 'alpha': 0.026000000000000002, 'activation': 'logistic'}
# balanced accuracy: 0.9558225012421631
#  movie_name bias changed label from:  music to movie
#  movie_name bias changed movie confidence:  0.0664306006951735 to 0.17899348496376793
#  movie_name bias changed music confidence:  0.19953778119460913 to 0.12966129306425728
#  movie_name bias changed audio confidence:  0.05836952953988912 to 0.041000719106287706
#  movie_name bias changed audiobook confidence:  0.03838406475857133 to 0.02735927502204998
#  movie_name bias changed podcast confidence:  0.07574883752569757 to 0.06868382622852034
#    MOVIE BIAS FAILED: music label conf decreased 0.06987648813035185 - less than 0.1
# ['keyword', 'media_type']_5 failed validation, does not exhibit wanted bias properties
#
# params: {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (91, 135),
#           'early_stopping': False, 'alpha': 0.05, 'activation': 'relu'}
# balanced accuracy: 0.9643499524893344
#  movie_name bias changed label from:  music to movie
#  movie_name bias changed movie confidence:  0.08609356538281901 to 0.49647713127467086
#  movie_name bias changed music confidence:  0.11461467530637932 to 0.05405275768567088
#  movie_name bias changed audio confidence:  0.03850747764008679 to 0.0314914430289013
#  movie_name bias changed audiobook confidence:  0.06673796967862386 to 0.015014068818545076
#  movie_name bias changed podcast confidence:  0.08181100693001511 to 0.044280487063434765

best = None
for (feats, acc, best_params, report, validated, clf) in accuracies.values():
    if not validated:
        continue
    if best is None or acc > best[1]:
        best = (feats, acc, best_params, report, clf)

if best is None:
    raise RuntimeError("None of the classifiers has the desired bias properties")

print("best accuracy:", best[1])
print("best features:", best[0])
print("best params:", best[2])
# best accuracy: 0.9645927759546236
# best features: ['keyword', 'media_type']
# best params: {'solver': 'adam', 'learning_rate': 'constant',
#               'hidden_layer_sizes': (47, 56, 45), 'early_stopping': False,
#               'alpha': 0.05, 'activation': 'relu'}


print(best[3])

# Balanced Accuracy: 0.9645927759546236
#               precision    recall  f1-score   support
#
#           ad       1.00      0.96      0.98      1200
#        adult       0.94      0.98      0.96       671
#   adult_asmr       1.00      0.97      0.98      1146
#        anime       0.98      0.87      0.92       961
#         asmr       0.99      0.97      0.98       779
#        audio       0.99      1.00      0.99      1032
#    audiobook       0.99      0.99      0.99      1170
#          bts       0.99      1.00      1.00       754
#     bw_movie       1.00      1.00      1.00       830
#      cartoon       0.91      0.97      0.94       905
#        comic       1.00      0.99      0.99       632
#  documentary       0.94      0.95      0.94      1005
#         game       1.00      0.98      0.99      1194
#       hentai       0.96      0.98      0.97       918
#        movie       0.87      0.99      0.93      1200
#        music       0.94      1.00      0.97      1199
#         news       0.98      1.00      0.99       876
#      podcast       0.93      0.98      0.96      1196
#        radio       0.99      0.97      0.98       596
#  radio_drama       0.97      0.98      0.98      1147
#       series       0.97      0.96      0.96      1176
#   short_film       0.95      0.80      0.87       354
# silent_movie       0.99      1.00      0.99       939
#      trailer       1.00      0.91      0.95       504
#   tv_channel       1.00      0.92      0.96      1172
#        video       0.97      0.98      0.97      1093
#
#     accuracy                           0.97     24649
#    macro avg       0.97      0.96      0.97     24649
# weighted avg       0.97      0.97      0.97     24649
#