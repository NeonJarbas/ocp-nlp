from ovos_utils.log import LOG

from ocp_nlp.classify import BiasedMediaTypeClassifier

LOG.set_level("DEBUG")

# download datasets from https://github.com/NeonJarbas/OCP-dataset


# each sentence template used once
csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"

# use N templates per label, where N is the label with less templates
csv_small_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_small_v0.csv"

# use N templates per label, where N is the label with more templates
# (repeat templates with different entity slots replaced)
csv_big_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_big_v0.csv"

WANTED_DEFAULT_LABEL = "music"
STRICT = False  # if True require other labels bias to decrease at least FAIL_THRESH
FAIL_THRESH = 0.1  # require all other bias to decrease by at least FAIL_THRESH
N_SEARCHES = 15


# check that the classifier has the wanted bias properties
def validate_movie_bias_properties(clf):
    # klownevilus is an unknown entity
    labels = clf.predict_labels(["play klownevilus"])[0]
    l1 = clf.predict(["play klownevilus"])[0]

    if l1 != WANTED_DEFAULT_LABEL:
        print(f"    BIAS FAILED: default label is {l1}")
        return False

    old = labels["movie"]
    old2 = labels["music"]
    old3 = labels["audio"]
    old4 = labels["audiobook"]
    old5 = labels["podcast"]
    old6 = labels["radio_drama"]
    old7 = labels["series"]

    # netflix is a {movie_streaming_provider} entity
    labels4 = clf.predict_labels(["play klownevilus in netflix"])[0]
    l4 = clf.predict(["play klownevilus in netflix"])[0]
    nf = labels4["movie"]
    nf2 = labels4["music"]
    nf3 = labels4["series"]

    # probability increases for movie
    clf.register_entity("movie_name", ["klownevilus"])

    labels2 = clf.predict_labels(["play klownevilus"])[0]
    l2 = clf.predict(["play klownevilus"])[0]
    n = labels2["movie"]
    n2 = labels2["music"]
    n3 = labels2["audio"]
    n4 = labels2["audiobook"]
    n5 = labels2["podcast"]
    n6 = labels2["radio_drama"]

    labels3 = clf.predict_labels(["play klownevilus movie soundtrack"])[0]
    l3 = clf.predict(["play klownevilus movie soundtrack"])[0]
    s = labels3["movie"]
    s2 = labels3["music"]

    # spotify is a {music_streaming_provider} entity
    labels5 = clf.predict_labels(["play klownevilus in spotify"])[0]
    l5 = clf.predict(["play klownevilus in spotify"])[0]
    sp = labels5["movie"]
    sp2 = labels5["music"]
    sp3 = labels5["audiobook"]
    sp4 = labels5["radio_drama"]

    # librivox is a {audiobook_streaming_provider} entity
    labels6 = clf.predict_labels(["play klownevilus in librivox"])[0]
    l6 = clf.predict(["play klownevilus in librivox"])[0]
    lb = labels6["movie"]
    lb2 = labels6["music"]
    lb3 = labels6["audiobook"]

    clf.deregister_entity("movie_name")

    print(" netflix bias changed label from: ", l1, "to", l4)
    print(" netflix bias changed movie confidence: ", old, "to", nf)
    print(" netflix bias changed music confidence: ", old2, "to", nf2)
    # TODO - series dataset needs some love
    print(" netflix bias changed series confidence: ", old7, "to", nf3)

    print(" movie_name bias changed label from: ", l1, "to", l2)
    print(" movie_name bias changed movie confidence: ", old, "to", n)
    print(" movie_name bias changed music confidence: ", old2, "to", n2)
    print(" movie_name bias changed audio confidence: ", old3, "to", n3)
    print(" movie_name bias changed audiobook confidence: ", old4, "to", n4)
    print(" movie_name bias changed podcast confidence: ", old5, "to", n5)
    print(" movie_name bias changed radio_drama confidence: ", old6, "to", n6)

    print("   movie_name + soundtrack changed label from: ", l2, "to", l3)
    print("   movie_name + soundtrack changed movie confidence: ", n, "to", s)
    print("   movie_name + soundtrack changed music confidence: ", n2, "to", s2)

    print("   movie_name + spotify changed label from: ", l2, "to", l5)
    print("   movie_name + spotify changed movie confidence: ", n, "to", sp)
    print("   movie_name + spotify changed music confidence: ", n2, "to", sp2)
    # TODO - audiobook dataset needs some love
    print("   movie_name + spotify changed audiobook confidence: ", n4, "to", sp3)
    print("   movie_name + spotify changed radio_drama confidence: ", n6, "to", sp4)

    print("   movie_name + librivox changed label from: ", l2, "to", l6)
    print("   movie_name + librivox changed movie confidence: ", n, "to", lb)
    print("   movie_name + librivox changed music confidence: ", n2, "to", lb2)
    print("   movie_name + librivox changed audiobook confidence: ", n4, "to", lb3)

    if l2 != "movie":
        print(f" MOVIE BIAS FAILED: predicted label is {l1}")
        return False

    if l3 != "music":
        print(f" MUSIC BIAS FAILED: can not disambiguate soundtracks - predicted label is {l3}")
        return False

    if l4 not in ["movie", "series"] or nf < old:  # or nf3 < old7
        # series also valid but never saw it during training
        print(f" NETFLIX BIAS FAILED: bias insufficient")
        return False

    if l5 not in ["radio_drama", "audiobook", "music"] or \
            sp4 < n6 or \
            sp2 < n2 or \
            sp > n:  # or  sp3 < n4
        # any of above is acceptable, media is probably a movie adaptation or a soundtrack
        print(f" SPOTIFY BIAS FAILED:  bias insufficient")
        return False

    if l6 not in ["audiobook"] or \
            lb3 < n4 or \
            lb2 > n2 or \
            lb > n:  # or  sp3 < n4
        # any of above is acceptable, media is probably a movie adaptation or a soundtrack
        print(f" LIBRIVOX BIAS FAILED:  bias insufficient")
        return False

    if n < FAIL_THRESH:
        print(f" MOVIE BIAS FAILED: confidence is less than {FAIL_THRESH}")
        return False

    if STRICT:  # TODO check other labels

        if n2 > 0.1 and old2 - n2 < FAIL_THRESH:
            print(f" MOVIE BIAS FAILED: music label conf decreased {old2 - n2} - less than {FAIL_THRESH}")
            return False

        if n3 > 0.1 and old3 - n3 < FAIL_THRESH:
            print(f"   MOVIE BIAS FAILED: audio label conf decreased {old3 - n3} - less than {FAIL_THRESH}")
            return False

        if n4 > 0.1 and old4 - n4 < FAIL_THRESH:
            print(f"   MOVIE BIAS FAILED: audiobook label conf decreased {old3 - n3} - less than {FAIL_THRESH}")
            return False

        if n5 > 0.1 and old5 - n5 < FAIL_THRESH:
            print(f"   MOVIE BIAS FAILED: podcast label conf decreased {old3 - n3} - less than {FAIL_THRESH}")
            return False

    return True


def validate_podcast_bias_properties(clf):
    # klownevilus is an unknown entity
    labels = clf.predict_labels(["play klownevilus"])[0]
    l1 = clf.predict(["play klownevilus"])[0]

    if l1 != WANTED_DEFAULT_LABEL:
        print(f"    BIAS FAILED: default label is {l1}")
        return False

    old = labels["movie"]
    old2 = labels["music"]
    old3 = labels["audio"]
    old4 = labels["audiobook"]
    old5 = labels["podcast"]

    # probability increases for podcast
    clf.register_entity("podcast_name", ["klownevilus"])

    labels2 = clf.predict_labels(["play klownevilus"])[0]
    l2 = clf.predict(["play klownevilus"])[0]
    n = labels2["movie"]
    n2 = labels2["music"]
    n3 = labels2["audio"]
    n4 = labels2["audiobook"]
    n5 = labels2["podcast"]

    clf.deregister_entity("podcast_name")

    print(" podcast_name bias changed label from: ", l1, "to", l2)
    print(" podcast_name bias changed movie confidence: ", old, "to", n)
    print(" podcast_name bias changed music confidence: ", old2, "to", n2)
    print(" podcast_name bias changed audio confidence: ", old3, "to", n3)
    print(" podcast_name bias changed audiobook confidence: ", old4, "to", n4)
    print(" podcast_name bias changed podcast confidence: ", old5, "to", n5)

    if l2 != "podcast":
        print(f" PODCAST BIAS FAILED: predicted label is {l1}")
        return False

    if n5 < FAIL_THRESH:
        print(f" PODCAST BIAS FAILED: confidence is less than {FAIL_THRESH}")
        return False

    return True


# TODO - dataset needs more book_name and narrator templates
def validate_audiobook_bias_properties(clf):
    # klownevilus is an unknown entity
    labels = clf.predict_labels(["play klownevilus"])[0]
    l1 = clf.predict(["play klownevilus"])[0]

    if l1 != WANTED_DEFAULT_LABEL:
        print(f"    BIAS FAILED: default label is {l1}")
        return False

    old = labels["movie"]
    old2 = labels["music"]
    old3 = labels["audio"]
    old4 = labels["audiobook"]
    old5 = labels["podcast"]

    # probability increases for podcast
    clf.register_entity("book_name", ["klownevilus"])

    labels2 = clf.predict_labels(["play klownevilus"])[0]
    l2 = clf.predict(["play klownevilus"])[0]
    n = labels2["movie"]
    n2 = labels2["music"]
    n3 = labels2["audio"]
    n4 = labels2["audiobook"]
    n5 = labels2["podcast"]

    clf.deregister_entity("book_name")

    print(" book_name bias changed label from: ", l1, "to", l2)
    print(" book_name bias changed movie confidence: ", old, "to", n)
    print(" book_name bias changed music confidence: ", old2, "to", n2)
    print(" book_name bias changed audio confidence: ", old3, "to", n3)
    print(" book_name bias changed audiobook confidence: ", old4, "to", n4)
    print(" book_name bias changed podcast confidence: ", old5, "to", n5)

    if l2 != "audiobook":
        print(f" AUDIOBOOK BIAS FAILED: predicted label is {l1}")
        return False

    if n4 < FAIL_THRESH:
        print(f" AUDIOBOOK BIAS FAILED: confidence is less than {FAIL_THRESH}")
        return False

    return True


# iterate over possible feature extractors and
# find best MLP classifier hyperparams + features combo
accuracies = {}

top_a = 0.95


def find_best_mlp(feats, n_searches=N_SEARCHES):
    global accuracies, top_a
    # keyword biased classifier
    for i in range(n_searches):
        clf = BiasedMediaTypeClassifier(enabled_features=feats)  # lang agnostic
        acc, best_params, report = clf.find_best_MLP(csv_path, test_csv_path=csv_big_path)
        print("params:", best_params)
        print("balanced accuracy:", acc)
        validated = validate_movie_bias_properties(clf) and \
                    validate_podcast_bias_properties(clf) #and \
                    #validate_audiobook_bias_properties(clf)

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
# best accuracy: 0.9651471749886363
# best features: ['keyword', 'media_type']
# best params: {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (40, 116, 71),
#               'early_stopping': False, 'alpha': 0.0095, 'activation': 'identity'}


print(best[3])

# Balanced Accuracy: 0.9617531060556558
#               precision    recall  f1-score   support
#
#           ad       1.00      0.96      0.98      1200
#        adult       0.95      0.98      0.96       671
#   adult_asmr       1.00      0.97      0.99      1146
#        anime       0.98      0.86      0.91       961
#         asmr       1.00      0.95      0.98       779
#        audio       0.99      1.00      0.99      1032
#    audiobook       0.99      0.99      0.99      1170
#          bts       0.99      1.00      1.00       754
#     bw_movie       1.00      1.00      1.00       830
#      cartoon       0.91      0.97      0.94       905
#        comic       1.00      0.99      0.99       632
#  documentary       0.94      0.92      0.93      1005
#         game       1.00      0.99      0.99      1194
#       hentai       0.96      0.98      0.97       918
#        movie       0.87      0.99      0.93      1200
#        music       0.92      0.99      0.96      1199
#         news       0.98      0.99      0.99       876
#      podcast       0.92      0.98      0.95      1196
#        radio       0.99      0.97      0.98       596
#  radio_drama       0.97      0.97      0.97      1147
#       series       0.97      0.97      0.97      1176
#   short_film       0.96      0.79      0.86       354
# silent_movie       0.99      0.99      0.99       939
#      trailer       0.99      0.90      0.94       504
#   tv_channel       1.00      0.93      0.96      1172
#        video       0.97      0.97      0.97      1093
#
#     accuracy                           0.97     24649
#    macro avg       0.97      0.96      0.97     24649
# weighted avg       0.97      0.97      0.97     24649
