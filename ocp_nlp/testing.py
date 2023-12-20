"""these utils methods test that a classifier has the intended
bias properties and reacts as expected to keywords

classifiers that dont pass the validation are discarded during training

if the constraints are not realistic then no useful model will be found
and you will be endlessly training MultiLayerPerceptrons

either improve feature extractors or loosen these constraints if that happens
"""

WANTED_DEFAULT_LABEL = "music"
STRICT = False  # if True require other labels bias to decrease at least FAIL_THRESH
FAIL_THRESH = 0.1  # require all other bias to decrease by at least FAIL_THRESH


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

    if l6 != "audiobook" or \
            lb3 < n4 or \
            lb2 > n2 or \
            lb > n:  # or  sp3 < n4
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

    clf.register_entity("podcaster", ["klownevilus"])
    l3 = clf.predict(["play klownevilus"])[0]
    s = labels2["movie"]
    s2 = labels2["music"]
    s3 = labels2["audio"]
    s4 = labels2["audiobook"]
    s5 = labels2["podcast"]

    print(" podcaster bias changed label from: ", l1, "to", l3)
    print(" podcaster bias changed movie confidence: ", old, "to", s)
    print(" podcaster bias changed music confidence: ", old2, "to", s2)
    print(" podcaster bias changed audio confidence: ", old3, "to", s3)
    print(" podcaster bias changed audiobook confidence: ", old4, "to", s4)
    print(" podcaster bias changed podcast confidence: ", old5, "to", s5)

    clf.deregister_entity("podcaster")

    if l2 != "podcast":
        print(f" PODCAST_NAME BIAS FAILED: predicted label is {l1}")
        return False

    if n5 < FAIL_THRESH:
        print(f" PODCAST_NAME BIAS FAILED: confidence is less than {FAIL_THRESH}")
        return False

    if l3 != "podcast":
        print(f" PODCASTER BIAS FAILED: predicted label is {l1}")
        return False

    if s5 < FAIL_THRESH:
        print(f" PODCASTER BIAS FAILED: confidence is less than {FAIL_THRESH}")
        return False

    return True


# TODO - dataset needs more book_name and narrator templates
# this validator always fails
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
