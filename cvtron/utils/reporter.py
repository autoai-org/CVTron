import numpy as np

def print_prob(prob,limit=5):
    from cvtron.data_zoo.imagenet_classes import CLASS_NAMES
    from cvtron.functions.softmax import softmax
    synset = CLASS_NAMES
    pred = np.argsort(-prob)[::-1][:limit]
    confidence = softmax(prob)
    topn = [(synset[pred[0][i]], confidence[0][i]) for i in range(limit)]
    print("Top "+str(limit)+": ", topn)
    return topn