import numpy as np

def print_prob(prob,limit=5):
    from cvtron.data_zoo.imagenet_classes import CLASS_NAMES
    from cvtron.functions.softmax import softmax
    synset = CLASS_NAMES
    num_classes = len(synset)
    pred = np.argsort(-prob)[::-1][:limit]
    prob = np.sort(-prob[0])
    topn = [(synset[pred[0][i]], -prob[i]) for i in range(limit)]
    print("Top "+str(limit)+": ", topn)
    return topn