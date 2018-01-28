import sys
sys.path.append('../../model_zoo')
sys.path.append('../../utils')
class ImageClassifier(object):
    def __init__(self, model_name='vgg_19',model_path=''):
        self.model_name = model_name
        self.model_path = model_path
        if model_name not in ['vgg_19','inception_v3']:
            raise ValueError('Only VGG 19 and Inception V3 are allowed')
        if not os.path.isfile(model_path):
            raise ValueError('Model Not Found')
        self.sess = tf.InteractiveSession()
        if model_name == 'vgg_19':
            from vgg_19 import simple_api
            x = tf.placeholder("float", [None, 224, 224, 3])
            self.network = simple_api(x)
        elif model_name == 'inception_v3':
            saver = tf.train.Saver()
            from inception import simple_api
            x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
            self.network = simple_api(x)
        else:
            raise ValueError('Only VGG 19 and Inception V3 are allowed')
        y = self.network.outputs
        self.probs = tf.nn.softmax(y, name="prob")
        self._init_model_(model_name,model_path='')

    def _init_model_(model_name='vgg_19',model_path=''):
        if model_name not in ['vgg_19','inception_v3']:
            raise ValueError('Only VGG 19 and Inception V3 are allowed')
        if model_name == 'vgg_19':
            tl.layers.initialize_global_variables(self.sess)
            npz = np.load(model_path, encoding='latin1').item()
            params = []
            for val in sorted( npz.items() ):
                W = np.asarray(val[1][0])
                b = np.asarray(val[1][1])
                print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
                params.extend([W, b])
            tl.files.assign_params(self.sess, params, self.network)
        elif model_name == 'inception_v3':
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)

    def classify(img_file):
        from image_loader import load_image
        if self.model_name=='vgg_19':
            image = load_image(img_file,224,224)
            image = image.reshape((1, 224, 224, 3))
        elif self.model_name=='inception_v3':
            image = load_image(img_file,299,299)
            image = image.reshape((1, 299, 299, 3))
        prob = sess.run(self.probs, feed_dict= {x : image})
        return prob


        