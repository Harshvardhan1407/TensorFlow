import logging


logger = logging.getLogger(__name__)
logging.basicConfig(filename='tf.log', filemode='w', level=logging.INFO,format='%(asctime)s %(message)s')