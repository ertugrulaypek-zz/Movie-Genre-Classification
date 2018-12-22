from ExperimentSuite import ExperimentSuite
import tensorflow as tf
from Vectorizer import Vectorizer
from Preprocessor import Preprocessor
DEFAULT_EPOCH = 75
DEFAULT_LAYERS = (512,)
DEFAULT_ACTIVATION = tf.nn.relu
DEFAULT_LOSS = "categorical_hinge"

tbCallBack=tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)




if __name__ == "__main__":

    raw_dataset_path = " "
    processed_dataset_path = " "
    p=Preprocessor(dataset_directory = raw_dataset_path, processed_dataset_directory = processed_dataset_path)
    p.preprocess()

    es = ExperimentSuite(dataset_directory=processed_dataset_path, train_directory = "TrainDataset", test_directory = "TestDataset")

    vec = Vectorizer(max_df=0.99, min_df=0.1)

    vec.fit(es.train_contents)
    #print("length of vocab: "+str(len(vec.vocabulary)))

    train_x =vec.transform(es.train_contents,method='count')
    test_x  =vec.transform(es.test_contents, method='count')

    es.train_model(layers= DEFAULT_LAYERS, tbCallBack=tbCallBack, train_x=train_x, train_y=es.train_y, test_x=test_x, test_y=es.test_y,  loss=DEFAULT_LOSS, activation=DEFAULT_ACTIVATION, epoch=DEFAULT_EPOCH)

