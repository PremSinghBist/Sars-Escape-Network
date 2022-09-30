import LanguageModel
import featurizer as FZ
if __name__ == '__main__':
    x_train, y_train  = FZ.get_train_encoded_dataset()
    x_val, y_val =  FZ.get_validation_encoded_dataset()
    #import keras_tuner as kt
    #LanguageModel.build_hyperModel(kt.HyperParameters())
 
    best_model = LanguageModel.initialize_escape_tuner(x_train, y_train, x_val, y_val)
    print(best_model)
    #Building model after search is 
    #best_model.build(input_shape=(1280, ))





