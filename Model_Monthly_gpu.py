import numpy as np
import pandas as pd
# import geopandas as gpd
import datetime as dt
from shapely.wkt import loads
import tensorflow as tf
import tensorflow.keras as keras
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_logical_devices('CPU')))
gpus = tf.config.list_logical_devices('GPU')
print(gpus)

wildfire_df = pd.read_csv('wildfire_data_vegetation.csv')
wildfire_df['location'] = wildfire_df['location'].apply(loads)

prediction_df = wildfire_df.dropna().loc[wildfire_df['year']>=2016][['year', 'month', 'x', 'y', 'lon', 'lat','location', 'category']]

wildfire_train_df = wildfire_df.dropna().loc[wildfire_df['year']<2016].drop(columns=['x', 'y','scale', 'acres', 'location'])
wildfire_test_df = wildfire_df.dropna().loc[wildfire_df['year']>=2016].drop(columns=['x', 'y','scale', 'acres', 'location'])

train_x_resample_df, train_y_resample_df = RandomOverSampler(random_state=20).fit_resample(wildfire_train_df.drop(columns='category'), wildfire_train_df['category'])

test_x_df = wildfire_test_df.drop(columns='category')
test_y_df = wildfire_test_df['category']

lon_min = min(train_x_resample_df['lon'].min(), test_x_df['lon'].min())
lon_max = max(train_x_resample_df['lon'].max(), test_x_df['lon'].max())
lat_min = min(train_x_resample_df['lat'].min(), test_x_df['lat'].min())
lat_max = max(train_x_resample_df['lat'].max(), test_x_df['lat'].max())
tmax_min = min(train_x_resample_df['tmax'].min().min(), test_x_df['tmax'].min().min())
tmax_max = max(train_x_resample_df['tmax'].max().max(), test_x_df['tmax'].max().max())
tmin_min = min(train_x_resample_df['tmin'].min().min(), test_x_df['tmin'].min().min())
tmin_max = max(train_x_resample_df['tmin'].max().max(), test_x_df['tmin'].max().max())
vp_min = min(train_x_resample_df['vp'].min().min(), test_x_df['vp'].min().min())
vp_max = max(train_x_resample_df['vp'].max().max(), test_x_df['vp'].max().max())
swe_min = min(train_x_resample_df['swe'].min().min(), test_x_df['swe'].min().min())
swe_max = max(train_x_resample_df['swe'].max().max(), test_x_df['swe'].max().max())
prcp_min = min(train_x_resample_df['prcp'].min().min(), test_x_df['prcp'].min().min())
prcp_max = max(train_x_resample_df['prcp'].max().max(), test_x_df['prcp'].max().max())
veg_min = min(train_x_resample_df['vegetation'].min().min(), test_x_df['vegetation'].min().min())
veg_max = max(train_x_resample_df['vegetation'].max().max(), test_x_df['vegetation'].max().max())
one_year_min = min(train_x_resample_df['1_year_history'].min().min(), test_x_df['1_year_history'].min())
one_year_max = max(train_x_resample_df['1_year_history'].max().max(), test_x_df['1_year_history'].max())
five_year_min = min(train_x_resample_df['5_year_history'].min().min(), test_x_df['5_year_history'].min())
five_year_max = max(train_x_resample_df['5_year_history'].max().max(), test_x_df['5_year_history'].max())
ten_year_min = min(train_x_resample_df['10_year_history'].min().min(), test_x_df['10_year_history'].min())
ten_year_max = max(train_x_resample_df['10_year_history'].max().max(), test_x_df['10_year_history'].max())
twenty_year_min = min(train_x_resample_df['20_year_history'].min().min(), test_x_df['20_year_history'].min())
twenty_year_max = max(train_x_resample_df['20_year_history'].max().max(), test_x_df['20_year_history'].max())

#Predictor Training Data
#=======================
x_train_raw = tf.convert_to_tensor(train_x_resample_df.drop(columns=['year']))

month_one_hot = tf.one_hot(tf.cast(x_train_raw[:, 0], tf.int32)-1, 12)
lon_norm = tf.cast(tf.reshape((x_train_raw[:, 1] - lon_min) / (lon_max - lon_min), [-1,1]), tf.float32)
lat_norm = tf.cast(tf.reshape((x_train_raw[:, 2] - lat_min) / (lat_max - lat_min), [-1,1]), tf.float32)
tmax_norm = tf.cast(tf.reshape((x_train_raw[:,3] - tmax_min) / (tmax_max-tmax_min), [-1,1]), tf.float32)
tmin_norm = tf.cast(tf.reshape((x_train_raw[:,4] - tmin_min) / (tmin_max-tmin_min), [-1,1]), tf.float32)
vp_norm = tf.cast(tf.reshape((x_train_raw[:,5] - vp_min) / (vp_max-vp_min), [-1,1]), tf.float32)
swe_norm = tf.cast(tf.reshape((x_train_raw[:,6] - swe_min) / (swe_max-swe_min), [-1,1]), tf.float32)
prcp_norm = tf.cast(tf.reshape((x_train_raw[:,7] - prcp_min) / (prcp_max-prcp_min), [-1,1]), tf.float32)
vegetation_norm = tf.cast(tf.reshape(x_train_raw[:,8], [-1,1]), tf.float32)
one_year_one_hot = tf.one_hot(tf.cast(x_train_raw[:,9], tf.int32), 6)
five_year_one_hot = tf.one_hot(tf.cast(x_train_raw[:,10], tf.int32), 6)
ten_year_one_hot = tf.one_hot(tf.cast(x_train_raw[:,11], tf.int32), 6)
twenty_year_one_hot = tf.one_hot(tf.cast(x_train_raw[:,12], tf.int32), 6)
x_train = tf.concat([month_one_hot, lon_norm, lat_norm, tmax_norm, tmin_norm, vp_norm, swe_norm, prcp_norm, vegetation_norm, one_year_one_hot, five_year_one_hot, ten_year_one_hot, twenty_year_one_hot], 1)

#Predictor Testing Data
#======================
x_test_raw = tf.convert_to_tensor(test_x_df.drop(columns=['year']))

month_one_hot = tf.one_hot(tf.cast(x_test_raw[:, 0], tf.int32)-1, 12)
lon_norm = tf.cast(tf.reshape((x_test_raw[:, 1] - lon_min) / (lon_max - lon_min), [-1,1]), tf.float32)
lat_norm = tf.cast(tf.reshape((x_test_raw[:, 2] - lat_min) / (lat_max - lat_min), [-1,1]), tf.float32)
tmax_norm = tf.cast(tf.reshape((x_test_raw[:,3] - tmax_min) / (tmax_max-tmax_min), [-1,1]), tf.float32)
tmin_norm = tf.cast(tf.reshape((x_test_raw[:,4] - tmin_min) / (tmin_max-tmin_min), [-1,1]), tf.float32)
vp_norm = tf.cast(tf.reshape((x_test_raw[:,5] - vp_min) / (vp_max-vp_min), [-1,1]), tf.float32)
swe_norm = tf.cast(tf.reshape((x_test_raw[:,6] - swe_min) / (swe_max-swe_min), [-1,1]), tf.float32)
prcp_norm = tf.cast(tf.reshape((x_test_raw[:,7] - prcp_min) / (prcp_max-prcp_min), [-1,1]), tf.float32)
vegetation_norm = tf.cast(tf.reshape(x_test_raw[:,8], [-1,1]), tf.float32)
one_year_one_hot = tf.one_hot(tf.cast(x_test_raw[:,9], tf.int32), 6)
five_year_one_hot = tf.one_hot(tf.cast(x_test_raw[:,10], tf.int32), 6)
ten_year_one_hot = tf.one_hot(tf.cast(x_test_raw[:,11], tf.int32), 6)
twenty_year_one_hot = tf.one_hot(tf.cast(x_test_raw[:,12], tf.int32), 6)
x_test = tf.concat([month_one_hot, lon_norm, lat_norm, tmax_norm, tmin_norm, vp_norm, swe_norm, prcp_norm, vegetation_norm, one_year_one_hot, five_year_one_hot, ten_year_one_hot, twenty_year_one_hot], 1)

y_train = tf.convert_to_tensor(train_y_resample_df)
y_train = tf.one_hot(y_train, 6)
y_test = tf.convert_to_tensor(test_y_df)
y_test = tf.one_hot(y_test, 6)
y_train, y_test

#Ordinal Regression Layer
#========================
class Ordinal(tf.keras.Model):
    def __init__(self, num_classes):
        super(Ordinal, self).__init__(name='')
        
        omega_init = tf.range(0., num_classes-1, dtype='float32')
        self.omega = tf.Variable(tf.expand_dims(omega_init, axis=0))
        
        const_zero = tf.zeros((1,1),dtype='float32')
        self.const_zero = tf.constant(const_zero)
        
        b_init = tf.zeros(1, dtype='float32')
        self.b = tf.Variable(b_init)
        
    def call(self, input_X, training=False):
        omega_diff = tf.nn.relu(self.omega[:,1:]-self.omega[:,:-1]) + 1
        
        theta_pdf = tf.concat([self.const_zero, omega_diff], axis=1)
        
        linear = tf.cumsum(theta_pdf, axis=1) - input_X - self.b*self.b
        
        cumulative_proba = tf.nn.sigmoid(linear)
        
        proba_list = [cumulative_proba[:, :1],
                     cumulative_proba[:,1:]-cumulative_proba[:,:-1],
                     1-cumulative_proba[:,-1:]]
        proba = tf.keras.layers.Concatenate(axis=1)(proba_list)
        
        return proba

#Calculate Metrics
#=================
def calculate_metrics(prediction_df):
    count = 0
    correct = 0
    prediction_count = {}
    for prediction in range(0,6):
        prediction_count[prediction] = 0
    occurence_tp = 0
    occurence_fn = 0
    occurence_fp = 0
    scale_tp = 0
    scale_fn = 0
    relaxed_scale_tp = 0
    relaxed_scale_fn = 0
    large_fire_tp = 0
    large_fire_fn = 0
    large_fire_fp = 0

    for prediction in prediction_df.to_dict('records'):
        prediction_count[prediction['prediction']]+=1
        
        if prediction['category']==prediction['prediction']:
            correct +=1
        count +=1
        
        if prediction['category']>0:
            if prediction['prediction']>0:
                occurence_tp+=1
            else:
                occurence_fn+=1
                
            if prediction['category']==prediction['prediction']:
                scale_tp+=1
            else:
                scale_fn+=1
            
            if (abs(prediction['category']-prediction['prediction'])<=1) and (prediction['prediction']!=0):
                relaxed_scale_tp+=1
            else:
                relaxed_scale_fn+=1      
        else:
            if prediction['prediction']>0:
                occurence_fp+=1
                
        if (prediction['category']==3) or (prediction['category']==4) or (prediction['category']==5):
            if (prediction['prediction']==3) or (prediction['prediction']==4) or (prediction['prediction']==5):
                large_fire_tp+=1
            else:
                large_fire_fn+=1
        else:
            if (prediction['prediction']==3) or (prediction['prediction']==4) or (prediction['prediction']==5):
                large_fire_fp+=1
        
    metric_calculations = [correct/count, occurence_tp/(occurence_tp+occurence_fn), occurence_tp/(occurence_tp+occurence_fp),
                            large_fire_tp/(large_fire_tp+large_fire_fn), large_fire_tp/(large_fire_tp+large_fire_fp), 
                            scale_tp/(scale_tp+scale_fn), relaxed_scale_tp/(relaxed_scale_tp+relaxed_scale_fn)]         
    print(prediction_count)      
    print('accuracy:', metric_calculations[0])
    print('occurence recall:', metric_calculations[1])
    print('occurence precision:', metric_calculations[2])
    print('large fire recall:', metric_calculations[3])
    print('large fire precision:', metric_calculations[4])
    print('scale accuracy:', metric_calculations[5])
    print('scale relaxed accuracy:', metric_calculations[6])
    return metric_calculations

#Neural Network and Training (Classification)
#============================================
Results_df = pd.DataFrame(columns=['learning rate', 'epochs', 'batch size', 'hidden layers', 'neurons', 'dropout', 'regularization',
                                   'accuracy', 'occurence recall', 'occurence precision', 'large fire recall', 'large fire precision', 'scale accuracy', 'scale relaxed accuracy'])
learning_rate = [0.05, 0.1]
num_epochs = [100] #[50, 200]
batch_size = [5000] #[5000,10000]
num_hidden_layers = [3] #[3, 4]
num_neurons = [100, 200] #[50,100]
dropout_val = [0, 0.15, 0.2, 0.3] #[0,0.1]
regularization_val = [0, 0.0005, 0.001] #[0,0.0001]

for lr in learning_rate:
    for epoch in num_epochs:
        for batch in batch_size:
            for hidden_layer in num_hidden_layers:
                for neuron in num_neurons:
                    for dropout in dropout_val:
                        for l2 in regularization_val:
                            hyper_parameters = [lr, epoch, batch, hidden_layer, neuron, dropout, l2]

                            with tf.device('/gpu:0'):
                                inputs = keras.layers.Input(shape=x_train.shape[1:])
                                x = keras.layers.BatchNormalization()(inputs)
                                for layer in range(hidden_layer):
                                    x = keras.layers.Dense(neuron, kernel_regularizer=keras.regularizers.l2(l2), activation='elu', kernel_initializer='he_normal')(x)
                                    x = keras.layers.Dropout(dropout)(x)
                                    x = keras.layers.BatchNormalization()(x)
                                x = keras.layers.Dense(1, activation='linear')(x)
                                outputs = Ordinal(6)(x)
                                model = keras.Model(inputs=[inputs], outputs=[outputs])

                                metrics = [
                                    'accuracy',
                                    keras.metrics.Recall(class_id=1, name='class_1_recall'),
                                    keras.metrics.Recall(class_id=2, name='class_2_recall'),
                                    keras.metrics.Recall(class_id=3, name='class_3_recall'),
                                    keras.metrics.Recall(class_id=4, name='class_4_recall'),
                                    keras.metrics.Recall(class_id=5, name='class_5_recall')

                                ]

                                model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                                                #optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.99),
                                                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),#, label_smoothing=0.1),
                                                metrics=metrics)

                                history = model.fit(
                                    x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=batch, shuffle=True)
                            
                            y_pred = np.argmax(model.predict(x_test), axis=1)
                            prediction_df['prediction'] = y_pred

                            metric_calculations = calculate_metrics(prediction_df)

                            Results_df.loc[len(Results_df)] = hyper_parameters+metric_calculations
                            Results_df.to_csv('wildfire_Results_gpu_3.csv')