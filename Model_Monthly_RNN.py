import numpy as np
import pandas as pd
# import geopandas as gpd
import datetime as dt
from shapely.wkt import loads
import tensorflow as tf
import tensorflow.keras as keras
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

wildfire_df = pd.read_csv('wildfire_data_vegetation.csv')
wildfire_df['location'] = wildfire_df['location'].apply(loads)
wildfire_df

#Drop locations with null values
#===============================
null_locations = []
for name, group in wildfire_df[wildfire_df.isna().any(axis=1)].groupby(['x','y']):
    null_locations.append(name)
for null_location in null_locations:
    wildfire_df.drop(wildfire_df[(wildfire_df['x'] ==null_location[0]) & (wildfire_df['y'] ==null_location[1])].index, inplace=True)

#Standardize all values
#======================
lon_min = wildfire_df['lon'].min()
lon_max = wildfire_df['lon'].max()
lat_min = wildfire_df['lat'].min()
lat_max = wildfire_df['lat'].max()
tmax_min = wildfire_df['tmax'].min()
tmax_max = wildfire_df['tmax'].max()
tmin_min = wildfire_df['tmin'].min()
tmin_max = wildfire_df['tmin'].max()
vp_min = wildfire_df['vp'].min()
vp_max = wildfire_df['vp'].max()
swe_min = wildfire_df['swe'].min()
swe_max = wildfire_df['swe'].max()
prcp_min = wildfire_df['prcp'].min()
prcp_max = wildfire_df['prcp'].max()
veg_min = wildfire_df['vegetation'].min()
veg_max = wildfire_df['vegetation'].max()
one_year_min = wildfire_df['1_year_history'].min()
one_year_max = wildfire_df['1_year_history'].max()
five_year_min = wildfire_df['5_year_history'].min()
five_year_max = wildfire_df['5_year_history'].max()
ten_year_min = wildfire_df['10_year_history'].min()
ten_year_max = wildfire_df['10_year_history'].max()
twenty_year_min = wildfire_df['20_year_history'].min()
twenty_year_max = wildfire_df['20_year_history'].max()

wildfire_df['lon'] = (wildfire_df['lon'] - lon_min) / ((lon_max - lon_min))
wildfire_df['lat'] = (wildfire_df['lat'] - lat_min) / ((lat_max - lat_min))
wildfire_df['tmax'] = (wildfire_df['tmax'] - tmax_min) / ((tmax_max - tmax_min))
wildfire_df['tmin'] = (wildfire_df['tmin'] - tmin_min) / ((tmin_max - tmin_min))
wildfire_df['vp'] = (wildfire_df['vp'] - vp_min) / ((vp_max - vp_min))
wildfire_df['swe'] = (wildfire_df['swe'] - swe_min) / ((swe_max - swe_min))
wildfire_df['prcp'] = (wildfire_df['prcp'] - prcp_min) / ((prcp_max - prcp_min))
wildfire_df['vegetation'] = (wildfire_df['vegetation'] - veg_min) / ((veg_max - veg_min))

wildfire_df.drop(columns=['x', 'y','scale', 'acres', 'location'], inplace=True)
wildfire_df

#Create 1 hot encoding
#=====================
wildfire_df['month_1'] = wildfire_df['month'].apply(lambda x: 1 if x==1 else 0)
wildfire_df['month_2'] = wildfire_df['month'].apply(lambda x: 1 if x==2 else 0)
wildfire_df['month_3'] = wildfire_df['month'].apply(lambda x: 1 if x==3 else 0)
wildfire_df['month_4'] = wildfire_df['month'].apply(lambda x: 1 if x==4 else 0)
wildfire_df['month_5'] = wildfire_df['month'].apply(lambda x: 1 if x==5 else 0)
wildfire_df['month_6'] = wildfire_df['month'].apply(lambda x: 1 if x==6 else 0)
wildfire_df['month_7'] = wildfire_df['month'].apply(lambda x: 1 if x==7 else 0)
wildfire_df['month_8'] = wildfire_df['month'].apply(lambda x: 1 if x==8 else 0)
wildfire_df['month_9'] = wildfire_df['month'].apply(lambda x: 1 if x==9 else 0)
wildfire_df['month_10'] = wildfire_df['month'].apply(lambda x: 1 if x==10 else 0)
wildfire_df['month_11'] = wildfire_df['month'].apply(lambda x: 1 if x==11 else 0)
wildfire_df['month_12'] = wildfire_df['month'].apply(lambda x: 1 if x==12 else 0)

wildfire_df['1_year_history_0'] = wildfire_df['1_year_history'].apply(lambda x: 1 if x==0 else 0)
wildfire_df['1_year_history_1'] = wildfire_df['1_year_history'].apply(lambda x: 1 if x==1 else 0)
wildfire_df['1_year_history_2'] = wildfire_df['1_year_history'].apply(lambda x: 1 if x==2 else 0)
wildfire_df['1_year_history_3'] = wildfire_df['1_year_history'].apply(lambda x: 1 if x==3 else 0)
wildfire_df['1_year_history_4'] = wildfire_df['1_year_history'].apply(lambda x: 1 if x==4 else 0)
wildfire_df['1_year_history_5'] = wildfire_df['1_year_history'].apply(lambda x: 1 if x==5 else 0)

wildfire_df['5_year_history_0'] = wildfire_df['5_year_history'].apply(lambda x: 1 if x==0 else 0)
wildfire_df['5_year_history_1'] = wildfire_df['5_year_history'].apply(lambda x: 1 if x==1 else 0)
wildfire_df['5_year_history_2'] = wildfire_df['5_year_history'].apply(lambda x: 1 if x==2 else 0)
wildfire_df['5_year_history_3'] = wildfire_df['5_year_history'].apply(lambda x: 1 if x==3 else 0)
wildfire_df['5_year_history_4'] = wildfire_df['5_year_history'].apply(lambda x: 1 if x==4 else 0)
wildfire_df['5_year_history_5'] = wildfire_df['5_year_history'].apply(lambda x: 1 if x==5 else 0)

wildfire_df['10_year_history_0'] = wildfire_df['10_year_history'].apply(lambda x: 1 if x==0 else 0)
wildfire_df['10_year_history_1'] = wildfire_df['10_year_history'].apply(lambda x: 1 if x==1 else 0)
wildfire_df['10_year_history_2'] = wildfire_df['10_year_history'].apply(lambda x: 1 if x==2 else 0)
wildfire_df['10_year_history_3'] = wildfire_df['10_year_history'].apply(lambda x: 1 if x==3 else 0)
wildfire_df['10_year_history_4'] = wildfire_df['10_year_history'].apply(lambda x: 1 if x==4 else 0)
wildfire_df['10_year_history_5'] = wildfire_df['10_year_history'].apply(lambda x: 1 if x==5 else 0)

wildfire_df['20_year_history_0'] = wildfire_df['20_year_history'].apply(lambda x: 1 if x==0 else 0)
wildfire_df['20_year_history_1'] = wildfire_df['20_year_history'].apply(lambda x: 1 if x==1 else 0)
wildfire_df['20_year_history_2'] = wildfire_df['20_year_history'].apply(lambda x: 1 if x==2 else 0)
wildfire_df['20_year_history_3'] = wildfire_df['20_year_history'].apply(lambda x: 1 if x==3 else 0)
wildfire_df['20_year_history_4'] = wildfire_df['20_year_history'].apply(lambda x: 1 if x==4 else 0)
wildfire_df['20_year_history_5'] = wildfire_df['20_year_history'].apply(lambda x: 1 if x==5 else 0)
wildfire_df

dates = []
for name, group in wildfire_df.groupby(['year', 'month']):
    dates.append(name)
locations = []
for name, group in wildfire_df.groupby(['lon', 'lat']):
    locations.append(name)
    if name == []:
        print('empty')
len(locations), len(dates)

feature_columns=['month_1', 'month_2', 'month_3',
       'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
       'month_10', 'month_11', 'month_12', 'lon', 'lat', 'tmax', 'tmin', 'vp', 'swe', 'prcp',
       'vegetation',
       '1_year_history_0', '1_year_history_1', '1_year_history_2', '1_year_history_3',
       '1_year_history_4', '1_year_history_5','5_year_history_0',
       '5_year_history_1', '5_year_history_2', '5_year_history_3',
       '5_year_history_4', '5_year_history_5', '10_year_history_0',
       '10_year_history_1', '10_year_history_2', '10_year_history_3',
       '10_year_history_4', '10_year_history_5', '20_year_history_0',
       '20_year_history_1', '20_year_history_2', '20_year_history_3',
       '20_year_history_4', '20_year_history_5']
label_column = ['category']

feature_dict = {}
label_dict = {}
for datapoint in wildfire_df.to_dict('records'):
    label_dict[datapoint['year'], datapoint['month'], datapoint['lon'], datapoint['lat']] = [datapoint[column] for column in label_column]
    feature_dict[datapoint['year'], datapoint['month'], datapoint['lon'], datapoint['lat']] = [datapoint[column] for column in feature_columns]

train_empty = True
test_empty = True

for date in range(len(dates)):
    if date-12>=0:
        datapoint_label = [label_dict[dates[date][0], dates[date][1], location[0], location[1]] for location in locations]
        datapoint_feature = [[feature_dict[past_date[0], past_date[1], location[0], location[1]] for past_date in dates[date-12:date]] for location in locations]
        if dates[date][0]<2016:
            if train_empty:
                train_data_x = np.array(datapoint_feature)
                train_data_y = np.array(datapoint_label)
                train_empty=False
            else:
                train_data_x = np.append(train_data_x, datapoint_feature, axis=0)
                train_data_y = np.append(train_data_y, datapoint_label, axis=0)
        else:
            if test_empty:
                test_data_x = np.array(datapoint_feature)
                test_data_y = np.array(datapoint_label)
                test_empty=False
            else:
                test_data_x = np.append(test_data_x, datapoint_feature, axis=0)
                test_data_y = np.append(test_data_y, datapoint_label, axis=0)
    print('{} Done'.format(dates[date]))

oversample = RandomOverSampler(random_state=20)#sampling_strategy='minority')
oversample.fit_resample(train_data_x[:,:,0], train_data_y)
train_x_resample_data = train_data_x[oversample.sample_indices_]
train_y_resample_data = train_data_y[oversample.sample_indices_]  

one_hot_labels = np.zeros((train_y_resample_data.size, train_y_resample_data.max()+1))
one_hot_labels[np.arange(train_y_resample_data.size),train_y_resample_data.flatten()] = 1
train_y_resample_data = one_hot_labels

one_hot_labels = np.zeros((test_data_y.size, test_data_y.max()+1))
one_hot_labels[np.arange(test_data_y.size),test_data_y.flatten()] = 1
test_data_y = one_hot_labels

train_x_resample_data = train_x_resample_data.astype('float32')
train_y_resample_data = train_y_resample_data.astype('float32')
test_data_x = test_data_x.astype('float32')
test_data_y = test_data_y.astype('float32')

#Calculate Metrics
#=================
def calculate_metrics(actual, predicted):
    actual = np.argmax(actual, axis=1)
    predicted = np.argmax(predicted, axis=1)

    accuracy = sum(actual==predicted)/len(actual)
    occurrence_recall = sum(np.logical_and(actual>0, predicted>0)) / sum(actual>0)
    occurrence_precision = sum(np.logical_and(actual>0, predicted>0)) / sum(predicted>0)
    large_fire_recall = sum(np.logical_and(actual>=3, predicted>=3)) / sum(actual>=3)
    large_fire_precision = sum(np.logical_and(actual>=3, predicted>=3)) / sum(predicted>=3)
    scale_accuracy = sum(np.logical_and(np.logical_and(actual>0, predicted>0), actual==predicted)) / sum(actual>0)
    relaxed_scale_accuracy = sum(np.logical_and(np.logical_and(actual>0, predicted>0), abs(actual-predicted)<=1)) / sum(actual>0)

    print('accuracy:', accuracy)
    print('occurence recall:', occurrence_recall)
    print('occurence precision:', occurrence_precision)
    print('large fire recall:', large_fire_recall)
    print('large fire precision:', large_fire_precision)
    print('scale accuracy:', scale_accuracy)
    print('scale relaxed accuracy:', relaxed_scale_accuracy)

    return [accuracy, occurrence_recall, occurrence_precision, large_fire_recall, large_fire_precision, scale_accuracy, relaxed_scale_accuracy]

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

#Neural Network and Training (Classification)
#============================================
Results_df = pd.DataFrame(columns=['learning rate', 'epochs', 'batch size', 'hidden layers', 'neurons', 'dropout', 'regularization',
                                   'accuracy', 'occurence recall', 'occurence precision', 'large fire recall', 'large fire precision', 'scale accuracy', 'scale relaxed accuracy'])
learning_rate = [0.01, 0.05, 0.1]
num_epochs = [100] #[50, 200]
batch_size = [1000, 5000] #[5000,10000]
num_hidden_layers = [2] #[3, 4]
num_neurons = [1, 20, 50]#[100, 200] #[50,100]
dropout_val = [0, 0.1, 0.2] #[0,0.1]
regularization_val = [0, 0.0001, 0.001] #[0,0.0001]

for lr in learning_rate:
    for epoch in num_epochs:
        for batch in batch_size:
            for hidden_layer in num_hidden_layers:
                for neuron in num_neurons:
                    for dropout in dropout_val:
                        for l2 in regularization_val:
                            hyper_parameters = [lr, epoch, batch, hidden_layer, neuron, dropout, l2]

                            inputs = keras.layers.Input(shape=(train_x_resample_data.shape[1], train_x_resample_data.shape[2]))
                            x = keras.layers.LSTM(neuron, return_sequences=True, kernel_regularizer=keras.regularizers.l2(l2), kernel_initializer='he_normal')(inputs)
                            x = keras.layers.Dropout(dropout)(x)
                            x = keras.layers.LSTM(neuron, kernel_regularizer=keras.regularizers.l2(l2), kernel_initializer='he_normal')(x)
                            x = keras.layers.Dropout(dropout)(x)
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

                            history = model.fit(train_x_resample_data, train_y_resample_data, validation_data=(test_data_x, test_data_y), epochs=epoch, batch_size=batch, shuffle=True)

                            metric_calculations = calculate_metrics(test_data_y, model.predict(test_data_x))

                            Results_df.loc[len(Results_df)] = hyper_parameters+metric_calculations
                            Results_df.to_csv('wildfire_Results_rnn_2.csv')