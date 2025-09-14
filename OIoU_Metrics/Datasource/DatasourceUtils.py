import re
import os 
import csv
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
from Plot import PlotUtils as pu    
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing as preproc
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

stop_words = set(stopwords.words('english'))


def stemming(data, stemmer):
    text = [stemmer.stem(word) for word in data]
    return data
    
def no_of_words(text):
    words = text.split()
    word_count = len(words)
    return word_count

def data_processing(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_token = word_tokenize(text)
    filtered_text =  [w for w in text_token if not w in stop_words]
    return " ".join(filtered_text) 

def loadIMDBWord2Vect( length_encodig =100):
    base_path = os.getcwd()
    dataset_path_train = base_path + os.sep +"Dataset"+ os.sep +"IMDB"+ os.sep +"train"+ os.sep
    dataset_path_test  = base_path + os.sep +"Dataset"+ os.sep +"IMDB"+ os.sep +"test"+ os.sep 
    
    reviews = []
    sentiment = []    
    
    #Train Negative Reviews
    file_list_train_neg = os.listdir( dataset_path_train+"neg"+ os.sep  )
    file_list_train_pos = os.listdir( dataset_path_train+"pos"+ os.sep  )
    
    file_list_test_neg  = os.listdir( dataset_path_test+"neg"+ os.sep  )
    file_list_test_pos  = os.listdir( dataset_path_test+"pos"+ os.sep  )
    
    for train_archive_neg, train_archive_pos, test_archive_neg , test_archive_pos in zip( file_list_train_neg, file_list_train_pos,
                                                      file_list_test_neg , file_list_test_pos  ): 
        
        a_train_neg = open(dataset_path_train+"neg"+ os.sep +train_archive_neg, 'rb')
        a_train_pos = open(dataset_path_train+"pos"+ os.sep +train_archive_pos, 'rb')
        
        a_test_neg = open(dataset_path_test+"neg"+ os.sep + test_archive_neg, 'rb')
        a_test_pos = open(dataset_path_test+"pos"+ os.sep +test_archive_pos, 'rb')
        
        reviews.append( a_train_neg.read().decode("UTF-8") )
        sentiment.append("negative")
        
        reviews.append( a_train_pos.read().decode("UTF-8") )
        sentiment.append("positive")
        
        reviews.append( a_test_neg.read().decode("UTF-8") )
        sentiment.append("negative")
        
        reviews.append( a_test_pos.read().decode("UTF-8") )
        sentiment.append("positive")
    
    data = {    
        "review"    : reviews,
        "sentiment" : sentiment    
    }
    
    df = pd.DataFrame(data)
    
    print(" -----------------------------")
    print( df.head() )
    print(f"DataFrame Shape: {df.shape}")
    
    df.review = df['review'].apply(data_processing)
    
    
    print( df.head() )
    print(f"DataFrame Shape: {df.shape}")
    
    print(" -----------------------------")
    # Clean data using the built in cleaner in gensim
    df['text_clean'] = df['review'].apply(lambda x: gensim.utils.simple_preprocess(x))

        # Encoding the label column
    df['sentiment']=df['sentiment'].map({'positive':1,'negative':0})
    
    print( df.head() )
    print(f"DataFrame Shape: {df.shape}")
    
    print(" -----------------------------")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split (df['text_clean'], df['sentiment'] , test_size=0.2)
    
    w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=length_encodig,
                                   window=5,
                                   min_count=2)
    
    w2v_model.wv.most_similar('movie')
    
    words = set(w2v_model.wv.index_to_key )

    X_train_vect = [np.array([w2v_model.wv[i] for i in ls if i in words] )
                         for ls in X_train]
    
    X_test_vect  = [np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test]
    
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
        
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    
    P = np.zeros( [len(X_train_vect_avg), X_train_vect_avg[0].shape[0]], np.float32 ) # t_P.shape -->   (2, 1000)
    T = np.zeros( [len(X_train_vect_avg), 1], np.int8 )    
    
    Ptest = np.zeros( [len(X_test_vect_avg), X_test_vect_avg[0].shape[0]], np.float32 )
    Ttest = np.zeros( [len(X_test_vect_avg), 1 ], np.int8 )
    
    for idx in range( P.shape[0] ):
        P[ idx ] = X_train_vect_avg[idx] 
        T[idx] = list(y_train)[idx]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = X_test_vect_avg[idx ]
        Ttest[ idx ] = list(y_test)[idx]


    def get_Data():
       for idx  in range( P.shape[0]):
           data = P[idx]
           target = T[idx]

           #data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
           #target = tf.convert_to_tensor(target, dtype=tf.float32)
         
           #yield data,target # Your supposed to yield in a loop
           yield data, target
    
    train_dataset = tf.data.Dataset.from_generator(get_Data, 
                                                   output_signature=(
                                                        tf.TensorSpec(shape=(P.shape[1]), dtype=tf.float32, name=None),
                                                        tf.TensorSpec(shape=(1), dtype=tf.int64, name=None)))
    
    for img in train_dataset:
        print(img)
        break
        
    
    train_dataset = train_dataset.batch(32)
    
    print("-----------------------------------------------------------------")
    print(f"\t Dataset Loaded: IMDB ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    print("-----------------------------------------------------------------")
    return train_dataset, None, (P, T),(Ptest, Ttest), [P.shape[1], 2 ,'DATA']

def load_MNIST():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test  = x_test.reshape(x_test.shape[0], -1)

    y_train = to_categorical( y_train , 10 )
    y_test  = to_categorical( y_test  , 10 )

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.batch(30000)
    test_dataset = test_dataset.batch(10000)

    print(f"\t Dataset Loaded: MNIST")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( x_train.shape) )  
    print("\t\t ---> T: " + str ( y_train.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( x_test.shape ))
    print("\t\t ---> Ttest: " + str( y_test.shape ))
    
    return train_dataset, test_dataset, (x_train, y_train),(x_test, y_test), [28*28, 10, 'IMAGE']     

def loadDataset_A(plot=False, to_categorical=False):
    
    base_path = os.getcwd()
    dataset_path =  base_path + os.sep +"Dataset"+os.sep+"A"+os.sep+"A.mat"
    data_set_name = " A "
    
    dict = loadmat( dataset_path )
    
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']
    
        
    t_P = np.array( t_p, dtype = np.float32)
    P = np.array( t_p, dtype = np.float32)
    
    t_T = np.array( t_t, dtype = np.int8 )
    T = np.array( t_t, dtype = np.int8 )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    Ptest = np.array( t_ptest, dtype = np.float32 )
    
    t_Ttest = np.array( t_ttest, dtype = np.int8 )
    Ttest = np.array( t_ttest, dtype = np.int8 )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest
    
    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 ) # t_P.shape -->   (2, 1000)
    T = np.zeros( [t_T.shape[1], t_T.shape[0]], np.int8 )      # t_T.shape -->(1, 1000)
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0] ], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]    ], np.int8 )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_P[ :, idx ] 
        T[idx] = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    input_dim = P.shape[1]

    num_classes =  np.max(T)
    
    T = T -1           
    Ttest = Ttest -1  

    if ( to_categorical ):
        T = to_categorical( T , 2 )
        Ttest = to_categorical(Ttest, 2 )
    
    print(f"\t Dataset Loaded: {data_set_name}")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((P, T))
    test_dataset = tf.data.Dataset.from_tensor_slices((Ptest, Ttest))

    train_dataset = train_dataset.batch(P.shape[0])
    test_dataset = test_dataset.batch(Ptest.shape[0])
    
    if plot:
        pu.plot_espiral( P, T )
    
    print("-----------------------------------------------------------------")
    return train_dataset, test_dataset, (P, T), (Ptest, Ttest), [input_dim, num_classes, 'DATA']

def loadDataset_B(plot=False, to_cat=False, normalize =False):
    
    base_path = os.getcwd()
    dataset_path =  base_path + os.sep + "Dataset"+ os.sep +"B"+  os.sep +"B.mat"
    data_set_name = " B "
    
    dict = loadmat( dataset_path )
    
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']
    
    t_P = np.array( t_p, dtype = np.float32)
    P = np.array( t_p, dtype = np.float32)
    
    t_T = np.array( t_t, dtype = np.int8 )
    T = np.array( t_t, dtype = np.int8 )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    Ptest = np.array( t_ptest, dtype = np.float32 )

    t_Ttest = np.array( t_ttest, dtype = np.int8 )
    Ttest = np.array( t_ttest, dtype = np.int8 )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest
    
    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0]], np.int8 )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0] ], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0] ], np.int8 )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_P[ :, idx ] 
        T[idx]  = t_T[ :, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    input_dim = P.shape[1]

    num_classes =  np.max(T)
    
    T = T -1          # ESTO ES POR EL CATEGORICAL 
    Ttest = Ttest -1  # ESTO ES POR EL CATEGORICAL
    
    if ( to_cat ):
        T = to_categorical( T , 3 )
        Ttest = to_categorical(Ttest, 3 )
    
    print(f"\t Dataset Loaded: {data_set_name}")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((P, T))
    test_dataset = tf.data.Dataset.from_tensor_slices((Ptest, Ttest))

    train_dataset = train_dataset.batch(P.shape[0])
    test_dataset = test_dataset.batch(Ptest.shape[0])
    
    if plot:
        pu.plot_N_class(P, T)
    
    print("-----------------------------------------------------------------")
    return train_dataset, test_dataset, (P, T),(Ptest, Ttest), [input_dim, num_classes, 'DATA']

def loadDataset_LetterRecognition(plot =False, to_cat =False):
    
    from sklearn.model_selection import  train_test_split

    
    base_path = os.getcwd()
    dataset_path =  base_path + os.sep + "Dataset"+ os.sep + "letterrecognition"+ os.sep +"letter-recognition.csv"

    data_set_name = "Letter Recognition "
    
    T_temp = []
    P_temp = []
    
    with open( dataset_path, 'r') as csvfile:
        csvReader = csv.reader( csvfile, delimiter= ',' )

        for row in csvReader :
            #Class Label
            T_temp.append( ord(row[0]) )
            P_temp.append( row[1:17] )
                   
    P_temp = np.array( P_temp, dtype = np.float32 )
    T_temp = np.array( T_temp, dtype = np.int8 )
    
    min_val = T_temp.min()
    
    P, Ptest, T, Ttest  =  train_test_split(P_temp, T_temp, test_size=0.2, random_state=4)
    
    del P_temp
    del T_temp
    
    input_dim = P.shape[1]
    num_classes =  26

    T = T -  min_val
    Ttest =  Ttest - min_val
    
    t_T=T.reshape((T.shape[0],1))
    t_Ttest=Ttest.reshape((Ttest.shape[0],1))
    
    if ( to_cat ):
        T = to_categorical( T , 26)
        Ttest = to_categorical(Ttest, 26)
    
    
    t_P = preproc.scale(P)                                 
    t_Ptest = preproc.scale(Ptest)       
             
    print("\t Dataset Letter Recognition Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing Letter Recognition   ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((P, T))
    test_dataset = tf.data.Dataset.from_tensor_slices((Ptest, Ttest))

    train_dataset = train_dataset.batch(P.shape[0])
    test_dataset = test_dataset.batch(Ptest.shape[0])
      
    print("-----------------------------------------------------------------")
    
    return train_dataset, test_dataset, (P, T),(Ptest, Ttest), [input_dim, num_classes, 'DATA']

def loadDataset_XOR(plot = False, to_categorical = False, normalize=False ):
    base_path = os.getcwd()

    dataset_path =  base_path + os.sep + "Dataset"+ os.sep + "XOR"+ os.sep + "X_OR_Gaussian2_dim.mat"
    data_set_name = "  XOR  "
        
    dict = loadmat( dataset_path )
    
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']
    
    if (normalize ):
        t_p = (t_p - np.mean(t_p))/np.std(t_p)
        t_t = (t_t - np.mean(t_t))/np.std(t_t)
    
        t_ptest = (t_ptest - np.mean(t_ptest))/np.std(t_ptest)
        t_ttest = (t_ttest - np.mean(t_ttest))/np.std(t_ttest)
    
    t_P = np.array( t_p, dtype = np.float32)
    P   = np.array( t_p, dtype = np.float32)
    
    t_T = np.array( t_t, dtype = np.int8 )
    T   = np.array( t_t, dtype = np.int8 )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    Ptest   = np.array( t_ptest, dtype = np.float32 )
    
    t_Ttest = np.array( t_ttest, dtype = np.int8 )
    Ttest   = np.array( t_ttest, dtype = np.int8 )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest
    
    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0]], np.int8 )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0]], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]], np.int8 )
    
    for idx in range( P.shape[0] ):
        P[idx ] =  t_P[ :, idx ] 
        T[idx] = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    input_dim = P.shape[1]

    num_classes =  np.max(T)
    
    T = T -1          # ESTO ES POR EL CATEGORICAL 
    Ttest = Ttest -1  # ESTO ES POR EL CATEGORICAL
    
    if ( to_categorical ):
        T = to_categorical( T , 2 )
        Ttest = to_categorical(Ttest, 2 )
    
    print(f"\t Dataset Loaded: {data_set_name}")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((P, T))
    test_dataset = tf.data.Dataset.from_tensor_slices((Ptest, Ttest))

    train_dataset = train_dataset.batch( P.shape[0] )
    test_dataset = test_dataset.batch(Ptest.shape[0])
    
    if plot:
        pu.plot_N_class( P, T )
    
    print("-----------------------------------------------------------------")

    return train_dataset, test_dataset, (P, T),(Ptest, Ttest), [input_dim, num_classes, 'DATA']

def loadDataset_Espiral_NClass_N_Loops ( n_class = 2, n_loops = 1, plot= False, normalize = False ):

    base_path = os.getcwd()
    data_set_name = ''

    if n_loops == 1 and n_class == 2:
        dataset_path =  base_path + os.sep +"Dataset"+ os.sep +"espiral"+ os.sep +"class_2"+ os.sep +"espiral_1.mat"
        data_set_name = "Spira 1 spin"
        
    if n_loops == 2 and n_class == 2:
        dataset_path =  base_path + os.sep +"Dataset"+ os.sep +"espiral"+ os.sep +"class_2"+ os.sep +"espiral_2.mat"
        data_set_name = "Spira 2 spin"
    
    if n_loops == 5 and n_class == 2:
        dataset_path =  base_path + os.sep + "Dataset"+ os.sep +"espiral"+ os.sep +"class_2"+ os.sep +"espiral_5.mat"
        data_set_name = "Spira 5 spin"
            
    if n_loops == 1 and n_class == 3:
        dataset_path =  base_path + os.sep +"Dataset"+ os.sep +"espiral"+ os.sep +"class_3"+ os.sep +"espiral_3_class_1.mat"
        data_set_name = "Spira 3 spin 1"
        
    if n_loops == 2 and n_class == 3:
        dataset_path =  base_path + os.sep + "Dataset"+ os.sep +"espiral"+ os.sep +"class_3"+ os.sep +"espiral_3_class_2.mat"
        data_set_name = "Spira 3 spin 2"
        
    dict = loadmat( dataset_path )
    
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']
    
    if ( normalize ):
        t_p = (t_p - np.mean(t_p))/np.std(t_p)
        t_ptest = (t_ptest - np.mean(t_ptest))/np.std(t_ptest)
    
    t_P = np.array( t_p, dtype = np.float32)
    P = np.array( t_p, dtype = np.float32)
    
    t_T = np.array( t_t, dtype = np.int8 )
    T = np.array( t_t, dtype = np.int8 )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    Ptest = np.array( t_ptest, dtype = np.float32 )
    
    t_Ttest = np.array( t_ttest, dtype = np.int8 )
    Ttest = np.array( t_ttest, dtype = np.int8 )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest
    
    P = np.zeros( [t_P.shape[1], t_P.shape[0] ], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0] ], np.int8 )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0] ], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]    ], np.int8 )
    
    for idx in range( P.shape[0] ): 
            P[idx ] = t_P[ :, idx ] 
            T[idx]  = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
            Ptest[ idx ] = t_Ptest[ :, idx ]
            Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    input_dim = P.shape[1]

    num_classes =  np.max(T)
    
    Tplot = T
    
    T = T -1          # ESTO ES POR EL CATEGORICAL 
    Ttest = Ttest -1  # ESTO ES POR EL CATEGORICAL
    
    if (  n_class > 2 ):
        T = to_categorical( T , n_class )
        Ttest = to_categorical(Ttest, n_class )
    
    print("-----------------------------------------------------------------")
    print(f"\t Dataset Loaded: {data_set_name}")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    
    print(f"\t Unique Count Train  --> { np.unique(T, return_counts=True) }" )
    print(f"\t Unique Count Test   --> { np.unique(Ttest, return_counts=True) }" )
    
    print("-----------------------------------------------------------------")
    
    train_dataset = tf.data.Dataset.from_tensor_slices((P, T))
    test_dataset = tf.data.Dataset.from_tensor_slices((Ptest, Ttest))

    if ( n_loops > 1):
        slices = tf.cast((P.shape[0]), tf.int64) 
        train_dataset = train_dataset.batch( slices )
        test_dataset = test_dataset.batch(Ptest.shape[0])
    else:
        #slices = tf.cast((P.shape[0]/2.0), tf.int64) 
        #train_dataset = train_dataset.batch( slices )
        
        train_dataset = train_dataset.batch( P.shape[0] )
        test_dataset = test_dataset.batch(Ptest.shape[0])
    
    if plot:
        pu.plot_N_class( P, Tplot )

    return train_dataset, test_dataset, (P, T),(Ptest, Ttest), [input_dim, num_classes, 'DATA']

def load_partArtifitialCharDataset (dataset_path,  to_categorical = True):
    
    print(" Loading Artifitial Char dataset .... ")
    
    T_temp = []
    P_temp = []
    
    for file in sorted (os.listdir( dataset_path )):
        file_str = str ( file )
        
        f = open( dataset_path +  os.sep + file_str , 'r')
        
        patron = []
        num_red = 0
        index = 1
        limit = 6 
        
        for line in f :   # un archivo por patron
            str_arr = line.split()
            patron.append( str_arr[3 :] )
            num_red = str_arr[1]
            
            if index == limit:
                break
            
            index += 1

        for miss in range( int(num_red)+1,  6 ):
             patron.append( [ 0, 0, 0, 0, 0, 0] )

        P_temp.append( np.reshape(patron, [1, 36]))
        
        if(str(file).startswith('a')):
            T_temp.append( 1 )
        if(str(file).startswith('c')):
            T_temp.append( 2 ) 
        if(str(file).startswith('d')):
            T_temp.append( 3 ) 
        if(str(file).startswith('e')):
            T_temp.append( 4 ) 
        if(str(file).startswith('f')):
            T_temp.append( 5 )
        if(str(file).startswith('g')):
            T_temp.append( 6 )  
        if(str(file).startswith('h')):
            T_temp.append( 7 )
        if(str(file).startswith('l')):
            T_temp.append( 8 )
        if(str(file).startswith('p')):
            T_temp.append( 9 )
        if(str(file).startswith('r')):
            T_temp.append( 10 )

    P_temp = np.array( P_temp, dtype = np.float32 )
    T_temp = np.array( T_temp, dtype = np.int8 )
    
    P_temp = np.squeeze( P_temp )

    return P_temp, T_temp

def loadArtifitialCharDataset (to_categorial = True):
    
    base_path = os.getcwd()
    
    dataset_path_train = base_path + os.sep + "Dataset"+ os.sep +"ArtificialCharactersDataSet"+ os.sep +"murphy"+ os.sep +"train"+ os.sep
    dataset_path_test  = base_path + os.sep + "Dataset"+ os.sep +"ArtificialCharactersDataSet"+ os.sep +"murphy"+ os.sep +"test"  + os.sep 

    P, T = load_partArtifitialCharDataset( dataset_path_train )
    Ptest, Ttest = load_partArtifitialCharDataset(dataset_path_test)
        
    input_dim = P.shape[1]
    
    P = preproc.scale(P)                                 
    Ptest = preproc.scale(Ptest)                # 0.9770        

    if ( to_categorial ):
        T = T -1
        Ttest =  Ttest -1
    
        T = to_categorical( T , 10)
        Ttest = to_categorical(Ttest, 10)
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((P, T))
    test_dataset = tf.data.Dataset.from_tensor_slices((Ptest, Ttest))
    
    train_dataset = train_dataset.batch( P.shape[0] )
    test_dataset = test_dataset.batch(Ptest.shape[0])
    
    print("-----------------------------------------------------------------")
    
    print("\t Dataset Artifitial Character Dataset  Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing Artifitial Character  ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    
    print(f"\t Unique Count Train  --> { np.unique(T, return_counts=True) }" )
    print(f"\t Unique Count Test   --> { np.unique(Ttest, return_counts=True) }" )
    
    print (" Done Loading Artifitial Character Dataset ... ")
    
    print("-----------------------------------------------------------------")
    return train_dataset, test_dataset, (P, T),(Ptest, Ttest), [input_dim, 10, 'DATA']



