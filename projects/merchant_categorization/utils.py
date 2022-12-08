import pickle
import numpy as np
import string

filename = 'projects/merchant_categorization/merchant_model_final.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

filenameidf = 'projects/merchant_categorization/idf_result.pkl'
loaded_dict = pickle.load(open(filenameidf, 'rb'))

tfidfcolumnsfile = 'projects/merchant_categorization/tfidfcolumns_list.pkl'
loaded_column_names = pickle.load(open(tfidfcolumnsfile, 'rb'))

dict_map_label = 'projects/merchant_categorization/mapping_dictionary_label.pkl'
loaded_map = pickle.load(open(dict_map_label, 'rb'))

def tfidf_array(merchant): #mengubah nama merchant menjadi representasi angka
    list_tfidf = []
    for x in loaded_column_names:
        tfidf = 0
        if x in merchant:
            idf = loaded_dict[x]
            count = merchant.count(x)
            tfidf = np.log(count+1)*idf
        list_tfidf.append(tfidf)
    return list_tfidf    

def merchant_predict(merchant_name):
    x_pred = tfidf_array(merchant_name)
    label_predict = loaded_model.predict([x_pred])[0] #predict label
    return loaded_map[label_predict] #mapping ke cat

def merchant_clean(merchant_name):
    merchant_name = merchant_name.lower()
    merchant_name = merchant_name.replace(".", " ")
    merchant_name = merchant_name.replace(",", " ")
    merchant_name = merchant_name.replace(";", " ")
    merchant_name = merchant_name.replace(" & ", " ")
    merchant_name = merchant_name.replace(" - ", " ")
    merchant_name = merchant_name.replace("-", " ")
    merchant_name = merchant_name.replace("_", " ")
    merchant_name = merchant_name.replace("(", " ")
    merchant_name = merchant_name.replace(")", " ")
    merchant_name = merchant_name.replace("/", " ")
    merchant_name = merchant_name.replace('"', "")
    merchant_name = merchant_name.replace('\t', "")
    merchant_name = merchant_name.replace("  ", " ")
    merchant_name = merchant_name.translate(str.maketrans('', '', string.punctuation))
    merchant_name = merchant_name.strip()
    return merchant_name
