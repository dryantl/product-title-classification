import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import string
import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class preprocessing:
    
	# Dimension adalah dimensi vektor embedding yang digunakan
	# Embedder adalah object fasttext dari library gensim
    def __init__(self,dimension,embedder):
        self.dimension=dimension
        self.embedder=embedder
	
	# Digunakan untuk menghitung jumlah masing2 kata
	# Input berupa list of list of token
	# Output berupa dictionary yang memetakan kata menjadi frekuensi kemunculan kata tersebut
    def word_count(self,sentences):
        counts = dict()
        for sentence in sentences:
            for word in sentence:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
        return counts
    
	# Melakukan filtering data berdasarkan kemunculan kata, jumlah karakter pada kata, jumlah kata pada kalimat
	# Kata dengah jumlah kemunculan < frequency akan dibuang
	# Kata dengan jumlah karakter < word_length akan dibuang
	# Kalimat dengan jumlah kata < N_words akan dibuang
	# Output berupa list of list of token(features) dan list of string(labels)
    def getFilteredData(self,product_title,labels,frequency, N_words, word_length):
        result=self.word_count(product_title)
        new_product_title=[]
        for sentence in tqdm.tqdm(product_title):
            new_product_title.append([word for word in sentence if result[word]>=frequency and len(word)>=word_length])
    
        new_features=[]
        new_labels=[]
        for index,title in tqdm.tqdm(enumerate(new_product_title)):
            if(len(title)>=N_words):
                new_features.append(title)
                new_labels.append(labels[index])
        
        return new_features,new_labels
    
	# Method untuk menghasilkan features berupa Tf-Idf dari input
	# Input berupa list of list of token
	# Output berupa vektor tfidf(list of list of real_value), object CountVectorizer, object TfidfTransformer
    def getTfIdf(self,new_product_title):
        concatenated_product_title=[]
        for sentence in tqdm.tqdm(new_product_title):
            concatenated_product_title.append(" ".join(sentence))
        cv=CountVectorizer()
        result=cv.fit_transform(concatenated_product_title)
        tftransformer = TfidfTransformer(smooth_idf=False)
        final_result=tftransformer.fit_transform(result)
        return final_result,cv,tftransformer
    
    # Method untuk menghapus angka dan tanda baca, serta melakukan tokenizing kata
	# Input berupa list of string
	# Output berupa list of list of token 
    def tokenize(self,input_string):
        input_string=''.join(i for i in input_string if not i.isdigit())
        result_string=input_string.lower()
        target_punctuations=string.punctuation
        for punctuation in target_punctuations:
            result_string=result_string.replace(punctuation, ' ')
        result_string=result_string.strip(' ').split()
        return result_string

    # Method untuk mengubah kata menjadi vektor fasttext
	# Input berupa token
	# Output berupa vektor dengan dimensi self.dimension
    def vectorize_word(self,product_title):
        try:
            result=self.embedder[product_title]
        except KeyError:
            result=0
        return result

    # Method untuk mengubah kalimat(list of token) menjadi vektor fasttext
	# Penggabungan vektor kata menjadi vektor kalimat menggunakan penjumlahan vektor
	# Vektor kata dapat diberi bobot berupa nilai tf-idf dari kata tersebut
	# doc_occ adalah dictionary yang memetakan kata menjadi jumlah kemunculan kata tersebut pada seluruh dokumen
	# total_doc adalah jumlah seluruh dokumen
	# Untuk referensi lebih lanjut dari doc_occ dan total_doc, silakan melihat rumus tf-idf
    def vectorize_sentence(self,input_sentence,doc_occ=None,total_doc=None):
        if(False):
            N_success=0
            result_vector=np.zeros(self.dimension)
            for word in input_sentence:
                result_vector+=self.vectorize_word(word)
                if(np.sum(self.vectorize_word(word))!=0):
                    N_success+=1

            if(N_success<2):
                result_vector=np.zeros(self.dimension)
            return result_vector
        
        else:
            N_success=0
            result_vector=np.zeros(self.dimension)
            ll=len(input_sentence)
            for word in input_sentence:
                
                c=0
                for word2 in input_sentence:
                    if(word==word2):
                        c+=1
                if(word in list(doc_occ)):
                    result_vector+=(self.vectorize_word(word)*((c/ll)*(np.log(total_doc/doc_occ[word]))))
                else:
                    result_vector+=(self.vectorize_word(word))
                                    
                if(np.sum(self.vectorize_word(word))!=0):
                    N_success+=1
            if(N_success<2):
                result_vector=np.zeros(self.dimension)
            return result_vector
            
	# Method yang merupakan pipeline preprocessing dari data mentah menjadi data siap pakai untuk klasifikasi
	# Input berupa list of string, list of string, dan object LabelEncoder(optional, jika ingin menggunakan object custom)
	# Output berupa pandas dataframe(features dan labels tergabung menjadi satu) dengan nama kolom "Labels" untuk labels 
	# ,nama kolom angka 1-100 untuk features, dan object LabelEncoder(jika user tidak menyediakan LabelEncoder custom)
    def preprocess_data(self,features,labels,encoder=None):
        embedded_data=pd.DataFrame()
        print("TOKENIZE DATA")
        embedded_data["Features"]=[self.tokenize(title) for title in tqdm.tqdm(features)]
        print("APPLYING FILTER")
        nf,nl=self.getFilteredData(embedded_data["Features"],list(labels),50,2,3)
        embedded_data=pd.DataFrame()
        embedded_data["Features"]=nf
        
        voc=set()
        for sentence in tqdm.tqdm(embedded_data["Features"]):
            for word in sentence:
                voc.add(word)
        
        total_doc=len(embedded_data["Features"])
        
        doc_occ={}
        for element in tqdm.tqdm(list(voc)):
            count_occ=0
            for sentence in embedded_data["Features"]:
                if (element in sentence):
                    count_occ+=1
            doc_occ[element]=count_occ
        
        print("ENCODING LABELS")
        if(encoder==None):
            label_encoder=LabelEncoder()
            embedded_data["Labels"]=label_encoder.fit_transform(nl)
        else:
            label_encoder=encoder
            embedded_data["Labels"]=label_encoder.transform(nl)
        
        print("CONVERTING SENTENCE TO VECTOR")
        embedded_data["Features Vector"]=[self.vectorize_sentence(title,doc_occ,total_doc) for title in tqdm.tqdm(embedded_data["Features"])]
    
        print("SAVE VECTOR TO PANDAS DATAFRAME")
        for i in tqdm.tqdm(range(self.dimension)):
            embedded_data[i]=[value[i] for value in embedded_data["Features Vector"]]
    
        embedded_data = embedded_data[[*range(self.dimension),"Labels"]]
        if(encoder==None):
            return embedded_data, label_encoder
        else:
            return embedded_data
			
	# Input berupa 2 list of string dan jumlah top N class yang diinginkan
	# Output berupa data dengan format sama seperti input tetapi hanya mengandung top N class		
    def getFilteredClasses(self,product_title,labels,top_N):
        print("1/3")
        sorted_by_value = sorted(self.class_count(labels).items(), key=lambda kv: kv[1])
        valid_class=[value[0] for value in sorted_by_value[-top_N:]]
        print("2/3")
        product_title=list(product_title)
        new_features=[]
        new_labels=[]
        for index,label in tqdm.tqdm(enumerate(labels)):
            if(label in valid_class):
                new_labels.append(label)
                new_features.append(product_title[index])
    
        return new_features,new_labels
	
    # Menghitung nilai Tf-Idf dari suatu kata
    # Input berupa nilai real yang dibutuhkan untuk menghitung Tf-Idf
    # Output berupa nilai Tf-Idf
    def tfidf_word(self,total_occ,total_words,doc_occ,total_doc):
        return (total_occ/total_words)*np.log(total_doc/doc_occ)
	
	# Method untuk menghasilkan features berupa Tf-Idf dari input tetapi hanya menggunakan kelas yang terdapat di vocab
	# Input berupa list of list of token dan list of string(vocab)
	# Output berupa vektor tfidf(list of list of real_value), object CountVectorizer, object TfidfTransformer
    def getTfIdfCustom(self,new_product_title,vocab):
        print("1/3")
        concatenated_product_title=[]
        for sentence in tqdm.tqdm(new_product_title):
            concatenated_product_title.append(" ".join(sentence))
        print("2/3")
        cv=CountVectorizer(vocabulary=vocab)
        result=cv.fit_transform(concatenated_product_title)
        print("3/3")
        tftransformer = TfidfTransformer(smooth_idf=False)
        final_result=tftransformer.fit_transform(result)
    
        return final_result,cv,tftransformer
		
    # Menghitung frekuensi kemunculan setiap kata dari list of list of token
    # Output berupa dictionary kata dan frekuensi kemunculannya
    def word_count(self,sentences):
        counts = dict()
        print("1/1")
        for sentence in sentences:
            for word in sentence:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
        return counts

    # Menghitung frekuensi kemunculan setiap kata dari list of token
    # Output berupa dictionary kata dan frekuensi kemunculannya
    def class_count(self,words):
        counts = dict()
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        return counts
		
	# Sama seperti method word_count, tetapi yang dihitung hanya data yang termasuk kelas target
    def word_count_label(self,sentences,labels,target):
        counts = dict()
        print("1/1")
        for index,sentence in enumerate(sentences):
            if(labels[index]==target):
                for word in sentence:
                    if word in counts:
                        counts[word] += 1
                    else:
                        counts[word] = 1
        return counts

        