
class preprocessing_pipeline:
    # fungsi untuk menghapus karakter tidak penting
    def remove_parentheses(input_string):
        result_string=input_string.lower()
        target_parentheses=['-','/','[',']','!','(',')',',','.','+','-',"'",'"',"|","*","@","#","!","<",">",":",";","?"]
        for parentheses in target_parentheses:
            result_string=result_string.replace(parentheses, ' ')
        result_string=result_string.strip(' ').split()
        return result_string

    # fungsi untuk mengubah kata menjadi vektor
    def vectorize_word(product_title,embedder):
        try:
            result=embedder[product_title]
        except KeyError:
            result=0
        return result

    # fungsi untuk mengubah kalimat menjadi vektor
    def vectorize_sentence(input_sentence,dimension,embedder):
        result_vector=np.zeros(dimension)
        for word in input_sentence:
            result_vector+=vectorize_word(word,embedder)
        return result_vector

    def preprocess_data(features,labels,dimension,embedder):
        label_encoder=LabelEncoder()
        embedded_data=pd.DataFrame()
        embedded_data["Labels"]=label_encoder.fit_transform(labels)
        embedded_data["Features"]=[remove_parentheses(title) for title in features]
        embedded_data["Features Vector"]=[vectorize_sentence(title,dimension,embedder) for title in embedded_data["Features"]]
    
        for i in range(dimension):
            embedded_data[i]=[value[i] for value in embedded_data["Features Vector"]]
    
        embedded_data = embedded_data[[*range(dimension),"Labels"]]
        return embedded_data, label_encoder
