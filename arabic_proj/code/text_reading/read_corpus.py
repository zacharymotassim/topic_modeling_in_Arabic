import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from pprint import pprint

# "/Users/zacharymotassim/Desktop/arabic_proj/data/arabic_big_corpus-master/arabic_big.txt"

# this function takes a string @path, opens the txt file at that path and returns it in the form of a list
# of lists, it needs to be a list of lists because Gensim's id2word object needs it in that format
def corpus_to_list(path:str)->list:
    data=list() # list to hold corpus
    corpus=open(path, "r") # open file
    for line in corpus:
        data.append(line.split()) # split each line by space and extend it  into  our data list
    return data

# removes stop words. @language must be lowercase and a language offered by NLTK's stopwords library
def remove_stop_words(data:list,language:str)->list:
    stop_words=set(stopwords.words(language)) # convering stop words list into a hash table for faster finding of words we do not want
    return [[word for word in line if word not in stop_words] for line in data]

#id2word = corpora.Dictionary(data)
#print(id2word)

def get_id_to_word(data:list):
    return corpora.Dictionary(data)
# creates a corpus using inpu data and gensims id2word fucntionality
def term_doc_freq(data:list,dictionary):
    corpus=[dictionary.doc2bow(array) for array in data]
    return corpus

def build_LDA_model(corpus,id2word,topic_count):
    return gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=topic_count)

def main():
    l=corpus_to_list("/Users/zacharymotassim/Desktop/arabic_proj/data/arabic_big_corpus-master/arabic_big.txt")
    #before=len(l)
    l=remove_stop_words(l,'arabic')
    d=get_id_to_word(l)
    corpus=term_doc_freq(l,d)
    num_topics=10
    lda_model=gensim.models.LdaMulticore(corpus=corpus,id2word=d,num_topics=num_topics)
    #pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    print(doc_lda)
if __name__ == "__main__":
    main()
