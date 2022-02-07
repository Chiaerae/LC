
import sys
import nltk
import codecs
import math
from nltk import bigrams
from math import log
#per evitare di dover scrivere più volte queste liste le vado a dichiarare come globali in modo tale che dovrò solo richiamarle 
ListaS = ["NN", "NNS", "NP"] 

ListaV = ["VB","VBD","VBG","VBN","VBP","VBZ"]

ListaA = ["JJ", "JJR","JJS"]



def VerbiPiuFreq(TestoPOS):     #funzione che prende il input il testo annotato
     ListaVerbi = []            #creo una lista vuota dove andrò ad appendere i bigramma
     for bigramma in TestoPOS: #ciclo tutti i bigramma del corpus
           if bigramma[1] in ListaV: #se il bigramma è in ListaV appendo il bigramma nella listaverbi
                ListaVerbi.append(bigramma)
           FreqVerbi = nltk.FreqDist(ListaVerbi)   #calcolo la frequenza dei bigrammi della listaverbi
           DieciVerbi =FreqVerbi.most_common(20)  #estraggo i primi 20 con frequenza più alta
     return DieciVerbi

def SostantiviPiuFreq(TestoPOS):   #la funzione esegue la stessa cosa di quella precedente, solo che lo fa per i sostantivi
    ListaSostantivi= []
    ListaBigrammi = []
    for bigramma in TestoPOS:
            if bigramma[1] in ListaS:
                ListaSostantivi.append(bigramma)
            FreqSostantivi = nltk.FreqDist(ListaSostantivi)
            DieciSostantivi =FreqSostantivi.most_common(20)
    return DieciSostantivi
 
def Bigrammi(bigrammiTOT):    
    SV_bigrammi = [] #creo la lista dove andrò ad appendere i bigrammi sostantivo-verbo
    SA_bigrammi = []#creo la lista dove andrò ad appendere i bigrammi sostantivo-aggettivo
    for big in bigrammiTOT: 
        pos1 = big[0][1] #inserisco in pos1 il pos del primo token
        pos2 = big[1][1] #inserisco in pos2 il pos del secondo token
        if pos1 in ListaS and pos2 in ListaV:  #condiozione: se il primo token è un sostantivo (appartiene alla listaS) e il secondo appartiene alla listaV
            SV_bigrammi.append(big) #appendo il bigramma alla listaSV
        elif pos1 in ListaS and pos2 in ListaA: #condizione: se il primo token è un sostantivo e il secondo un aggettivo
            SA_bigrammi.append(big) #appendo il bigramma alla listaSA
    
    DistBigrammiSV = nltk.FreqDist(SV_bigrammi)  #calcolo la frequenza delle due liste
    DistBigrammiSA = nltk.FreqDist(SA_bigrammi)
    DistBigrammiOrdSV = DistBigrammiSV.most_common(20)
    DistBigrammiOrdSA = DistBigrammiSA.most_common(20) #estraggo i venti bigrammi con frequenza più alta
    print()
    print()
    for el in DistBigrammiOrdSV: #ciclo le tuple per stampare i risultati
        token1 = el[0][0][0]
        token2 = el[0][1][0]
        freq = el[1]
        print("SOSTANTIVO:",token1,"\t\t\tVERBO:",token2,"\t\t\tFREQUENZA:", freq)
    print()

    return DistBigrammiOrdSV, DistBigrammiOrdSA  
        




def AnnotazioneLinguistica(frasi):  #funzione che prende il input tutte le frasi del corpus
    tokensTOT = []  #creo le varie liste 
    tokensPOStot = []
    bigrammiPOSTOT = []
    distPOSordinate = []
    for frase in frasi:  #itero le frasi
        tokens = nltk.word_tokenize(frase) #estraggo i tokens dalle frasi
        tokensPOS = nltk.pos_tag(tokens) #estraggo i token
        tokensPOStot = tokensPOStot + tokensPOS #appendo i pos estratti alla lista tokensPOStot
        bigrammi_frase = bigrams(tokensPOS) #estraggo i bigrammi dalla frase
        bigrammiPOSTOT += bigrammi_frase  #appendo i bigrammi estratti alla lista bigrammiPOSTOT
        tokensTOT += tokens #appendo i tokens estratti alla lista tokensTOT
    distPOS = nltk.FreqDist(tokensPOStot) #estraggo le frequenze da tokensPOStot
    distPOSordinate = distPOS.most_common(20) #prendo i primo venti pos con frequenza maggiore
    for pos in distPOSordinate:  #ciclo la lista distPOSordinate e stampo i token con il pos corrispondente e la frequenza
        print("TOKEN:",pos[0][0],"\t\tPOS:", pos[0][1],"\t\tFREQUENZA:", pos[1])
    return tokensTOT, tokensPOStot,bigrammiPOSTOT #restituisco al main tre liste
        
def Ordina(Lista):
     return sorted(Lista, reverse = True)   #la funzione prende in input una lista e la ordina
     
def EstraiNE(frasi, file):
       tokensTOT = [] # creo la lista dove andrò a mettere il totale dei token
       tokensPOSTOT = [] #creo la lista dove metto il totale dei token annotati
       ListaPersona = [] #creo la lista dove andranno ne Ne Persona
       ListaLuoghi = [] #creo la lista dove andranno ne Ne Luoghi
       
       
       dizionario_NE = {} #creo un dizionario dove andrò ad appendere le Named Entity
       for frase in frasi: #ciclo le frasi
            tokens = nltk.word_tokenize(frase) #estraggo i tokens
            tokensTOT = tokensTOT+tokens #appendo i tokens alla lista tokensTOT
            tokensPOS = nltk.pos_tag(tokens) #estraggo i tokens annotati
            tokensPOSTOT+= tokensPOS #appendo i tokens annotati alla lista tokensPOSTOT
            analisi = nltk.ne_chunk(tokensPOS) #Applico la Named Entity Chunk per ricavare le informazioni
            
            
            
            
            for nodo in analisi: #ciclo l'albero scorrendo i nodi
              nome = ""
              luoghi = ""
              
              if hasattr(nodo, 'label'): #controlla se è un nodo intermedio o una foglia
                   
                     if nodo.label() not in dizionario_NE.keys():  #vado a creare la condizioni tale che se il dizionario è vuoto lo inizializzo per la prima volta
                            dizionario_NE[nodo.label()] = 1
                     else:
                             dizionario_NE[nodo.label()] += 1 #in caso non sia vuoto ogni volta che all'interno del mio corpus trovo una NE di quel tipo la incremento
                     
              if hasattr(nodo, 'label'): #controlla se è un nodo intermedio o una foglia
                     
                     if nodo.label()  in ['PERSON']:       #estraggo l'etichetta della ne dal nodo
                            for parteNE in nodo.leaves():      #ciclo le foglie del nodo selezionato
                                
                                   nome = nome + '' + parteNE[0] #concateno il nome alla lista 
                            ListaPersona.append(nome)          #appendo il risultato alla lista finale
                     elif nodo.label() in ['GPE']:          #eseguo la stessa operazione per trovare tutte le NE GPE
                            for parteNE in nodo.leaves():
                                   luoghi = luoghi + '' + parteNE[0]
                            ListaLuoghi.append(luoghi) #appendo il risultato alla lista finale
       print()
       print()
      
     
       print("NE PRESENTI NEL FILE", file,":") #stampo il dizionario contenente le NE
       print(dizionario_NE)
       
      
       
       frequenza = nltk.FreqDist(ListaPersona) #cacolo la frequenza della lista dei nomi
       nomitrovati = frequenza.most_common(15)#prendo i primi 15 con frequenza maggiore
       print()
       print("15 NOMI PROPRI PIU' FREQUENTI CON RELATIVA FREQUENZA DEL FILE", file, ":")
       frequenzaL = nltk.FreqDist(ListaLuoghi) #cacolo la frequenza della lista dei luoghi
       luoghitrovati = frequenzaL.most_common(15)#prendo i primi 15 con frequenza maggiore
       for elem in nomitrovati:
            print("NOME:", elem[0], "\t\t\t\t", "FREQUENZA", elem[1]) #ciclo la lista per stamparla
       print()
       print("15 LUOGHI PIU' FREQUENTI CON RELATIVA FREQUENZA DEL FILE", file, ":")
      
       for el in luoghitrovati:       #ciclo la lista per stamparla
            print("LUOGO:", el[0], "\t\t\t\t", "FREQUENZA", el[1])
            



def Calcolostatistichebigrammi(bigrammi,tokens):
     bigrammidiversi = list(set(bigrammi)) #vado a creare una lista con tutti i bigrammi prendendoli solo una volta
     probCongiuntaMAX = 0.0 #creo una variabile che la inizializzo a zero nella quale andrà poi il valore massimo della probabilita congiunta
     probCondizionataMAX = 0.0 #creo una variabile che la inizializzo a zero nella quale andrà poi il valore massimo della probabilita condizionata
     LMI = 0.0 #creo una variabile che la inizializzo a zero nella quale andrà poi il valore massimo della LMI
     ListaprobCondizionata = [] #creo delle liste 
     ListaprobCongiunta = []
     ListaLMI = []
     
     for bigramma in bigrammidiversi:
          token1 = bigramma[0]
          token2 = bigramma[1]
          freqtoken1 = tokens.count(token1) #calcolo la frequenza del token nel corpus
          freqtoken2 = tokens.count(token2)
          freqbigramma = bigrammi.count(bigramma) #frequenza oesservata del bigramma
          if freqtoken1 > 3 and freqtoken2 > 3: #se la frequenza dei due token è maggiore di 3 allora eseguo le operazioni successivo in caso contrario non vengono eseguite le operazioni e si passa al bigrama successivo
               probCondizionata = (freqbigramma*1.0)/(freqtoken1*1.0) #formula probabilità condizionata
               ListaprobCondizionata.append([probCondizionata,bigramma])#appendo il risultato alla lista
               probToken1 = freqtoken1*1.0/len(tokens)*1.0 #probabilità del token è data dalla sua frequenza/ numero di tokens
               probCongiunta = probCondizionata * probToken1 #formula della probabilità congiunta data dal prodotto fra la probabilità condizionata e la probabilità del primo token
               ListaprobCongiunta.append([probCongiunta,bigramma])#appendo il risultato alla lista
               ListaprobCondizionata = Ordina(ListaprobCondizionata)#richiamo la funzione che ordina la lista
               ListaprobCongiunta = Ordina(ListaprobCongiunta) #richiamo la funzione che ordina la lista
               FA = (freqtoken1 * freqtoken2)*1.0/len(tokens)*1.0 #calcolo la frequenza attesa
               LMI = (freqbigramma*1.0)*math.log((freqbigramma*1.0)/(FA*1.0),2) #formula LMI
               ListaLMI.append([LMI,bigramma]) #appendo il risultato alla lista
               ListaLMI = Ordina(ListaLMI) #richiamo la funzione che ordina la lista
               
     return ListaprobCongiunta[0:20], ListaprobCondizionata[0:20], ListaLMI[0:20]

def stampaListe(lista): #funzione che prende in input una lista
     for elem in lista: #ciclo la lista e la stampo
          print("PROBABILITA':",elem[0],"\tBIGRAMMA:","\t", elem[1])   

def CalcolaProbabilitaMarkov1(LunghezzaCorpus, DistribuzioneDiFrequenzaToken, DistribuzioneDiFrequenzaBigrammi, BigrammiFrase, VocabolarioLunghezza):
        token1 = BigrammiFrase[0][0] #inserisco nella variabile token il primo token 
        probabilita = (( DistribuzioneDiFrequenzaToken[token1]*1.0)+1)/((LunghezzaCorpus*1.0)+ VocabolarioLunghezza*1.0) #inserisco nella variabile probabilità calcolata sul primo token applicando la add one smoothing
        for bigramma in BigrammiFrase:# ciclo i bigrammi 
                FreqBigramma = (DistribuzioneDiFrequenzaBigrammi[bigramma]) #inserisco nella variabile la frequenza del bigramma
                frequenzaA = DistribuzioneDiFrequenzaToken[bigramma[0]] #inserisco nella varibile la frequenza del primo token 
                probCondizionata = ((FreqBigramma*1.0)+1)/((frequenzaA*1.0)+ VocabolarioLunghezza*1.0) #calcolo la probabilita sul bigramma
                probabilita = probabilita*probCondizionata #inserisco nella variabile il prodotto tra la probabilità calcolata nel ciclo precedente e la probabilità calcolata sul bigramma corrente
        return probabilita #restituisco la probabilità
     #FORMULA MARKOV ORDINE 1 + ADD ONE SMOOTHING
     #(f(primotoken)+1)/(|c|+|v|)*(f(bigramma1)+1)/(f(primotoken)+|v|)*....*f(bigramma_n)+1)/(f(token_n-1)





def Markov1(tokensTOT,frasi): 
     Vocabolario = list(set(tokensTOT)) #calcolo il vocabolario
     VocabolarioLunghezza = len(Vocabolario) #calcolo la lunghezza del vocabolario
     LunghezzaCorpus = len(tokensTOT) #cacolo la lunghezza del corpus
     bigrammi = list(bigrams(tokensTOT)) #calcolo i bigrammi
     probabilitaMAX = 0 #creo una variabile alla quale assegno il valore zero 
     fraseMAXPRO = [] #creo una lista vuota dove andrò a mettere la frase con probabilità maggiore 
     DistribuzioneDiFrequenzaToken = nltk.FreqDist(tokensTOT)# calcolo la frequenza dei vari tokens
     DistribuzioneDiFrequenzaBigrammi = nltk.FreqDist(bigrammi) #calcolo la frequenza dei bigrammi
     for frase in frasi: #ciclo le frasi 
          tokensFrase = nltk.word_tokenize(frase) #estraggo i token dalla frase corrente
          if (len(tokensFrase) > 7 and len(tokensFrase) < 16): #se la frase ha dagli 8 ai 15 tokens il programma esegue le condizioni dell'if in caso contrario passa alla frase successiva
                    BigrammiFrase = list(bigrams(tokensFrase)) 
                    probabilita1 = CalcolaProbabilitaMarkov1(LunghezzaCorpus, DistribuzioneDiFrequenzaToken, DistribuzioneDiFrequenzaBigrammi, BigrammiFrase, VocabolarioLunghezza)
                    if probabilitaMAX < probabilita1: #se la probabilita alla quale avevo assegnato valore massimo nel ciclo precedente è minore della probabilita appena trovata vado a sostituirla
                         probabilitaMAX = probabilita1 #sostituisco
                         fraseMAXPROB = frase #sostituisco anche la frase
     print("La probabilità Condizionata della frase", fraseMAXPROB, "è quella con probabilità maggiore di tutto il Corpus", probabilitaMAX)



def main(file1,file2):
    fileInput1 = open(file1, mode="r", encoding="utf-8")#prende il file codificato in utf-8
    raw1= fileInput1.read() #viene letto il file e viene assegnato tutto il suo contenuto ad una variabile di tipo string
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    raw2= fileInput2.read() #viene letto il file e viene assegnato tutto il suo contenuto ad una variabile di tipo string

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #viene caricato il modello statistico
                                                     #su cui ci si basa per dividere il testo in frasi.
    frasi1 = sent_tokenizer.tokenize(raw1) #divide le frasi in token
    frasi2 = sent_tokenizer.tokenize(raw2)
    print()
    
    #STAMPO LE 10 POS (PART OF SPEECH) PIU FREQUENTI
    print("I 10 POS PIU' FREQUENTI DEL TESTO", file1, " SONO:")
    print()
    TokensTOT1,TokensPOStot1,bigrammiPOSTOT1 = AnnotazioneLinguistica(frasi1)
    print()
    print("I 10 POS PIU' FREQUENTI DEL TESTO", file2, " SONO:")
    print()
    TokensTOT2,TokensPOStot2,bigrammiPOSTOT2 = AnnotazioneLinguistica(frasi2)
    DieciSostantivi1 = SostantiviPiuFreq(TokensPOStot1)
    DieciSostantivi2 = SostantiviPiuFreq(TokensPOStot2)
    DieciVerbi1 = VerbiPiuFreq(TokensPOStot1)
    DieciVerbi2 = VerbiPiuFreq(TokensPOStot2)
    print()

    #STAMPO DI 20 SOSTANTIVI PIU' FREQUENTI
    print("VENTI SOSTANTIVI PIU' FREQUENTI DEL TESTO", file1, "SONO:")
    print()
    for S1 in DieciSostantivi1:
        print("SOSTANTIVO:", S1[0][0],"\t\t\t","FREQUENZA:", S1[1])
    print()
    
    print("VENTI SOSTANTIVI PIU' FREQUENTI DEL TESTO", file2, "SONO:")
    print()
    for S2 in DieciSostantivi2:
        print("SOSTANTIVO:",S2[0][0],"\t\t\t","FREQUENZA:", S2[1])
    print()
    #STAMPO I VENTI VERBI PIU' FREQUENTI
    print("VENTI VERBI PIU' FREQUENTI DEL TESTO", file1, "SONO:")
    print()
    for V1 in DieciVerbi1:
        print("VERBO:",V1[0][0],"\t\t\t","FREQUENZA:", V1[1])
    print()    
    print("VENTI VERBI PIU' FREQUENTI DEL TESTO", file1, "SONO:")
    print()
    for V2 in DieciVerbi2:
        print("VERBO:",V2[0][0],"\t\t\t","FREQUENZA:", V2[1])
    bigrammiTOT1 =  list(bigrams(TokensTOT1))
    Calcolostatistichebigrammi(bigrammiTOT1,TokensTOT1)
    bigrammiTOT2 =  list(bigrams(TokensTOT2))
    
   #STAMPO I 20 BIGRAMMI PIU' FREQUENTI SOSTANTIVO + VERBO
    print()
    print("I 20 BIGRAMMI PI' FREQUENTI SOSTANTIVO + VERBO:", file1) 
     
    DistBigrammiOrdSV1, DistBigrammiOrdSA1 = Bigrammi(bigrammiPOSTOT1)
    print()
    print("I 20 BIGRAMMI PI' FREQUENTI SOSTANTIVO + VERBO:", file2)
    print()
    DistBigrammiOrdSV2, DistBigrammiOrdSA2 = Bigrammi(bigrammiPOSTOT2)
    #stampo i bigrammi ordinati con le frequenze:
    print()
    #STAMPO I 20 BIGRAMMI PIU' FREQUENTI SOSTANTIVO + AGGETTIVO
    print("I 20 BIGRAMMI PI' FREQUENTI SOSTANTIVO + AGGETTIVO:", file1)
    print()
    for el in DistBigrammiOrdSA1:
        token1 = el[0][0][0]
        token2 = el[0][1][0]
        freq = el[1]
        print("SOSTANTIVO:",token1,"\t\t\t\t\tAGGETTIVO:",token2,"\t\t\t\t\t FREQUENZA:", freq)
    print()
    
    print("I 20 BIGRAMMI PI' FREQUENTI SOSTANTIVO + AGGETTIVO:", file2)
    print()
    for el1 in DistBigrammiOrdSA2:
        token3 = el1[0][0][0]
        token4 = el1[0][1][0]
        freq1 = el1[1]
        print("SOSTANTIVO:",token3,"\t\t \t\t\tAGGETTIVO:",token4,"\t\t\t\t\t FREQUENZA:", freq1)
    ListaprobCongiunta1, ListaprobCondizionata1, ListaLMI1=Calcolostatistichebigrammi(bigrammiTOT1,TokensTOT1)
    ListaprobCongiunta2, ListaprobCondizionata2, ListaLMI2=Calcolostatistichebigrammi(bigrammiTOT2,TokensTOT2)
    print()
    #STAMPO I 20 BIGRAMMI CON PROBABILITA' CONGIUNTA MASSIMA 
    print("PROBABILITA' CONGIUNTA DEL FILE", file1, ":")
    print()
    stampaListe(ListaprobCongiunta1)
    print()
    print("PROBABILITA' CONGIUNTA DEL FILE", file2, ":")
    print()
    stampaListe(ListaprobCongiunta2)
#STAMPO I 20 BIGRAMMI  CON PROBABILITA' CONDIZIONATA MASSIMA
    print()
    print("PROBABILITA' CONDIZIONATA DEL FILE", file1, ":")
    stampaListe(ListaprobCondizionata1)
    print()
    print("PROBABILITA' CONDIZIONATA DEL FILE", file2, ":")
    stampaListe(ListaprobCondizionata2)
    print()
    #STAMPO LA LOCAL MUTUAL INFORMATION SUI 20 BIGRAMMI CON LMI MAGGIORE
    print("LMI  DEL FILE", file1, ":")
    print()
    for elemento in ListaLMI1: #ciclo la lista e la stampo
          print("FORZA ASSOCIATIVA:",elemento[0],"\tBIGRAMMA:","\t", elemento[1])   
    print()
    print("LMI DEL FILE", file2, ":")
    print()
    for elemento2 in ListaLMI2: #ciclo la lista e la stampo
          print("FORZA ASSOCIATIVA:",elemento2[0],"\tBIGRAMMA:","\t", elemento2[1])  
    
    
 
    
    

    
   





#STAMPO MODELLO DI MARKOV DI ORDINE 1 + ADD ONE SMOOTHING
    print()
    print("MODELLO DI MARKOV DI ORDINE 1+ ADD ONE SMOOTHING SU", file1)
    print()
    Markov1(TokensTOT1,frasi1)
    print()
    print()
    print("MODELLO DI MARKOV DI ORDINE 1+ ADD ONE SMOOTHING SU", file2)
    print()
    Markov1(TokensTOT2,frasi2)
#STAMPO I 15 NOMI DI PERSONA PIU' FREQUENTI
    EstraiNE(frasi1, file1)
    EstraiNE(frasi2,file2)



    #STAMPO I 15 LUOGHI PIU' FREQUENTI

main(sys.argv[1], sys.argv[2])
