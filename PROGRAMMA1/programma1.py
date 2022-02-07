
import sys
import codecs
import nltk
import math
from nltk import bigrams, trigrams


def NumeroCaratterii(tokens):    #funzione che prende in input il numero totale dei tokens 
     conta = 0
     for tok in tokens: #ciclo i vari tokens e di ognuno vado a prendere il numero di caratteri e lo inserisco nella variabile conta
         conta = conta + len(tok) #alla fine del ciclo avrò il numero totale di caratteri del corpus
     return conta


    
def TypeTokenRatio(Tokens):
    Vocabolario = []  #creo la lista Vocabolario 
    Vocabolario = Tokens[0:5000] #calcolo il Vocabolario sui primi 5000 tokens
    Vocabolario = list(set(Vocabolario)) #vado a prendere tutti i tokens una volta sola
    Vocabolario = list(sorted(Vocabolario)) #vado a ordinare la mia lista Vocabolario
    VocabolarioLunghezza = len(Vocabolario)  #ottengo la lunghezza del Vocabolario
    ttr = VocabolarioLunghezza*1.0/len(Tokens)*1.0 #calcolo la Type Token Ratio
    return ttr







def CalcolaFrequenza(tokens):
    hapax = [] #creo una lista vuota dove andrò a mettere tutti i tokens con frequenza=1
    Freq5 = []#creo una lista vuota dove andrò a mettere tutti i tokens con frequenza=5
    Freq10 = [] #creo una lista vuota dove andrò a mettere tutti i tokens con frequenza=10
    print("ANDAMENTO LESSICO:")
    for i in range (0, len(tokens), 500): 
          listaTokens = tokens[0:i + 500] #si crea una sottolista ogni 500 tokens letti
          vocabolario = list(set(listaTokens)) #calcolo il vocabolario ogni volta che aggiungo 500 tokens
          for tok in vocabolario: 
             conteggio = tokens.count(tok)
             if conteggio == 1: # se il token viene conteggiato una sola volta all'interno del testo lo aggiungo alla lista hapax
                 hapax.append(tok)
             elif conteggio == 5: # se il token viene conteggiato una cinque volte all'interno del testo lo aggiungo alla lista  |V5|
                 Freq5.append(tok)
             elif conteggio == 10: # se il token viene conteggiato una dieci volte all'interno del testo lo aggiungo alla lista  |V10|
                 Freq10.append(tok)
          print() #stampo a ogni ciclo il numero delle varie frequenze 
          print ("Numero tokens:",len(listaTokens))
          print("HAPAX =",len(hapax))
          print("|V5| = ",  len(Freq5))
          print("|V10| = ",  len(Freq10))
          print()
 
def AnnotazioneLinguistica(frasi): #prendo in input la lista totale dei tokens
    tokensPOStot = [] #creo una lista vuota 
    for frase in frasi:
         tokens = nltk.word_tokenize(frase) #estraggo i tokens dalla frase
         tokensPOS = nltk.pos_tag(tokens) #appendo i token con i relativi pos
         tokensPOStot = tokensPOStot + tokensPOS #appendo i pos estratti alla lista tokensPOStot
    
    return tokensPOStot #restituisco la lista al main


def EstraiSequenza(tokensPOS):
     ListaPosVerbi = [] #Lista in cui inserisco i verbi
     ListaVerbi = ["VB","VBD","VBG","VBN","VBZ"]  #creo una lista con tutti i pos tag verbi
     ListaSostantivi = ["NN", "NNS", "NNP", "NNPS"]#creo una lista con tutti i pos tag dei sostantivi 
     ListaAvverbi = ["RB","RBR","RBS"]#creo una lista con tutti i pos tag degli avverbi
     ListaAggettivi = ["JJ", "JJR","JJS"]#creo una lista con tutti i pos tag degli aggettivi
     ListaPosAggettivi = [] #Lista in cui inserisco gli aggettivi
     ListaPosSostantivi = [] #Lista in cui inserisco i sostantivi
     ListaPosAvverbi = [] #Listain cui inserisco gli avverbi
     for bigramma in tokensPOS: 
          if bigramma[1] in ListaVerbi: #se il bigramma ha come pos uno di quelli nella lista appendo il token alla lista
               ListaPosVerbi.append(bigramma[0]) #eseguo questo controllo su tutte e quattro le liste
          elif bigramma[1] in ListaSostantivi:
               ListaPosSostantivi.append(bigramma[0])
          elif bigramma[1] in ListaAvverbi:
               ListaPosAvverbi.append(bigramma[0])
          elif bigramma[1] in ListaAggettivi:
               ListaPosAggettivi.append(bigramma[0])
     
     return (ListaPosSostantivi, ListaPosVerbi,ListaPosAggettivi,ListaPosAvverbi) #rimando al main le liste che contengono rispettivamente tutti i sostantivi, tutti i verbi, tutti gli aggettivi
#e tutti gli avverbi
                

    

def EliminazionePunteggiatura(tokensPOS):
    TOT = []
    ListaPunteggiatura = [".",","] #creo una lista con i segni di punteggiatura che dovrò andare a sottrarre al resto dei pos
    for bigramma  in tokensPOS:
        if not bigramma[1] in ListaPunteggiatura:#se il pos non è contenuto nella lista appendo il token alla lista TOT
               TOT.append(bigramma[0])
        
    return TOT #restuisco al main la lista con tutti i token esclusi quelli con pos (. ,)
  
     


def CalcolaLunghezza(frasi):
    lunghezzaTOT = 0.0
    numeroFrasi = 0.0 
    tokensTOT = [] #creo la lista dove andrò a mettere il totale dei tokens del corpus
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        #calcolo la lunghezza
        lunghezzaTOT= lunghezzaTOT+len(tokens) #aggiungo il numero di tokens a lunghezzaTOT---> alla fine del ciclo avrò il numero totale dei tokens del corpus
        tokensTOT = tokensTOT+ tokens
        #restituisco al main il risultato cioè la lunghezza tootale delle frasi cioè quanti tokens ha il corpus e la lista che contiene tutti i tokens del corpus
    return lunghezzaTOT,tokensTOT
 

def main(file1, file2):
    
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    raw1= fileInput1.read() #viene letto il file e viene assegnato tutto il suo contenuto ad una variabile di tipo string
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    raw2= fileInput2.read() #viene letto il file e viene assegnato tutto il suo contenuto ad una variabile di tipo string
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #viene caricato il modello statistico
                                                     #su cui ci si basa per dividere il testo in frasi.

    frasi1 = sent_tokenizer.tokenize(raw1) #divide le frasi in token
    frasi2 = sent_tokenizer.tokenize(raw2) #divide le frasi in token
    #chiamo la funzione CalcolaLunghezza per entrambi i testi
    lunghezza1, tokens1 = CalcolaLunghezza(frasi1) #richiamo la funzione CalcolaLunghezza che prende in input tutte le frasi del corpus e mi restituisce i tokens e il loro numero.
    lunghezza2, tokens2  = CalcolaLunghezza(frasi2)
    ttr1 = TypeTokenRatio(tokens1) #vado a richiamare la funzione che mi calcola la TTR prendendo in input la lista contenente tutti i tokens del corpus
    ttr2 = TypeTokenRatio(tokens2)
    tokensPOS1 = AnnotazioneLinguistica(frasi1) #vado a richiamare la funzione che prende in input tutti i tokens del corpus e restituisce i tokens annotati
    tokensPOS2 = AnnotazioneLinguistica(frasi2)
    NoPunteggiatura1 = EliminazionePunteggiatura(tokensPOS1) #vado a richiamare la funzione che prende in input tutti i tokens annotati e mi restituisce una lista con tutti i tokens esclusi quelli con pos (,.)
    NoPunteggiatura2 =  EliminazionePunteggiatura(tokensPOS2)
    ListaPosSostantivi1, ListaPosVerbi1,ListaPosAggettivi1,ListaPosAvverbi1 = EstraiSequenza(tokensPOS1) #funzione che preso in input tutti i tokens annotati mi restituisce liste contenenti ognuna una parte del discorso
    ListaPosSostantivi2, ListaPosVerbi2,ListaPosAggettivi2, ListaPosAvverbi2 = EstraiSequenza(tokensPOS2)
    #stampo i risultati ottenuti
    #1PRINT: NUMERO FRASI E DI TOKEN:
   #stampo il numero totale di frasi per entrambi i testi
    print()
    print("NUMERO TOTALE DI FRASI:")
    print()
    print("Il numero totale di frasi del testo", file1, "è",len(frasi1))  
    print()
    print("Il numero totale di frasi del testo", file2, "è",len(frasi2))
    #faccio il confronto fra i due testi per il numero di frasi
    print()
    print("CONFRONTO DEI DUE TESTI:")
    print()
    if len(frasi1) > len(frasi2):       #metto a confronto il numero totale di frasi dei due corpus per capire chi ne ha di più
        print("Il file", file1, "ha più frasi di", file2)
    elif len(frasi1) < len(frasi2):
        print("Il file", file2, "ha più frasi di", file1)
    else:
        print("I due file hanno lo stesso numero di frasi")
    print()
    #stampo i risultati ottenuti DEL NUMERO DI TOKENS
    print("NUMERO DI TOKENS:")
    print()
    print("Il file", file1, "ha", lunghezza1, "tokens")
    print("Il file", file2, "ha", lunghezza2, "tokens")
    #confronto le due lunghezze
    print()
    print("CONFRONTO FRA TOKENS:")  #metto a confronto il numero totale di tokens dei due corpus per capire chi ne ha di più
    print()
    if lunghezza1> lunghezza2:
        print("Il file", file1, "è piu lungo di", file2)
    elif lunghezza1<lunghezza2:
        print("Il file", file2, "è più lungo di", file1)
    else:
        print("I due file hanno la stessa lunghezza")
    print()
 
#2PRINT LUNGHEZZA MEDIA DELLE FRASI IN TERMINI DI TOKEN E DELLE PAROLE IN TERMINI DI CARATTERI
    print("LUNGHEZZA MEDIA DELLE FRASI IN TERMINI DI TOKENS:")
    print()
    lunghezzamediafrasi1 = lunghezza1*1.0/len(frasi1) #lunghezzamedia delle frasi è  data dal rapporto del numero di tokens per il numero di frasi
    lunghezzamediafrasi2 = lunghezza2*1.0/len(frasi2)
    
    print("La lunghezza media in termini di token delle frasi nel", file1, "è",lunghezzamediafrasi1)
    print()
    print("La lunghezza media in termini di token delle frasi nel", file2, "è",lunghezzamediafrasi2)
    print()
    print("CONFRONTO:")
    print()
    #confronto fra le due medie:
    if lunghezzamediafrasi1> lunghezzamediafrasi2:
        print("il file", file1, "ha frasi più lunghe di", file2)
    elif lunghezzamediafrasi1<lunghezzamediafrasi2:
        print("il file", file2, "ha frasi più lunghe di", file1)
    else:
        print("i due file hanno in media le frasi lunghe uguali")
    print()
#print lunghezza media in termini di caratteri
    print("LUNGHEZZA MEDIA IN TERMINI DI CARATTERI:")
    print()
    NumeroCaratteri1 = NumeroCaratterii(tokens1)
    NumeroCaratteri2 = NumeroCaratterii(tokens2)
    MediaCaratteri1 =NumeroCaratteri1*1.0/lunghezza1*1.0 #  MediaCaratteri è data 
    MediaCaratteri2 =NumeroCaratteri2*1.0/lunghezza2*1.0
    print("Il file", file1, "ha in media", MediaCaratteri1, "caratteri.")
    print()
    print("Il file", file2, "ha in media", MediaCaratteri2, "caratteri.")
    print()
    print("CONFRONTO:")
    print()
     #confronto fra le due medie:
    if  MediaCaratteri1 >  MediaCaratteri2:
        print("il file", file1, "ha frasi con parole più lunghe di", file2)
    elif MediaCaratteri1 < MediaCaratteri2:
        print("il file", file2, "ha frasi con parole più lunghe di", file1)
    else:
        print("i due file hanno in media le frasi con parole lunghe uguali")
#3PRINT GRANDEZZA DEL VOCABOLARIO E RICCHEZZA LESSICALE CALCOLATA ATTRAVERSO LA TYPE TOKEN RATIO
    print() 
    print("GRANDEZZA VOCABOLARIO")
    print()
    Vocabolario1 = set(tokens1) 
    Vocabolario2 = set(tokens2)
    print("La grandezza del Vocabolario del file", file1, "è di", len(Vocabolario1), "tokens")
    print()
    print("La grandezza del Vocabolario del file", file2, "è di", len(Vocabolario2), "tokens")
    print()
    print("TYPE TOKEN RATIO:")
    print()
    print("Type Token Ratio del file", file1, "è:",ttr1)
    print()
    print("Type Token Ratio del file", file2, "è:",ttr2)
    print()
    print("CONFRONTO:")
    print()
    if  ttr1 >  ttr2:
        print("Il file", file1, "ha la ttr maggiore di", file2)
    elif ttr1 < ttr2:
        print("Il file", file2, "ha la ttr maggiore di", file1)
    else:
        print("I due file hanno ttr uguale")
    

#4PRINT DISTRIBUZIONE DELLE CLASSI DI FREQUENZA |V1|, |V5|, |V10| ALL'AUMENTARE DEL CORPUS PER PORZIONI INCREMENTALI DI 500 TOKENS
    print()
    print("DISTRIBUZIONI DELLE CLASSI DI FREQUENZA |V1|, |V5|, |V10|, ALL'AUMENTARE DEL CORPUS PER PORZIONI INCREMENTALI DI 500 TOKEN")
    print()
    print("ANALISI TESTO: Biden.txt") #chiamo la funzione CalcolaFrequenza con i vari incrementi
    CalcolaFrequenza(tokens1)
    
    print("ANALISI TESTO: Trump.txt")
    CalcolaFrequenza(tokens2)
    
    print()
#5PRINT  MEDIA SOSTANTIVI E VERBI PER FRASE
    print("MEDIA DEI SOSTANTIVI PER FRASE:") 
    print()
    MediaSostantivi1 = (len(ListaPosSostantivi1)*1.0)/(len(frasi1)*1.0) #la media dei sostantivi è data dalla lunghezza della listaPosSostantivi/ il numero di frasi
    print("In media nel testo ", file1, "ci sono", MediaSostantivi1, "Sostantivi")
    print()
    MediaSostantivi2 = (len(ListaPosSostantivi2)*1.0)/(len(frasi2)*1.0)
    print("In media nel testo ", file2, "ci sono", MediaSostantivi2, "Sostantivi")
    print()
    print("MEDIA DEI VERBI PER FRASE:")
    print()
    MediaVerbi1 = (len(ListaPosVerbi1)*1.0)/(len(frasi1)*1.0)#la media dei verbi è data dalla lunghezza della listaPosSostantivi/ il numero di frasi
    print("In media nel testo", file1, "ci sono", MediaVerbi1, "Verbi")
    print()
    MediaVerbi2 = (len(ListaPosVerbi2)*1.0)/(len(frasi2)*1.0)
    print()
    print("In media nel testo", file2, "ci sono", MediaVerbi2, "Verbi")
    print()
    print("CONFRONTO:")
    print()
    if MediaVerbi1 >  MediaVerbi2: #confronto le due medie per vedere quale dei tuoi testi ha la media più alta di verbi per frase
        print("Il file", file1, "ha in media più verbi di per frase di ", file2)
    elif MediaVerbi1 < MediaVerbi2:
        print("Il file", file2, "ha in media più verbi di per frase di ", file1)
    else:
        print("I due file hanno in media lo stesso numero di verbi")
    print()
    if MediaSostantivi1 >  MediaSostantivi2:  #confronto le due medie per vedere quale dei tuoi testi ha la media più alta di sostantivi per frase
        print("Il file", file1, "ha in media più sostantivi di per frase di", file2)
    elif MediaSostantivi1 < MediaSostantivi2:
        print("Il file", file2, "ha in media più sostantivi di " ,file1)
    else:
        print("I due file hanno in media lo stesso numero di sostantivi")
    print()

#6PRINT DENSITA' LESSICALE
    print("DENSITA' LESSICALE:")
    print()
    sommaPartiDiscorso1= len(ListaPosSostantivi1) +len(ListaPosVerbi1) + len(ListaPosAggettivi1) + len(ListaPosAvverbi1) #sommo il numero delle varie parti del discorso cioè le liste contenenti: nomi, aggettivi, verbi e avverbi
    sommaPartiDiscorso2= len(ListaPosSostantivi2) + len(ListaPosVerbi2) + len(ListaPosAggettivi2) + len(ListaPosAvverbi2)
    Totale1 = (sommaPartiDiscorso1*1.0)/len(NoPunteggiatura1)*1.0 #vado a fare il rapporto tra la somma delle parti del discorso prima eseguita e la lista che contiene il numero di tutte le parti del discorso esclusi i pos(,.)
    Totale2 = (sommaPartiDiscorso1*1.0)/len(NoPunteggiatura2)*1.0
    
    print("La densità Lessicale di ", file1, "è di", Totale1)
    print()
    print("La densità Lessicale di ", file2, "è di", Totale2)
    print()
    print("CONFRONTO:")
    print()
    if Totale1 >  Totale2: #confronto le due medie per vedere quale dei tuoi testi ha la desnità lessicale più alta
        print("Il file", file1, "ha una densità lessicale più alta di ", file2)
    elif Totale1 < Totale2:
        print("Il file", file2, "ha una densità lessicale più alta di ", file1)
    else:
        print("I due file hanno la stessa densità lessicale")
    print()
   

main(sys.argv[1], sys.argv[2])
