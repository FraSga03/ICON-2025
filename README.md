# Progetto 

Questo progetto è realizzato per l'esame di ICON, tenuto dal Professore Nicola Fanizzi presso l'Università degli Studi di Bari Aldo Moro.

Tutta la documentazione è disponibile nel file documentation.md o in formato PDF documentation.pdf

# Esecuzione script

Guida all’esecuzione degli script del progetto:

## Installazione librerie

Una volta scaricato il progetto, installa le librerie con:
```commandline
pip install -r requirements.txt
```

## Refactor Dati

Effettua il preprocessing sul dataset originario (i file sono già presenti).

```commandline
python refactorCSV.py
```

Genera nella cartella dataset i .csv necessari ai modelli.

## Apprendimento Supervisionato

Allena e testa i modelli supervisionati.

```commandline
python supervised_model.py [0-4]
# ad esempio
python supervised_model.py 0
```

- 0 - albero di decisione
- 1 - random forest
- 2 - SVC
- 3 - Logistic Regression
- 4 - KNN

Per ogni modello ci sono 3 varianti:

- Modello allenato con KF e shuffle
- Modello allenato con KF
- Modello allenato senza KF

## Reti bayesiane

Allena e genera esempi casuali per delle reti bayesiane

```commandline
python bayes_network.py [0-1]
# ad esempio
python bayes_network.py 0
```

- 0 - rete bayesiana generata automaticamente
- 1 - rete bayesiana creata da me

## Rete neurale

Allena e testa una rete neurale.

```commandline
python neural_network.py
```

## Knowledge Graph

Crea ed effettua query su un KG

- *scontri precedenti fra due team*

    ```commandline
    python knowledge_graph.py 0 [team_a] [team_b]
    
    #ad esempio
    python knowledge_graph.py 0 DLS PHO
    ```

- *statistiche di un giocatore in una partita*

    ```commandline
    python knowledge_graph.py 1 [game] [player]
    
    #ad esempio
    python knowledge_graph.py 1 1 Luka_Doncic
    ```

  L’unico giocatore presente è Luka_Doncic, altri giocatori non produrranno alcun risultato.

- *partite vinte da un team*

    ```commandline
    python knowledge_graph.py 2 [team]
    
    #ad esempio
    python knowledge_graph.py 2 DLS
    ```

- *partite giocate in una data*

    ```commandline
    python knowledge_graph.py 3 [date]
    
    #ad esempio
    python knowledge_graph.py 3 2019-10-25
    ```

  Il formato della data è sempre YYYY-MM-DD.

- *triple doppie di un giocatore*

    ```commandline
    python knowledge_graph.py 4 [player]
    
    #ad esempio
    python knowledge_graph.py 4 Luka_Doncic
    ```

  L’unico giocatore presente è Luka_Doncic, altri giocatori non produrranno alcun risultato.

- *media punti in un periodo di un giocatore*

    ```commandline
    python knowledge_graph.py 5 [player] [from] [to]
    
    #ad esempio
    python knowledge_graph.py 5 Luka_Doncic 2023-01-01 2023-02-01
    ```

  L’unico giocatore presente è Luka_Doncic, altri giocatori non produrranno alcun risultato.

  Il formato della data è sempre YYYY-MM-DD.