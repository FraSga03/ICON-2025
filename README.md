# Esecuzione script

Guida all’esecuzione degli script del progetto:

## Refactor Dati

Effettua il preprocessing sul dataset originario.

```python
python refactorCSV.py
```

Genera nella cartella dataset i .csv necessari ai modelli, i file sono già presenti

## Apprendimento Supervisionato

Allena e testa i modelli supervisionati.

```python
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

```python
python bayes_network.py [0-1]
# ad esempio
python bayes_network.py 0
```

- 0 - rete bayesiana generata automaticamente
- 1 - rete bayesiana creata da me

## Rete neurale

Allena e testa una rete neurale.

```python
python neural_network.py
```

## Knowledge Graph

Crea ed effettua query su un KG

- *scontri precedenti fra due team*

    ```python
    python knowledge_graph.py 0 [team_a] [team_b]
    
    #ad esempio
    python knowledge_graph.py 0 DLS PHO
    ```

- *statistiche di un giocatore in una partita*

    ```python
    python knowledge_graph.py 1 [game] [player]
    
    #ad esempio
    python knowledge_graph.py 1 1 Luka_Doncic
    ```

  L’unico giocatore presente è Luka_Doncic, altri giocatori non produrranno alcun risultato.

- *partite vinte da un team*

    ```python
    python knowledge_graph.py 2 [team]
    
    #ad esempio
    python knowledge_graph.py 2 DLS
    ```

- *partite giocate in una data*

    ```python
    python knowledge_graph.py 3 [date]
    
    #ad esempio
    python knowledge_graph.py 3 2023-01-01
    ```

  Il formato della data è sempre YYYY-MM-DD.

- *triple doppie di un giocatore*

    ```python
    python knowledge_graph.py 4 [player]
    
    #ad esempio
    python knowledge_graph.py 4 Luka_Doncic
    ```

  L’unico giocatore presente è Luka_Doncic, altri giocatori non produrranno alcun risultato.

- *media punti in un periodo di un giocatore*

    ```python
    python knowledge_graph.py 5 [player] [from] [to]
    
    #ad esempio
    python knowledge_graph.py 5 2023-01-01 2023-02-01
    ```

  L’unico giocatore presente è Luka_Doncic, altri giocatori non produrranno alcun risultato.

  Il formato della data è sempre YYYY-MM-DD.