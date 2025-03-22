import sys
import pandas as pd
from rdflib import Graph, URIRef, Namespace, Literal

def get_graph():
    data = pd.read_csv("./datasets/doncic_cut_ref.csv")
    g = Graph()
    namespace = Namespace("http://icon-uniba.uni/exam/")

    player = URIRef(namespace + "players/Luka_Doncic")
    player_team = URIRef(namespace + "teams/DLS")

    for i, row in data.iterrows():
        game = URIRef(namespace + "games/" + str(i))
        opp =  URIRef(namespace + "teams/" + row.OPP)
        stat = URIRef(namespace + "stats/" + str(i) + "/1")

        # giocate da
        g.add((game, namespace.gioca_in_casa, player_team if row.HOME else opp))
        g.add((game, namespace.gioca_in_trasferta, opp if row.HOME else player_team))

        # vittoria
        g.add((game, namespace.vinta_da, player_team if row.WIN else opp))

        # data
        g.add((game, namespace.giocata_il, Literal(row.Date)))

        # info prestazione
        g.add((game, namespace.ha, stat))
        g.add((stat, namespace.giocatore, player))
        g.add((stat, namespace.del_team, player_team))

        # rimbalzi
        g.add((stat, namespace.defensive_rebounds, Literal(row.DRB)))
        g.add((stat, namespace.offensive_rebounds, Literal(row.ORB)))
        g.add((stat, namespace.total_rebounds, Literal(row.TRB)))

        # punti su azione
        g.add((stat, namespace.field_goals, Literal(row.FG)))
        g.add((stat, namespace.field_goals_avg, Literal(row.FGA)))
        g.add((stat, namespace.field_goals_perc, Literal(row["FG%"])))

        # 3 punti
        g.add((stat, namespace.tree_points, Literal(row["3P"])))
        g.add((stat, namespace.tree_points_avg, Literal(row["3PA"])))
        g.add((stat, namespace.tree_points_perc, Literal(row["3P%"])))

        # tiri liberi
        g.add((stat, namespace.free_throws, Literal(row.FT)))
        g.add((stat, namespace.free_throws_avg, Literal(row.FTA)))
        g.add((stat, namespace.free_throws_perc, Literal(row["FT%"])))

        # resto delle statistiche
        g.add((stat, namespace.turnover, Literal(row.TOV)))
        g.add((stat, namespace.assists, Literal(row.AST)))
        g.add((stat, namespace.steals, Literal(row.STL)))
        g.add((stat, namespace.bloks, Literal(row.BLK)))
        g.add((stat, namespace.personal_fouls, Literal(row.PF)))

        g.add((stat, namespace.points, Literal(row.PTS)))

        g.add((stat, namespace.GmSc, Literal(row.GmSc)))
        g.add((stat, namespace.difference, Literal(row["+/-"])))

        if i == 0:
            print("ESEMPIO DEL GRAFO:\n", g.serialize(format="turtle"))

    return g

def print_results(results, formatters):
    if len(results) == 0:
        print("Nessun risultato trovato")
    else:
        print("------------------------>")
        for row in results:
            for name, key in formatters:
                print(f"{name}: {row[key]}")
            print("------------------------>")


def get_previous_matches(graph, team_a, team_b):
    return graph.query(f"""
        PREFIX ns: <http://icon-uniba.uni/exam/>

        SELECT ?game ?home_team ?away_team
        WHERE {{
            ?game ns:gioca_in_casa ?home_team .
            ?game ns:gioca_in_trasferta ?away_team .
            FILTER((?home_team = <http://icon-uniba.uni/exam/teams/{team_b}> && ?away_team = <http://icon-uniba.uni/exam/teams/{team_a}>) || 
                   (?home_team = <http://icon-uniba.uni/exam/teams/{team_a}> && ?away_team = <http://icon-uniba.uni/exam/teams/{team_b}>))
        }}
    """)

def get_player_stats(graph, game, player):
    return graph.query(f"""
        PREFIX ns: <http://icon-uniba.uni/exam/>
        
        SELECT ?game ?player ?points ?assists ?rebounds
        WHERE {{
            ?game ns:ha ?stat .
            ?stat ns:giocatore ?player .
            ?stat ns:points ?points .
            ?stat ns:assists ?assists .
            ?stat ns:total_rebounds ?rebounds .
            FILTER (?player = <http://icon-uniba.uni/exam/players/{player}> && ?game = <http://icon-uniba.uni/exam/games/{game}>)
        }}
    """)

def get_won_games(graph, team):
    return graph.query(f"""
        PREFIX ns: <http://icon-uniba.uni/exam/>

        SELECT ?game ?date ?home_team ?away_team
        WHERE {{
            ?game ns:vinta_da <http://icon-uniba.uni/exam/teams/{team}> .
            ?game ns:gioca_in_casa ?home_team .
            ?game ns:gioca_in_trasferta ?away_team .
            ?game ns:giocata_il ?date .
        }}
    """)

def get_games_by_date(graph, date):
    return graph.query(f"""
        PREFIX ns: <http://icon-uniba.uni/exam/>

        SELECT ?game ?date ?home_team ?away_team
        WHERE {{
            ?game ns:gioca_in_casa ?home_team .
            ?game ns:gioca_in_trasferta ?away_team .
            ?game ns:giocata_il ?date .
            FILTER(?date = "{date}")
        }}
    """)

def get_player_triple_doubles(graph, player):
    return graph.query(f"""
        PREFIX ns: <http://icon-uniba.uni/exam/>
        
        SELECT ?game ?date ?home_team ?away_team ?player ?points ?assists ?rebounds
        WHERE {{
            ?game ns:giocata_il ?date .
            ?game ns:ha ?stat .
            ?stat ns:giocatore ?player .
            ?game ns:gioca_in_casa ?home_team .
            ?game ns:gioca_in_trasferta ?away_team .
            ?stat ns:points ?points .
            ?stat ns:assists ?assists .
            ?stat ns:total_rebounds ?rebounds .
            FILTER (
                ?player = <http://icon-uniba.uni/exam/players/{player}> && 
                ?points >= 10 && 
                ?assists >= 10 && 
                ?rebounds >= 10
            )
        }}
    """)

def get_avg_pts_between_dates(graph, player, from_d, to_d):
    return graph.query(f"""
        PREFIX ns: <http://icon-uniba.uni/exam/>
        
        SELECT (AVG(?points) AS ?avg_points)
        WHERE {{
            ?game ns:giocata_il ?date .
            ?game ns:ha ?stat .
            ?stat ns:giocatore <http://icon-uniba.uni/exam/players/{player}> .
            ?stat ns:points ?points .
            FILTER (
                ?date >= "{from_d}" && ?date <= "{to_d}"
            )
        }}
    """)

try:
    graph = get_graph()

    graph.serialize("kg.rdf", format="turtle")

    command = int(sys.argv[1])

    match command:
        case 0:
            q_results = get_previous_matches(graph, sys.argv[2], sys.argv[3])
            q_formatters = [
                ("Partita", "game"),
                ("Team di casa", "home_team"),
                ("Team in trasferta", "away_team"),
            ]
        case 1:
            q_results = get_player_stats(graph, int(sys.argv[2]), sys.argv[3])
            q_formatters = [
                ("Partita", "game"),
                ("Giocatore", "player"),
                ("Punti", "points"),
                ("Assist", "assists"),
                ("Rimbalzi", "rebounds"),
            ]
        case 2:
            q_results = get_won_games(graph, sys.argv[2])
            q_formatters = [
                ("Partita", "game"),
                ("Data", "date"),
                ("Team di casa", "home_team"),
                ("Team in trasferta", "away_team"),
            ]
        case 3:
            q_results = get_games_by_date(graph, sys.argv[2])
            q_formatters = [
                ("Partita", "game"),
                ("Data", "date"),
                ("Team di casa", "home_team"),
                ("Team in trasferta", "away_team"),
            ]
        case 4:
            q_results = get_player_triple_doubles(graph, sys.argv[2])
            q_formatters = [
                ("Partita", "game"),
                ("Data", "date"),
                ("Team di casa", "home_team"),
                ("Team in trasferta", "away_team"),
                ("Giocatore", "player"),
                ("Punti", "points"),
                ("Assist", "assists"),
                ("Rimbalzi", "rebounds"),
            ]
        case 5:
            q_results = get_avg_pts_between_dates(graph, sys.argv[2], sys.argv[3], sys.argv[4])
            q_formatters = [
                ("Media", "avg_points"),
            ]

    print("RISULTATO QUERY: \n")

    print_results(q_results, q_formatters)
except:
    print("Parametri errati o mancanti")
