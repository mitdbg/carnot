from typing import Dict

# Strategy for Query 10: 1912 films set in England (No Truncation)
def execute_query(retrieve, k1, k2):
    films_1912_dict = retrieve("1912 films", k1)
    films_set_in_england_dict = retrieve("English films", k2)
    
    films_1912_titles = set(films_1912_dict.keys())
    films_set_in_england_titles = set(films_set_in_england_dict.keys())

    intersecting_titles = films_1912_titles & films_set_in_england_titles
    
    final_docs_dict = {
        title: films_1912_dict[title] 
        for title in intersecting_titles
    }
    
    return final_docs_dict, films_1912_dict, films_set_in_england_dict