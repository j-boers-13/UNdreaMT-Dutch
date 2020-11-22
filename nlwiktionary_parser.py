import xml.dom.pulldom as pulldom
import xpath # from http://code.google.com/p/py-dom-xpath/
import wikitextparser as wtp
import re
import pickle
from collections import defaultdict

x=0
events = pulldom.parse('nlwiki-latest-pages-articles.xml')

sep_list = ["is een", "is de", "was een", "was de", "waren een", "waren de", "zijn een", "zijn de"]
def_dict = defaultdict(set)
bad_list = [":", "/"]
def custom_split(sepr_list, str_to_split):
    # create regular expression dynamically
    regular_exp = '|'.join(map(re.escape, sepr_list))
    return re.split(regular_exp, str_to_split)

for event, node in events:
    if event == 'START_ELEMENT' and node.tagName=='page':
        x += 1

        events.expandNode(node) # node now contains a dom fragment
        title = xpath.findvalue('title', node)
        title = re.sub("[\(|].*?[\)]", "", title).strip().lower()

        if len(title.split()) > 1 or any(bad in title for bad in bad_list):
            continue

        revision = xpath.findvalue('revision', node)
        text = xpath.findvalues('revision/text', node)
        wiki_parsed = wtp.parse(text[0]).sections[0]
        wiki_parsed_str = str(wiki_parsed)
        for table in wiki_parsed.tables:
            wiki_parsed_str = wiki_parsed_str.replace(str(table), "")
        for tmpl in wiki_parsed.templates:
            wiki_parsed_str = wiki_parsed_str.replace(str(tmpl), "")
        for ref in wiki_parsed.get_tags():
            wiki_parsed_str = wiki_parsed_str.replace(str(ref), '')
        for link in wiki_parsed.wikilinks:
            wiki_parsed_str = wiki_parsed_str.replace(str(link), link.title)
        
        wiki_parts = wiki_parsed_str.strip().split('\n')
        for part in wiki_parts:
            if any(sep in part for sep in sep_list):
                definition = custom_split(sep_list, part)[1]
                definition = re.sub("[\(|].*?[\)]", "", definition)
                definition = definition.split(".")[0].lower()
                definition = re.sub("['''|\";,\.\:\(\)\±\=\]\[]", "", definition)
                definition = re.sub("[<.*?>].*?[<.*?>]", "", definition)
                def_dict[title].update(definition.split())

        if x % 1000 == 0:
            print("%d: %s" % (x, title))
        
with open('wiktionary_definitions.p', 'wb') as f:
    pickle.dump(def_dict, f, protocol=pickle.HIGHEST_PROTOCOL)