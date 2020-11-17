from bs4 import BeautifulSoup
import requests  
from requests.exceptions import HTTPError
from requests.exceptions import ConnectionError
import urllib.request
import re
import json
from collections import defaultdict
import urllib.parse
import http.client as http
import aiohttp
import asyncio
import os
from aiohttp import ClientSession
os.environ['PYTHONASYNCIODEBUG'] = '1'
import logging
import pickle
logging.basicConfig(level=logging.DEBUG)


JSON_PATH="dict2vec/data/synonyms.json"
WORDS_PATH="dict2vec/data/wordlist.txt"
URL = "https://www.mijnwoordenboek.nl/synoniemen/"
DEF_DICT = defaultdict(set)
SYN_DICT = defaultdict(set)
FAILED_WORDS = []

def extract_definitions(html_str):
    soup = BeautifulSoup(html_str, 'html.parser')
    def_elems = soup.findAll('a')                          # get the linked element
    defs = [el.get_text().lower() for el in def_elems]     # get only the text from the element
    defs = [re.sub("[\(].*?[\)]", "", x) for x in defs]    # Remove content including parentheses
    defs = [word for definition in defs for word in definition.split()]   # split strings into words
    return defs

def extract_words(html_str):
    soup = BeautifulSoup(html_str, 'html.parser')
    word_elems = soup.findAll('a')
    words = [el.get_text().lower() for el in word_elems]
    return words

async def extract_synonyms_definitions(response, word):
    soup = BeautifulSoup(response, 'html.parser')
    top_split = str(soup).split("Synoniemen van {}".format(word),1)
    if type(top_split) == list and len(top_split) > 1:
        top_removed = top_split[1]
        bottom_split = top_removed.split("Puzzelomschrijvingen van {}:".format(word),1)

        if type(bottom_split) == list and len(bottom_split) > 1:
            synonym_split = bottom_split[0]
            puzzel_split = bottom_split[1].split("Cryptische omschrijvingen van {}:".format(word),1)[0]
            synonyms = extract_words(synonym_split)
            definitions = extract_definitions(puzzel_split)

            return synonyms, definitions

    return [],[]

async def download_word(word, session):
    url = URL + urllib.parse.quote(word)

    try:
        response = await session.request(method='GET', url=url)
        response.raise_for_status()
        response_text = await response.text()
        return response_text
    except HTTPError:
        print("HTTPError: ", word)
    except UnicodeDecodeError:
        print("UnicodeDecodeError: ", word)
    except Exception as e:
        print("Exception in download_word:", e, word)
        FAILED_WORDS << word
        return None



async def run_program(line, session, sem):
    """Wrapper for running program in an asynchronous manner"""
    try:
        word = line.strip().lower()
        async with sem:
            response = await download_word(word, session)
        if response:
            synonyms, definitions = await extract_synonyms_definitions(response, word)
            if synonyms:
                SYN_DICT[word].update(synonyms)
            if definitions:
                DEF_DICT[word].update(definitions)
            print(synonyms, definitions)
    except Exception as err:
        FAILED_WORDS << word
        print(f"Exception in run_program occured: {err}")

        pass

async def main():
    conn = aiohttp.TCPConnector(limit=0)
    words_file = open(WORDS_PATH, 'r') 
    lines = words_file.readlines()
    headers={'Connection': 'keep-alive', 'Transfer-Encoding': 'chunked'}
    sem = asyncio.Semaphore(50)

    async with ClientSession(headers=headers, connector=conn) as session:
        await asyncio.gather(*[run_program(line, session, sem) for line in lines])

    with open('failed_words.p', 'wb') as ffw:
        pickle.dump(FAILED_WORDS, ffw, protocol=pickle.HIGHEST_PROTOCOL)

    with open('synonyms.p', 'wb') as fs:
        pickle.dump(SYN_DICT, fs, protocol=pickle.HIGHEST_PROTOCOL)

    with open('definitions.p', 'wb') as fd:
        pickle.dump(DEF_DICT, fd, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    asyncio.run(main())
