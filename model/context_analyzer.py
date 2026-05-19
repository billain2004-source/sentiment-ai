import subprocess
import sys
from typing import List, Dict
import spacy


def _load_spacy():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
        return spacy.load('en_core_web_sm')


class ContextAnalyzer:
    def __init__(self):
        self.nlp = _load_spacy()

        self.pronoun_map = {
            'she':  {'type': 'Person · Female',        'description': 'Third-person singular feminine pronoun', 'css_class': 'pronoun-female'},
            'her':  {'type': 'Person · Female',        'description': 'Feminine objective / possessive pronoun', 'css_class': 'pronoun-female'},
            'hers': {'type': 'Person · Female',        'description': 'Feminine possessive pronoun',            'css_class': 'pronoun-female'},
            'he':   {'type': 'Person · Male',          'description': 'Third-person singular masculine pronoun', 'css_class': 'pronoun-male'},
            'him':  {'type': 'Person · Male',          'description': 'Masculine objective pronoun',            'css_class': 'pronoun-male'},
            'his':  {'type': 'Person · Male',          'description': 'Masculine possessive pronoun',           'css_class': 'pronoun-male'},
            'they': {'type': 'Person / Group',         'description': 'Gender-neutral or plural pronoun',       'css_class': 'pronoun-neutral'},
            'them': {'type': 'Person / Group',         'description': 'Gender-neutral plural objective',        'css_class': 'pronoun-neutral'},
            'we':   {'type': 'Group · People',         'description': 'First-person plural pronoun',            'css_class': 'pronoun-neutral'},
            'i':    {'type': 'Person · Self',          'description': 'First-person singular pronoun',          'css_class': 'pronoun-neutral'},
            'it':   {'type': 'Object / Entity',        'description': 'Third-person neuter pronoun',            'css_class': 'pronoun-neutral'},
        }

        self.domain_dict = {
            # Sports
            'cricket':    {'type': 'Sport',                    'description': 'Bat-and-ball sport popular in South Asia & UK'},
            'bat':        {'type': 'Sport / Animal',           'description': 'Cricket bat  OR  flying nocturnal mammal'},
            'pitch':      {'type': 'Sport / Music / Place',    'description': 'Cricket pitch, musical pitch, or sales pitch'},
            'football':   {'type': 'Sport',                    'description': 'Team ball sport (soccer or American football)'},
            'basketball': {'type': 'Sport',                    'description': 'Hoop-shooting team sport'},
            'tennis':     {'type': 'Sport',                    'description': 'Racket-based court sport'},
            'hockey':     {'type': 'Sport',                    'description': 'Ice or field hockey'},
            'golf':       {'type': 'Sport',                    'description': 'Club-and-ball course sport'},
            'swimming':   {'type': 'Sport / Activity',         'description': 'Aquatic sport or leisure activity'},
            'running':    {'type': 'Sport / Activity',         'description': 'Athletic event or general exercise'},
            'bowling':    {'type': 'Sport / Cricket',          'description': 'Tenpin bowling  OR  cricket delivery'},
            'match':      {'type': 'Sport / General',          'description': 'Sporting contest  OR  fire match'},
            'goal':       {'type': 'Sport / General',          'description': 'Scoring point in sport  OR  personal objective'},
            'court':      {'type': 'Sport / Legal',            'description': 'Sports court  OR  judicial court'},
            'race':       {'type': 'Sport / Society',          'description': 'Athletic race  OR  ethnic/social race'},
            'league':     {'type': 'Sport / Organization',     'description': 'Sports league  OR  alliance / coalition'},
            'coach':      {'type': 'Sport / Transport',        'description': 'Sports coach  OR  a bus/carriage'},
            'champion':   {'type': 'Sport / General',          'description': 'Sports winner  OR  advocate for a cause'},
            'field':      {'type': 'Sport / Agriculture / Science', 'description': 'Playing field, farm field, or academic field'},
            'club':       {'type': 'Sport / Social',           'description': 'Sports club  OR  nightclub / social group'},
            'net':        {'type': 'Sport / Technology',       'description': 'Sports net  OR  the internet / network'},
            'base':       {'type': 'Sport / Military / Science', 'description': 'Baseball base, military base, or chemical base'},
            # Technology
            'python':     {'type': 'Technology / Animal',      'description': 'Programming language  OR  large constrictor snake'},
            'java':       {'type': 'Technology / Place',       'description': 'Programming language  OR  island in Indonesia'},
            'apple':      {'type': 'Company / Fruit',          'description': 'Apple Inc. (tech company)  OR  the fruit'},
            'amazon':     {'type': 'Company / Geography',      'description': 'E-commerce giant  OR  South American river'},
            'cloud':      {'type': 'Meteorology / Technology', 'description': 'Weather cloud  OR  cloud computing'},
            'bug':        {'type': 'Biology / Technology',     'description': 'Insect  OR  software defect'},
            'virus':      {'type': 'Biology / Technology',     'description': 'Biological pathogen  OR  malware'},
            'mouse':      {'type': 'Animal / Technology',      'description': 'Rodent  OR  computer pointing device'},
            'chip':       {'type': 'Technology / Food',        'description': 'Semiconductor chip  OR  potato chip / crisp'},
            'window':     {'type': 'Architecture / Technology','description': 'Glass building window  OR  Windows operating system'},
            'drive':      {'type': 'Technology / Transport',   'description': 'Storage drive  OR  driving a vehicle'},
            'key':        {'type': 'Security / Technology / Music', 'description': 'Physical key, keyboard key, or musical key'},
            'stream':     {'type': 'Geography / Technology / Media', 'description': 'Water stream  OR  media streaming'},
            'shell':      {'type': 'Biology / Technology',     'description': 'Animal shell  OR  command-line shell'},
            'memory':     {'type': 'Biology / Technology',     'description': 'Human memory  OR  computer RAM / storage'},
            'port':       {'type': 'Transport / Technology',   'description': 'Seaport  OR  network port  OR  data port'},
            # Science & Nature
            'mercury':    {'type': 'Science / Mythology',      'description': 'Element (Hg), planet Mercury, or Roman god'},
            'mars':       {'type': 'Science / Mythology',      'description': 'Planet, Roman god of war, or the chocolate bar'},
            'spring':     {'type': 'Season / Technology',      'description': 'Season, water spring, coil, or Spring Framework'},
            'bank':       {'type': 'Finance / Geography',      'description': 'Financial institution  OR  river bank'},
            'star':       {'type': 'Astronomy / Entertainment','description': 'Celestial body  OR  celebrity / famous person'},
            'cell':       {'type': 'Biology / Technology',     'description': 'Biological cell  OR  mobile phone / prison cell'},
            'root':       {'type': 'Biology / Technology',     'description': 'Plant root  OR  root user / fundamental origin'},
            # General
            'iron':       {'type': 'Material / Sport / Appliance', 'description': 'Metal element, golf iron, or clothes iron'},
            'board':      {'type': 'General / Technology',     'description': 'Wooden board, circuit board, or board of directors'},
            'tank':       {'type': 'Military / Container',     'description': 'Military tank  OR  storage tank / aquarium'},
            'cap':        {'type': 'Clothing / Finance',       'description': 'Hat cap  OR  financial cap / limit'},
            'pool':       {'type': 'Sport / General',          'description': 'Swimming pool, billiards, or resource pool'},
            'break':      {'type': 'Sport / General',          'description': 'Tennis break, snooker break, or rest period'},
        }

        self.ner_css = {
            'PERSON': 'PERSON', 'ORG': 'ORG', 'GPE': 'GPE', 'LOC': 'GPE',
            'PRODUCT': 'domain', 'EVENT': 'domain', 'WORK_OF_ART': 'domain',
            'LAW': 'ORG', 'LANGUAGE': 'domain', 'DATE': 'domain',
            'TIME': 'domain', 'MONEY': 'domain', 'NORP': 'GPE',
            'FAC': 'GPE', 'CARDINAL': 'domain', 'PERCENT': 'domain',
            'QUANTITY': 'domain',
        }

    def analyze(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        contexts = []
        seen = set()

        for ent in doc.ents:
            key = ent.text.lower()
            if key in seen:
                continue
            seen.add(key)
            css = self.ner_css.get(ent.label_, 'default')
            description = spacy.explain(ent.label_) or ent.label_
            contexts.append({
                'word': ent.text,
                'type': ent.label_.replace('_', ' ').title(),
                'description': description.capitalize(),
                'css_class': css,
            })

        for token in doc:
            key = token.text.lower()
            if token.pos_ == 'PRON' and key in self.pronoun_map and key not in seen:
                seen.add(key)
                info = self.pronoun_map[key]
                contexts.append({
                    'word': token.text,
                    'type': info['type'],
                    'description': info['description'],
                    'css_class': info['css_class'],
                })

        for token in doc:
            key = token.text.lower()
            if key in self.domain_dict and key not in seen:
                seen.add(key)
                info = self.domain_dict[key]
                contexts.append({
                    'word': token.text,
                    'type': info['type'],
                    'description': info['description'],
                    'css_class': 'domain',
                })

        return contexts
