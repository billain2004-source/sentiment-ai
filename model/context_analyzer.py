import subprocess
import sys
from typing import List, Dict
import spacy
import pycountry
import gender_guesser.detector as gender_detector
from names_dataset import NameDataset


def _load_spacy():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
        return spacy.load('en_core_web_sm')


class ContextAnalyzer:
    def __init__(self):
        self.nlp = _load_spacy()
        self._nd = NameDataset()                                    # 160k+ names, 101 countries
        self._gender_detector = gender_detector.Detector(case_sensitive=False)  # fallback

        # Build a set of known country/territory names to avoid misclassifying them as persons
        self._country_names = {c.name.lower() for c in pycountry.countries}
        self._country_names.update({c.common_name.lower() for c in pycountry.countries if hasattr(c, 'common_name')})

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

    def _name_gender(self, name: str, min_confidence: float = 0.55):
        # Primary: names-dataset (160k+ names, 101 countries, with probability scores)
        try:
            r = self._nd.search(name.strip())
            fn = r.get('first_name') if r else None
            if fn:
                g = fn.get('gender', {})
                male = g.get('Male', 0)
                female = g.get('Female', 0)
                total = male + female
                if total > 0:
                    if male / total >= min_confidence:
                        return {'type': 'Person · Male', 'description': f'"{name}" is a male first name', 'css_class': 'pronoun-male'}
                    if female / total >= min_confidence:
                        return {'type': 'Person · Female', 'description': f'"{name}" is a female first name', 'css_class': 'pronoun-female'}
        except Exception:
            pass

        # Fallback: gender-guesser (48k names, 79 countries)
        result = self._gender_detector.get_gender(name.strip())
        if result in ('male', 'mostly_male'):
            return {'type': 'Person · Male', 'description': f'"{name}" is a male first name', 'css_class': 'pronoun-male'}
        if result in ('female', 'mostly_female'):
            return {'type': 'Person · Female', 'description': f'"{name}" is a female first name', 'css_class': 'pronoun-female'}
        return None

    def analyze(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        contexts = []
        seen = set()

        for ent in doc.ents:
            key = ent.text.lower().strip()
            if key in seen:
                continue
            seen.add(key)

            # For PERSON entities, try to detect gender from name
            if ent.label_ == 'PERSON':
                first_word = ent.text.split()[0]
                gender = self._name_gender(first_word)
                if gender:
                    contexts.append({
                        'word': ent.text,
                        'type': gender['type'],
                        'description': gender['description'],
                        'css_class': gender['css_class'],
                    })
                else:
                    contexts.append({
                        'word': ent.text,
                        'type': 'Person',
                        'description': f'"{ent.text}" is a named individual',
                        'css_class': 'PERSON',
                    })
                continue

            # If spaCy misclassified a name as GPE/NORP/LOC, correct it
            # Skip if the full entity text is a known country name
            if ent.label_ in ('GPE', 'NORP', 'LOC') and ent.text.lower() not in self._country_names:
                first_word = ent.text.split()[0]
                if first_word.lower() not in self._country_names:
                    # Use strict 85% confidence threshold to avoid city names being treated as persons
                    gender = self._name_gender(first_word, min_confidence=0.85)
                    if gender:
                        contexts.append({
                            'word': ent.text,
                            'type': gender['type'],
                            'description': gender['description'],
                            'css_class': gender['css_class'],
                        })
                        continue

            # If spaCy tagged as ORG/PRODUCT, check if it's actually a name
            if ent.label_ in ('ORG', 'PRODUCT', 'WORK_OF_ART'):
                first_word = ent.text.split()[0]
                gender = self._name_gender(first_word)
                if gender:
                    contexts.append({
                        'word': ent.text,
                        'type': gender['type'],
                        'description': gender['description'],
                        'css_class': gender['css_class'],
                    })
                    continue
                # Single title-cased word tagged as ORG is likely a misidentified person name
                words = ent.text.split()
                if (len(words) <= 2
                        and all(w[0].isupper() for w in words if w)
                        and ent.text.lower() not in self.domain_dict
                        and not any(w.lower() in {'inc', 'ltd', 'corp', 'llc', 'co', 'group', 'company', 'industries'} for w in words)):
                    contexts.append({
                        'word': ent.text,
                        'type': 'Person',
                        'description': f'"{ent.text}" appears to be a personal name',
                        'css_class': 'PERSON',
                    })
                    continue

            css = self.ner_css.get(ent.label_, 'default')
            description = spacy.explain(ent.label_) or ent.label_
            contexts.append({
                'word': ent.text,
                'type': ent.label_.replace('_', ' ').title(),
                'description': description.capitalize(),
                'css_class': css,
            })

        # Catch proper nouns spaCy missed as entities
        for token in doc:
            key = token.text.lower().strip()
            if token.pos_ == 'PROPN' and key not in seen:
                gender = self._name_gender(key)
                if gender:
                    seen.add(key)
                    contexts.append({
                        'word': token.text,
                        'type': gender['type'],
                        'description': gender['description'],
                        'css_class': gender['css_class'],
                    })
                elif (token.text[0].isupper()
                      and key not in self.domain_dict
                      and not token.is_stop):
                    seen.add(key)
                    contexts.append({
                        'word': token.text,
                        'type': 'Person',
                        'description': f'"{token.text}" appears to be a personal name',
                        'css_class': 'PERSON',
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
