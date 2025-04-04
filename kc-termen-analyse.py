from lxml import etree
import pandas as pd
from typing import Dict, List, Optional, Any
from itertools import product
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from rapidfuzz.fuzz import ratio
import xml.sax
import csv

file_path = 'data/KC_Thesaurus_all_2025-03-21.xml'

# KC_Thesaurus_all_2025-03-21.
kc_all = 'data/KC_Thesaurus_all_2025-03-21.xml'
kc_collect = 'data/KC_Thesaurus_collect_2025-03-21.xml'

# ---------------check structuur xml----------------------
# SAX alle paden naar een csv

class PathCollector(xml.sax.ContentHandler):
    def __init__(self):
        self.current_path = []
        self.paths = set()

    def startElement(self, name, attrs):
        self.current_path.append(name)
        full_path = "/" + "/".join(self.current_path)
        self.paths.add(full_path)

    def endElement(self, name):
        self.current_path.pop()

def extract_paths_from_xml(file_path):
    handler = PathCollector()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    parser.parse(file_path)
    return sorted(handler.paths)

def export_paths_to_csv(paths, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['xml_path'])
        for path in paths:
            writer.writerow([path])

if __name__ == "__main__":
    xml_file = "data/KC_Thesaurus_all_2025-03-21.xml"
    output_csv = "xml_paths.csv"

    paths = extract_paths_from_xml(xml_file)
    export_paths_to_csv(paths, output_csv)

    print(f"{len(paths)} paths exported to {output_csv}")



#---------


def collect_structure(elem, path="", structure=None):
    if structure is None:
        structure = defaultdict(set)
    current_path = f"{path}/{elem.tag}"
    structure[path].add(elem.tag)
    for child in elem:
        collect_structure(child, current_path, structure)
    return structure

tree = ET.parse(kc_all)
root = tree.getroot()

structure = collect_structure(root)
for parent, children in structure.items():
    print(f"{parent} -> {children}")

#----------parse xml-----------------------------------------
def ensure_list(x: Any) -> List:
    
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def get_text(element: etree._Element, path: str, default: Optional[str] = None) -> Optional[str]:
   # XPath '/text()' to extract the text node.
    found = element.xpath(path)
    return found[0] if found else default

def get_attr(element: etree._Element, attr: str, default: Optional[str] = None) -> Optional[str]:
    # Krijg attribuut per element
    return element.get(attr, default)

def parse_elements(element: etree._Element, path: str, processor) -> List[Dict]:
    #generieke parser
    elements = ensure_list(element.xpath(path))
    return [processor(el) for el in elements]

def process_term(element: etree._Element) -> Dict:
    texts = parse_elements(element, 'text', lambda t: {
        'text': t.text,
        'language': get_attr(t, 'language')
    })
    return {
        'type': get_attr(element, 'option'),
        'value': get_attr(element, 'value'),
        'texts': texts
    }

def process_linked_term(element: etree._Element) -> Dict:
    return {
        'linkref': get_attr(element, 'linkref'),
        'linkfield': get_attr(element, 'linkfield'),
        'linkreffield': get_attr(element, 'linkreffield'),
        'linkdb': get_attr(element, 'linkdb'),
        'term': get_text(element, 'term/text()'),
        'term_number': get_attr(element, 'priref') # % nieuw
    }

def expand_term_data(terms: List[Dict], prefix: str) -> List[Dict]:
    expanded = []
    for term in terms:
        texts = term.get('texts') or []
        if not texts:
            expanded.append({
                f'{prefix}option': term.get('type'),
                f'{prefix}value': term.get('value'),
                f'{prefix}text': None,
                f'{prefix}language': None
            })
        else:
            for text in texts:
                expanded.append({
                    f'{prefix}option': term.get('type'),
                    f'{prefix}value': term.get('value'),
                    f'{prefix}text': text.get('text'),
                    f'{prefix}language': text.get('language')
                })
    return expanded if expanded else [{}]

def expand_linked_terms(terms: List[Dict], prefix: str) -> List[Dict]:
    if not terms:
        return [{}]
    return [{
        f'{prefix}linkref': term.get('linkref'),
        f'{prefix}linkfield': term.get('linkfield'),
        f'{prefix}linkreffield': term.get('linkreffield'),
        f'{prefix}linkdb': term.get('linkdb'),
        f'{prefix}term': term.get('term'),
        f'{prefix}term_number': term.get('term_number') # % nieuw
    } for term in terms]

def process_record(record: etree._Element) -> List[Dict]:
    base = {
        'priref': get_text(record, 'priref/text()'),
        'use_count': get_text(record, 'use_count/text()'),
        'main_term': get_text(record, 'term/text()'),
        'term_number': get_text(record, 'term.number/text()'),
        'source': get_text(record, 'source/text()'),
        'source_link': get_text(record, 'source_link/text()'),
        'scope_note': get_text(record, 'scope_note/text()'),
        'input_date': get_text(record, 'input.date/text()'),
        'input_name': get_text(record, 'input.name/text()'),
        'input_time': get_text(record, 'input.time/text()'),
        'input_source': get_text(record, 'input.source/text()'),
        'narrower_term_lrefs': '|'.join(record.xpath('narrower_term.lref/text()')) or None,
        'broader_term_lrefs': '|'.join(record.xpath('broader_term.lref/text()')) or None,
        'used_for_lrefs': '|'.join(record.xpath('used_for.lref/text()')) or None,
    }
    
    edits = parse_elements(record, 'Edit', lambda e: {
        'edit_date': get_text(e, 'edit.date/text()'),
        'edit_name': get_text(e, 'edit.name/text()'),
        'edit_time': get_text(e, 'edit.time/text()'),
        'edit_source': get_text(e, 'edit.source/text()'),
        'edit_notes': get_text(e, 'edit.notes/text()')
    })
    term_types = parse_elements(record, 'term.type', process_term)
    term_statuses = parse_elements(record, 'term.status', process_term)
    broader_terms = parse_elements(record, 'broader_term', process_linked_term)
    narrower_terms = parse_elements(record, 'narrower_term', process_linked_term)  # % nieuw
    used_for_terms = parse_elements(record, 'used_for', process_linked_term)
    

    combinations = product(
        edits or [{}],
        expand_term_data(term_types, prefix='term_type_'),
        expand_term_data(term_statuses, prefix='status_'),
        expand_linked_terms(broader_terms, prefix='broader_'),
        expand_linked_terms(narrower_terms, prefix='narrower_'),  # % nieuw
        expand_linked_terms(used_for_terms, prefix='used_for_')
    )
    
    rows = []
    for combo in combinations:
        edit_data, type_data, status_data, broader_data, narrower_data, used_for_data = combo  # % aangepast
        row = {**base, **edit_data, **type_data, **status_data, **broader_data, **narrower_data, **used_for_data} # % aangepast
        rows.append(row)
    return rows

def xml_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Parse the XML file and convert it to a pandas DataFrame with all attributes.
    """
    tree = etree.parse(file_path)
    records = tree.xpath('//record')
    data = []
    for record in records:
        data.extend(process_record(record))
    return pd.DataFrame(data)

df = xml_to_dataframe(file_path)

final_columns = [
    'priref', 'use_count', 'main_term', 'term_number',
    'source', 'source_link', 'scope_note',
    'input_date', 'input_name', 'input_time', 'input_source',
    'narrower_term_lrefs', 'broader_term_lrefs', 'used_for_lrefs',
    'edit_date', 'edit_name', 'edit_time', 'edit_source', 'edit_notes',
    'term_type_option', 'term_type_value', 'term_type_text', 'term_type_language',
    'status_option', 'status_value', 'status_text', 'status_language',
    'broader_linkref', 'broader_linkfield', 'broader_linkreffield', 'broader_linkdb', 'broader_term',
    'narrower_linkref', 'narrower_linkfield', 'narrower_linkreffield', 'narrower_linkdb', 'narrower_term_number','narrower_term',  # % nieuw
    'used_for_linkref', 'used_for_linkfield', 'used_for_linkreffield', 'used_for_linkdb', 'used_for_term'
]

# Select only columns that exist in the DataFrame
df = df[[col for col in final_columns if col in df.columns]]
# df.to_csv("kc_export.csv", index=False)
#------------------------------------------------------------------------

# # Filter data
# cols = ['priref', # nummer term in systeem
#  'use_count', # aantal keer term gebruikt
#  'main_term', # prefLabel term
#  'term_number', # AAT URI i.e. 300000240
#  'source', # Thesaurus, i.e. AAT, Wikipedia
#  'source_link', # AAT link
#  'input_name', # wie term heeft toegevoegd
#  'input_source', # waar/ welke collectie het is ingevoerd
#  'narrower_term_lrefs', # nummer narrower term in systeem
#  'broader_term_lrefs', # nummer broader term in systeem
#  'used_for_lrefs', # altLabel, is wel een eigen term in het systeem.
#  'edit_name', # name of the editor
#  'edit_source', # waar de edit heeft plaatsgevonden i.e. thesau
#  'term_type_option', # iets van een categorie binnen de systeem, zoals DIM voor afmeting/dimension
#  'term_type_value', # categorie waarin term onder valt
#  'term_type_text', # in meerdere talen de categorienaam
#  'status_option', # nummer behorend tot de status van een term, bijv. 3=kandidaat
#  'status_value', # label van de status van een term, zoals kandidaat of niet gedefinieerd
#  'status_text', # meerdere vertalingen van de status
#  'status_language', # nummer van de taal i.e. 0 = engels, van de status label
#  'broader_linkref', # systeem id van de broader term
#  'broader_linkfield', # type broader term (is altijd term)
#  'broader_linkreffield', # veldnaam dat aangeeft dat het om een brader term gaat
#  'broader_linkdb', # path in addlib, soort opslag pad
#  'broader_term', # prefLabel broader term
#  'used_for_linkref', # nummer van altLabel term
#  'used_for_linkfield', # wat voor type dat bovenste is (=term)
#  'used_for_linkreffield', #categorie naam (=term)
#  'used_for_linkdb', # # path in addlib, soort opslag pad
#  'used_for_term'] # altLabel main term

#-------------------------------------------------------------
# import pandas as pd
# df = pd.read_csv("kc_export.csv")
#----

def bevat_cijfer_teken_of_1letter(term):
    if pd.isna(term):
        return 'nee'

    # Check op cijfer
    if re.search(r'\d', term):
        return 'ja'
    
    # Check op speciaal teken (niet letter of spatie)
    if re.search(r'[^\w\s]', term):
        return 'ja'
    
    # Check op 1-letterwoorden
    woorden = term.strip().split()
    if any(len(w) == 1 for w in woorden):
        return 'ja'
    
    return 'nee'

#------------fuzzy------------------------------------------

def build_fuzzy_index(term_list):
    term_map = {t.lower(): t for t in term_list if isinstance(t, str)}
    term_dict = defaultdict(list)
    for t_lower in term_map.keys():
        term_dict[len(t_lower)].append(t_lower)
    return term_map, term_dict

def fuzzy_match_with_flag(term, target_dict, target_map, threshold=85, exclude_exact=True):
    """
    Retourneert:
    - 'ja' of 'nee' afhankelijk van of er matches zijn
    - string met: originele_term, match1, match2
    
    Parameters:
    - exclude_exact: als True, sluit exacte matches (zelfde lowercase string) uit
    """
    if pd.isna(term):
        return 'nee', ''
    
    term_orig = str(term)
    term_lower = term_orig.lower().strip()
    term_len = len(term_lower)

    mogelijke_matches = (
        target_dict.get(term_len, []) +
        target_dict.get(term_len - 1, []) +
        target_dict.get(term_len + 1, [])
    )

    matches = [
        target_map[t]
        for t in mogelijke_matches
        if (not exclude_exact or t != term_lower) and ratio(term_lower, t) > threshold
    ]

    if matches:
        return 'ja', f"{term_orig}, {', '.join(matches)}"
    else:
        return 'nee', ''

#-------------------------------------------------------
#-----------------------Toepassingen--------------------------
# Filter data
df_termen_vs_broader = df[
    ['priref','use_count','main_term','term_number','source','source_link','term_type_option','term_type_value','status_value',
     'broader_term_lrefs','broader_term',
     'narrower_term_lrefs','narrower_term_number','narrower_term']
     ]
df_termen_vs_broader = df_termen_vs_broader.drop_duplicates()


df_termen_vs_broader['broader_term_uppercase'] = df_termen_vs_broader['broader_term'].apply(
    lambda term: 'ja' if isinstance(term, str) and term.isupper() else 'nee'
)

# Check of URI ontbreekt
df_termen_vs_broader['main_term_mist_uri'] = df_termen_vs_broader['source_link'].apply(
    lambda x: 'ja' if pd.isna(x) or x == '' else 'nee'
)

# Groeperen er uri en als er >2 zijn worden ze toegevoegd. 
uri_mapping = (
    df_termen_vs_broader
    .loc[df_termen_vs_broader['main_term_mist_uri'] == 'nee']
    .groupby('main_term')['source_link']
    .agg(lambda x: ', '.join(sorted(set(x))) if len(set(x)) > 1 else '')
    .to_dict()
)

# Voeg toe aan kolom, leeg als geen meerdere URIs
df_termen_vs_broader['meerdere_uris_main_term'] = df_termen_vs_broader['main_term'].map(uri_mapping)


#----------Zijn er termen die tot meerdere categorien behoren? i.e. land en provincie?------
unique_term_categorie = df_termen_vs_broader.groupby('main_term')['term_type_value'].nunique()
df_termen_vs_broader['meerdere_categorien_main_term'] = df_termen_vs_broader['main_term'].map(unique_term_categorie).gt(1).map({True:'ja', False: 'nee'})

#----
categorie_per_main = (
    df_termen_vs_broader
    .groupby('main_term')['term_type_option']
    .agg(lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v) and str(v).strip() != ''))))
    .to_dict()
)

def meerdere_categorien(term):
    cat_list = categorie_per_main.get(term, '')
    if ',' in cat_list:
        return cat_list
    else:
        return ''

df_termen_vs_broader['uitwerking_meerdere_categorien_voor_main'] = df_termen_vs_broader['main_term'].map(meerdere_categorien)

#-----
main_term_list = df_termen_vs_broader['main_term'].dropna().unique()
main_term_map, main_term_dict = build_fuzzy_index(main_term_list)

resultaten_main = df_termen_vs_broader['main_term'].apply(
    lambda t: fuzzy_match_with_flag(t, main_term_dict, main_term_map)
)

df_termen_vs_broader[['meerdere_spelling_main_term', 'gevonden_matches_main_in_main']] = pd.DataFrame(
    resultaten_main.tolist(), index=df_termen_vs_broader.index
)
#----
resultaten_broader_to_main = df_termen_vs_broader['broader_term'].apply(
    lambda t: fuzzy_match_with_flag(t, main_term_dict, main_term_map, threshold=80)
)

df_termen_vs_broader[['broader_term_in_main_term', 'broader_terms_match_main_term']] = pd.DataFrame(
    resultaten_broader_to_main.tolist(), index=df_termen_vs_broader.index
)
#----
broader_term_list = df_termen_vs_broader['broader_term'].dropna().unique()
broader_term_map, broader_term_dict = build_fuzzy_index(broader_term_list)

resultaten_broader_self = df_termen_vs_broader['broader_term'].apply(
    lambda t: fuzzy_match_with_flag(t, broader_term_dict, broader_term_map, exclude_exact=True)
)

df_termen_vs_broader[['meerdere_spelling_broader_in_broader', 'gevonden_spellingen_broader_in_broader']] = pd.DataFrame(
    resultaten_broader_self.tolist(), index=df_termen_vs_broader.index
)
#------
df_termen_vs_broader['main_term_bevat_tekens_of_cijfers'] = df_termen_vs_broader['main_term'].apply(bevat_cijfer_teken_of_1letter)
#----
resultaten_narrower = df_termen_vs_broader['narrower_term'].apply(
    lambda t: fuzzy_match_with_flag(t, main_term_dict, main_term_map, exclude_exact=False)
)
df_termen_vs_broader[['narrower_in_main_term', 'gevonden_matches_narrower_in_main']] = pd.DataFrame(resultaten_narrower.tolist(), index=df_termen_vs_broader.index)
#----
#--------------------naar excel------------------------------------------------------------------------------
order_kolommen = ['priref', 'use_count', 'main_term', 'term_number', 'source',
       'source_link','term_type_option', 'term_type_value', 'status_value', 'main_term_mist_uri',
       'meerdere_uris_main_term', 'meerdere_categorien_main_term',
       'meerdere_spelling_main_term', 'gevonden_matches_main_in_main','main_term_bevat_tekens_of_cijfers',

       'broader_term_lrefs','broader_term', 'broader_term_uppercase', 
       'broader_term_in_main_term', 'broader_terms_match_main_term',
       'meerdere_spelling_broader_in_broader','gevonden_spellingen_broader_in_broader',
       
       'narrower_term_lrefs', 'narrower_term_number',
       'narrower_term','narrower_in_main_term','gevonden_matches_narrower_in_main']

df_termen_vs_broader = df_termen_vs_broader[order_kolommen]
#-----
with pd.ExcelWriter("kc_datakwaliteit.xlsx", engine='openpyxl', mode='w') as writer:
    df_termen_vs_broader.to_excel(writer, sheet_name='termen_kwaliteitscheck', index=False)