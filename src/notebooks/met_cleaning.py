"""
Python Script to clean raw MET data in preparation for data preprocessing and model training.
"""
import warnings
from country_name import determine_met_country_names
import pandas as pd

warnings.filterwarnings('ignore')

def clean_met_data() -> pd.DataFrame:
    """
    Function to clean raw MET data in preparation for data preprocessing and model training.

    Args:
        None

    Returns:
        met: A Pandas dataframe containing the clean MET dataframe.
    """

    # Read in full dataset from MET github
    met = pd.read_csv('https://media.githubusercontent.com/media/metmuseum'
                      '/openaccess/refs/heads/master/MetObjects.csv')

    met = met.rename(columns={'Object ID':'object_id','Object Number':'object_number',
                              'Is Highlight':'is_highlight', 'Is Timeline Work':'is_timeline_work', 
                              'Is Public Domain':'is_public_domain','Gallery Number':
                              'gallery_number', 'Department':'department', 'AccessionYear':
                              'accession_year', 'Object Name':'object_name', 'Title':'title',
                              'Culture':'culture', 'Period':'period', 'Dynasty':'dynasty',
                              'Reign':'reign', 'Portfolio':'portfolio', 'Constituent ID':
                              'constituent_id','Artist Role':'artist_role', 'Artist Prefix':
                              'artist_prefix', 'Artist Display Name':'artist_display_name',
                              'Artist Display Bio':'artist_display_bio', 'Artist Suffix':
                              'artist_suffix', 'Artist Alpha Sort':'artist_alpha_sort',
                              'Artist Nationality':'artist_nationality', 'Artist Begin Date':
                              'artist_begin_date', 'Artist End Date':'artist_end_date',
                              'Artist Gender':'artist_gender', 'Artist ULAN URL':'artist_ulan_url',
                              'Artist Wikidata URL':'artist_wikidata_url','Object Date':
                              'object_date', 'Object Begin Date':'object_begin_date',
                              'Object End Date':'object_end_date', 'Medium':'medium',
                              'Dimensions':'dimensions', 'Credit Line':'credit_line',
                              'Geography Type':'geography_type', 'City':'city', 'State':
                              'state','County':'county', 'Country':'country', 'Region':
                              'region', 'Subregion':'subregion', 'Locale':'locale', 'Locus':
                              'locus','Excavation':'excavation', 'River':'river',
                              'Classification':'classification', 'Rights and Reproduction':
                              'rights_and_reproduction','Link Resource':'link_resource',
                              'Object Wikidata URL':'object_wikidata_url', 'Metadata Date':
                              'metadata_date', 'Repository':'repository','Tags':'tags',
                              'Tags AAT URL':'tags_aat_url', 'Tags Wikidata URL':
                              'tags_wikidata_url'})
    met.set_index('object_id', inplace=True)

    met = determine_met_country_names(met, "../data/helper/countries.csv")

    met['is_significant'] = met['is_timeline_work'] | met['is_highlight']

    met['age'] = 2025 - pd.to_numeric(met['object_begin_date'])
    met['access_year'] = met['accession_year'].astype(str).str.findall(r'\d{4}').str[0]

    met['artist_gender'] = met['artist_gender'].replace(False, "0")
    met['artist_gender'] = met['artist_gender'].replace(True, "1")

    met.loc[:,'tags'] = met['tags'].str.split(pat="|")

    met = met[['is_significant', 'is_public_domain',
            'department', 'access_year', 'object_name',
            'title', 'culture', 'portfolio',
            'artist_display_name', 
            'artist_nationality', 
            'artist_gender', 'mapped_country',
            'age',  'medium', 'tags']]

    return met
