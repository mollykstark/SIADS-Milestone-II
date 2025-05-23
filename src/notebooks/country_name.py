"""
Python Script to create a column labeling the country of origin for MET artworks.
"""
from functools import reduce
import pandas as pd

def determine_met_country_names(met_df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Function to create a column labeling the country of origin for MET artworks.

    Args:
        file_path (str): A string of the path to the CSV file 
        containing relevant country information.

    Returns:
        Pandas DataFrame: A Pandas dataframe containing the 
        original dataframe with the additional country column.
    """

    country_classification_columns = (
        "country",
        "culture",
        "artist_nationality",
        "region",
        "subregion",
        "tags",
    )

    met_df.loc[:,"combined_country"] = reduce(
        lambda x, y: x.combine_first(met_df[y]),
        country_classification_columns,
        met_df[country_classification_columns[0]],
    )

    country_terms = pd.read_csv(file_path)
    country_terms = country_terms[
        [
            "name.common",
            "name.official",
            "capital",
            "altSpellings",
            "demonyms.eng.m",
            "demonyms.eng.f",
        ]
    ]

    country_terms = country_terms[
        country_terms["name.common"] != "British Indian Ocean Territory"
    ]
    country_terms = country_terms[
        country_terms["name.common"] != "United States Minor Outlying Islands"
    ]

    new_rows = pd.DataFrame(
        {
            "name.common": ["French Polynesia", "Nepal", "Tibet"],
            "demonyms.eng.m": ["French Polynesian", "Nepalese", "Tibetan"],
        }
    )

    country_terms = pd.concat([country_terms, new_rows], ignore_index=True)
    country_terms.fillna("no data", inplace=True)

    country_map = {}

    for _, row in country_terms.iterrows():
        official_name = row["name.common"]
        country_map[official_name] = official_name

        for col in country_terms.columns:
            if col != "name.common":
                country_map[row[col]] = official_name

    country_map.update(
        {
            "Netherlandish": "Netherlands","Bohemian": "Czechia","Byzantine Egypt": "Egypt",
            "Flemish": "Belgium","Scottish": "United Kingdom","Welsh": "United Kingdom",
            "Korea": "Korea","England": "United Kingdom","Scotland": "United Kingdom",
            "Wales": "United Kingdom","Sumerian": "Iraq","Mesopotamia": "Iraq",
            "Etruscan": "Italy","Minoan": "Greece","West Bengal": "India",
            "Vendel": "Sweden","Edomite": "Jordan","Sino-Tibet": "Tibet",
            "Philippine": "Philippines","North China": "China","Alsatian": "France",
            "Côte d'Ivoire": "Ivory Coast","Malayan": "Malaysia","Mycenaean": "Greece",
            "Republic of Congo": "DR Congo","Hittite": "Turkey","Sasanian": "Iran",
            "Sarawak": "Malaysia","Aegean": "Greece","South India": "India",
            "North German": "Germany","Italic": "Italy","Formosan": "Taiwan",
            "South Netherlands": "Netherlands","Elamite": "Iran","Javanese": "Indonesia",
            "North America": "United States","Sicilian": "Italy","Hattian": "Turkey",
            "America": "United States","Macedonia": "North Macedonia","Yi": "China",
            "Greek Islands": "Greece","Greek (Attic)": "Greece","Thailand (Ban Chiang)": "Thailand",
            "North French": "France","Indian": "India","Avar": "Russia",
            "South Netherlandish": "Netherlands","Alanic": "Russia","Caroline Islands": "Palau",
            "Southern German": "German","Native American": "United States","Celtic": 
            "United Kingdom","Cycladic": "Greece","Mangareva": "French Polynesia",
            "Burma": "Myanmar","Franco-Netherlandish": "France","the Republic of Congo": "DR Congo",
            "Persian": "Iran","Rhodian": "Greece","Indus": "Pakistan",
            "Nabataean": "Jordan","Yortan": "Turkey","North Spanish": "Spain",
            "Argentinian": "Argentina","Swiss French": "Switzerland","Campanian": "Italy",
            "The Netherlands": "Netherlands","South Indian": "India","USA": "United States",
            "Façon de Venise": "Italy","Venetian": "Italy","Lydian": "Turkey",
            "Deccan": "India","Sarmatian": "Iran","Western Turkmenistan": "Turkmenistan",
            "Western Iran": "Iran","Italic-Native": "Italy","Villanovan": "Italy",
            "Baluchistan": "Pakistan","South Italian": "Italy","South German": "Germany",
            "Western Tibet": "Tibet","west-central Turkey": "Turkey","West Indian": "India",
            "Akkadian": "Iraq","South Netherland": "Netherlands","Chemehuevi": "United States",
            "Sumatran": "Indonesia","Gemany": "Germany","Egypt possibly": "Egypt",
            "Coptic": "Egypt","Xiongnu": "Mongolia","Phoenician": "Lebanon","Mitanni": "Turkey",
            "North Netherlandish": "Netherlands","Indigenous American": "United States",
            "Icelandic": "Iceland","Nias": "Indonesia","North Indian": "India",
            "Central India": "India","Helladic": "Greece","Marquesas Islands": "French Polynesia",
            "Nubia": "Sudan","Dyak": "Indonesia","North Netherland": "Netherlands",
            "Anatolia": "Turkey","Silesian": "Poland","Nias people": "Indonesia",
            "Greek Neolithic": "Greece","Ubaid": "Iraq","North Netherlands": "Netherlands",
            "Ottoman": "Turkey","Anglo-Saxon": "United Kingdom","Saxon": "Germany",
            "Phrygian": "Turkey","Sassanian": "Iran","Catalan": "Spain",
            "Chimú": "Peru","Siberia": "Russia","Democratic Republic of Congo": "DR Congo",
            "Urartian": "Armenia","Neo-Sumerian": "Iraq","New Guinea": "Australia",
            "Cretan": "Greece","Babylonian": "Iraq","Indonesia(": "Indonesia",
            "Russia Federation": "Russia","Borneo": "Indonesia","Minangkabau": "Indonesia",
            "Republic of Kiribati": "Kiribati","Lapland": "Finland","Vietnam(": "Vietnam",
            "Balinese": "Indonesia","Egyptian possibly": "Egypt","First Nations": "Canada",
            "South China": "China","North Central Indian": "India","Buganda": "Uganda",
            "Northern Italian": "Italy","Madurese": "Indonesia","Praenestine": "Italy",
            "Kirghiztan": "Kyrgyzstan","Tuscany": "Italy","North France": "France",
            "Canaanite": "Palestine","North Italian": "Italy","Western Australia": "Australia",
            "Façcon de Venise": "Italy","Madura": "Indonesia","Sardinian": "Italy",
            "Sienese": "Italy","Venice": "Italy","Sulawesi": "Indonesia",
            "Western India": "India","Euboean": "Greece","Congo": "DR Congo",
            "Southwestern German": "Germany","Rumanian": "Romania","Central Tibet": "Tibet",
            "Sinhala": "Sri Lanka","Proto-Elamite": "Iran","Ceylonese": "Sri Lanka",
            "Ojibwe": "Canada","North India": "India","Anatolian": "Turkey",
            "Kuna": "Panama","Livonian": "Latvia","Austia": "Austria",
            "British and American": "United Kingdom","Batak": "Indonesia","Locrian": "Greece",
            "Argentinean": "Argentina","North Central Thailand": "Thailand",
            "northern France": "France","Northwestern Iran": "Iran","Flanders": "Belgium",
            "Crimean": "Ukraine","Bornean": "Indonesia","Leipzig": "Germany","iran": "Iran",
            "Alemannic": "Switzerland","Friesland": "Netherlands","Southwestern Iran": "Iran",
            "Diné": "United States","German(": "Germany","Southwest Italian": "Italy",
            "Greenlandish": "Denmark","Cote d'Ivoire": "Ivory Coast","South French": "France",
            "Tewa": "United States","Southern French": "France","Central Italian": "Italy",
            "Central Turkey": "Turkey","English": "United Kingdom","Volga region": "Russia",
            "Central Italy": "Italy","West Central Turkey": "Turkey","Bohemia": "Czechia",
            "southern England": "United Kingdom","Dimini culture": "Greece",
            "the Netherlands": "Netherlands","Slovakian": "Slovakia","Native Italic": "Italy",
            "Façon de venise": "Italy","Antwerp": "Belgium","Peninsular Thailand": "Thailand",
            "Sesklo culture": "Greece","North Swiss": "Switzerland","Siberian": "Russia",
            "Apuilan": "Italy","Republic of  Cameroon": "Cameroon","Central Indian": "India",
            "Corsican": "France","Southern Netherlandish": "Netherlands","Central Iran": "Iran",
            "Hacilar": "Turkey","Carthaginian": "Tunisia","Circassian": "Russia",
            "Isin-Larsa": "Iraq","Republic of Timor-Leste": "Timor-Leste","UK": "United Kingdom",
            "Central Anatolia": "Turkey","Native Italian": "Italy","Central German": "Germany",
            "Anvers": "Belgium",
        }
    )

    met_df.loc[:,"combined_country"] = met_df["combined_country"].str.strip()
    met_df.loc[:,"combined_country"] = (
        met_df["combined_country"].str.split(r"[,/.:;?]| or| \(").str[0]
    )

    met_df.loc[:,"combined_country"] = met_df["combined_country"].str.lstrip("|")
    met_df.loc[:,"combined_country"] = met_df["combined_country"].str.split("|").str[0]

    met_df.loc[:,"combined_country"] = met_df["combined_country"].str.replace(
        r"^(northern|southern|eastern|east|probably|northeastern"
        r"|northwest|northwestern|northeast|southeastern)\s+",
        "",
        case=False,
        regex=True,
    )

    met_df.loc[:,"combined_country"] = met_df["combined_country"].str.replace(
        r"^(possibly|Byzantine|present-day|Colonial|north-central|)\s+",
        "",
        case=False,
        regex=True,
    )

    met_df.loc[:,"mapped_country"] = met_df.loc[:,"combined_country"].map(country_map)

    return met_df
