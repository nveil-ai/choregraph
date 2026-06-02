# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Geolocation operations for choregraph.

This module provides functions for geocoding locations and retrieving country boundaries.
"""

import json
import unicodedata
from pathlib import Path

import pandas as pd


def geocode_location(df: pd.DataFrame, column: str, scope: str = "unknown") -> pd.DataFrame:
    """
    Enriches a DataFrame with geolocation data based on a column containing location names or codes.
    
    Adds the following columns: latitude, longitude, country, iso2, iso3.
    
    Args:
        df: Input DataFrame
        column: Name of the column containing location names (e.g., "France", "Paris") or codes (e.g., "FR", "US")
        scope: Hint for the type of locations - "city", "country", or "unknown" (default).
               "unknown" will try to match countries first, then cities.
    
    Returns:
        DataFrame with added columns: latitude, longitude, country, iso2, iso3
    """
    from choregraph._extras import optional_dep
    with optional_dep():
        import geonamescache

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    # Initialize cache
    gc = geonamescache.GeonamesCache(min_city_population=5000)
    countries = gc.get_countries()
    countries_by_name = gc.get_countries_by_names()
    cities = gc.get_cities()
    
    # === BUILD COUNTRY LOOKUPS ===
    # By ISO2, ISO3, FIPS codes
    iso2_lookup = {info['iso'].upper(): info for info in countries.values()}
    iso3_lookup = {info['iso3'].upper(): info for info in countries.values()}
    fips_lookup = {info['fips'].upper(): info for info in countries.values() if info.get('fips')}
    
    # By name (lowercase) - includes official names from geonamescache
    country_name_lookup = {name.lower(): info for name, info in countries_by_name.items()}
    # Also add the 'name' field directly (handles cases like "Netherlands" -> "The Netherlands")
    for info in countries.values():
        country_name_lookup[info['name'].lower()] = info
    
    # Add common variations for countries with articles or alternative names
    # Handle "The X" -> "X" pattern (Netherlands, Bahamas, Philippines, etc.)
    for name, info in list(country_name_lookup.items()):
        if name.startswith('the '):
            short_name = name[4:]  # Remove "the "
            if short_name not in country_name_lookup:
                country_name_lookup[short_name] = info
    
    # Add well-known alternative country names not covered by ISO/FIPS
    COUNTRY_ALTERNATES = {
        'great britain': 'GB', 'britain': 'GB', 'england': 'GB', 'scotland': 'GB', 'wales': 'GB',
        'russia': 'RU', 'ussr': 'RU', 'soviet union': 'RU',
        'korea': 'KR', 'south korea': 'KR', 'republic of korea': 'KR',
        'north korea': 'KP', 'dprk': 'KP',
        'iran': 'IR', 'persia': 'IR',
        'czech republic': 'CZ', 'czechia': 'CZ',
        'ivory coast': 'CI', "cote d'ivoire": 'CI',
        'uae': 'AE', 'emirates': 'AE',
        'holland': 'NL',
        'burma': 'MM', 'myanmar': 'MM',
        'congo': 'CD', 'drc': 'CD', 'zaire': 'CD',
        'vatican': 'VA', 'holy see': 'VA',
        'taiwan': 'TW', 'republic of china': 'TW',
        'palestine': 'PS',
        'east timor': 'TL', 'timor-leste': 'TL',
        # Regions/states often used as location names (e.g., F1 Grand Prix names)
        'emilia romagna': 'IT', 'emilia-romagna': 'IT',  # Imola GP
        'styria': 'AT', 'styrian': 'AT',  # Austrian GP variant
        'tuscany': 'IT', 'toscana': 'IT',  # Mugello GP
        'eifel': 'DE',  # Nürburgring GP
        'sakhir': 'BH',  # Bahrain outer circuit
        'catalonia': 'ES', 'catalunya': 'ES',  # Spanish GP (Barcelona)
        'lombardy': 'IT', 'lombardia': 'IT',  # Monza region
        'las vegas': 'US',  # Las Vegas GP
        'jeddah': 'SA',  # Saudi Arabian GP city
        'imola': 'IT',  # Emilia Romagna GP circuit
        'baku': 'AZ',  # Azerbaijan GP city
        'sochi': 'RU',  # Russian GP city
        'portimao': 'PT', 'portimão': 'PT',  # Portuguese GP
        'sepang': 'MY',  # Malaysian GP
        'yas marina': 'AE', 'yas island': 'AE',  # Abu Dhabi GP
    }
    for alt_name, iso2 in COUNTRY_ALTERNATES.items():
        if alt_name not in country_name_lookup and iso2 in iso2_lookup:
            country_name_lookup[alt_name] = iso2_lookup[iso2]
    
    # === BUILD CITY LOOKUPS ===
    # By primary name (keeping highest population per name)
    city_by_name = {}
    # By alternate names (keeping highest population per alternate)
    city_by_altname = {}
    
    for city_id, city_info in cities.items():
        # Primary name
        city_name_lower = city_info['name'].lower()
        if city_name_lower not in city_by_name or city_info['population'] > city_by_name[city_name_lower]['population']:
            city_by_name[city_name_lower] = city_info
        
        # Alternate names (includes translations, local names, etc.)
        for altname in city_info.get('alternatenames', []):
            if isinstance(altname, str) and len(altname) > 1:
                altname_lower = altname.lower()
                if altname_lower not in city_by_altname or city_info['population'] > city_by_altname[altname_lower]['population']:
                    city_by_altname[altname_lower] = city_info
    
    # Cache for capital coordinates (lazily populated)
    capital_coords_cache = {}
    
    def get_capital_coords(capital: str) -> tuple:
        """Get capital coordinates with caching."""
        if capital in capital_coords_cache:
            return capital_coords_cache[capital]
        
        capital_lower = capital.lower()
        
        # Try exact match in city lookup first (fast)
        city_info = city_by_name.get(capital_lower) or city_by_altname.get(capital_lower)
        if city_info:
            coords = (city_info['latitude'], city_info['longitude'])
            capital_coords_cache[capital] = coords
            return coords
        
        capital_coords_cache[capital] = (None, None)
        return (None, None)

    def normalize_text(text: str) -> str:
        """Normalize text by replacing unicode whitespace and stripping."""
        # Normalize unicode (NFKC converts non-breaking spaces to regular spaces)
        text = unicodedata.normalize('NFKC', text)
        # Replace any remaining unicode whitespace with regular space
        text = ''.join(' ' if c.isspace() else c for c in text)
        # Collapse multiple spaces and strip
        return ' '.join(text.split())

    def get_row_data(name: str, current_scope: str) -> list:
        if not name or not isinstance(name, str):
            return [None] * 5
        name_clean = normalize_text(name)
        if not name_clean:
            return [None] * 5
        name_upper = name_clean.upper()
        name_lower = name_clean.lower()

        # 1. PRIORITY: Country/Code Match
        if current_scope in ["country", "unknown"]:
            # Try all code lookups (ISO2, ISO3, FIPS)
            c_match = iso2_lookup.get(name_upper) or iso3_lookup.get(name_upper) or fips_lookup.get(name_upper)
            
            # Try name lookup
            if not c_match:
                c_match = country_name_lookup.get(name_lower)
            
            if c_match:
                capital = c_match.get('capital', '')
                lat, lon = get_capital_coords(capital) if capital else (None, None)
                return [lat, lon, c_match['name'], c_match['iso'], c_match['iso3']]

        # 2. PRIORITY: City Match
        if current_scope in ["city", "unknown"]:
            # Check both primary and alternate name lookups, prefer higher population
            primary_match = city_by_name.get(name_lower)
            alt_match = city_by_altname.get(name_lower)
            
            # Choose the match with higher population (handles cases like "Geneva" - US city vs Swiss Genève)
            if primary_match and alt_match:
                city_info = primary_match if primary_match['population'] >= alt_match['population'] else alt_match
            else:
                city_info = primary_match or alt_match
            
            if city_info:
                c_info = iso2_lookup.get(city_info['countrycode'])
                if c_info:
                    return [city_info['latitude'], city_info['longitude'], c_info['name'], c_info['iso'], c_info['iso3']]
                return [city_info['latitude'], city_info['longitude'], None, city_info['countrycode'], None]
        
        return [None] * 5

    # Deduplicate unique names for massive speedup
    unique_names = df[column].unique()
    unique_map = {name: get_row_data(name, scope) for name in unique_names}

    res_df = pd.DataFrame(
        df[column].map(unique_map).tolist(),
        index=df.index,
        columns=['latitude', 'longitude', 'country', 'iso2', 'iso3']
    )
    
    return pd.concat([df, res_df], axis=1)


def get_country_contours(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Joins country boundary geometries to the DataFrame based on ISO3 codes in the specified column.
    
    Uses Natural Earth 10m resolution data bundled with the package.
    Geometries are stored as GeoJSON strings for parquet compatibility and direct use with deck.gl.
    
    Args:
        df: Input DataFrame containing ISO3 codes
        column: Name of the column containing ISO3 country codes (e.g., "FRA", "USA", "BRA")
    
    Returns:
        DataFrame with the original columns plus a 'geometry' column containing GeoJSON strings
    """
    from choregraph._extras import optional_dep
    with optional_dep():
        import geopandas as gpd
    from shapely.geometry import mapping
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    # Get unique ISO3 codes from the column
    iso3_list = df[column].dropna().unique().tolist()
    
    # Use bundled asset file - assets are in parent package directory
    assets_dir = Path(__file__).parent.parent / "assets"
    filepath = assets_dir / "ne_10m_admin_0_countries.zip"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Natural Earth data not found at {filepath}. "
            "Please ensure the assets directory contains ne_10m_admin_0_countries.zip"
        )

    world = gpd.read_file(filepath)
    # Filter to requested ISO3 and keep only geometry and key for join
    contours = world[world['ADM0_A3'].isin(iso3_list)][['ADM0_A3', 'geometry']].copy()
    
    # Convert shapely geometries to GeoJSON strings for parquet compatibility
    contours['geometry'] = contours['geometry'].apply(
        lambda g: json.dumps(mapping(g)) if g is not None else None
    )
    
    # Join geometry to the original dataframe on the ISO3 column
    result = df.merge(contours, left_on=column, right_on='ADM0_A3', how='left')
    # Drop the ADM0_A3 column used for joining (keep only geometry)
    result = result.drop(columns=['ADM0_A3'])
    
    # Return as regular DataFrame (not GeoDataFrame) since geometry is now GeoJSON strings
    return pd.DataFrame(result)
