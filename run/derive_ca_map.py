import pandas as pd

import unicodedata
from rapidfuzz import fuzz, process

def normalize_text(text):
    """
    Normalize text by:
    - Converting to lowercase
    - Removing accents
    - Standardizing apostrophes
    - Removing extra whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Standardize apostrophes
    text = text.replace("'", "'").replace("'", "'")
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def discover_common_prefixes(texts, min_frequency=2, min_length=3, max_prefix_length=50):
    """
    Dynamically discover common prefixes in a collection of texts.
    
    Parameters:
    -----------
    texts : list of str
        List of texts to analyze for common prefixes
    min_frequency : int, default=2
        Minimum number of texts that must share a prefix
    min_length : int, default=3
        Minimum length of prefixes to consider
    max_prefix_length : int, default=50
        Maximum length of prefixes to consider
        
    Returns:
    --------
    list of str
        List of common prefixes sorted by specificity (longest first)
    """
    if not texts:
        return []
    
    # Count prefix frequencies
    prefix_counts = {}
    
    for text in texts:
        text_lower = text.lower().strip()
        
        # Try all possible prefixes from min_length to max_prefix_length
        for length in range(min_length, min(max_prefix_length + 1, len(text_lower) + 1)):
            prefix = text_lower[:length]
            
            # Only consider prefixes that end with a space or apostrophe (common word boundaries)
            if prefix.endswith(' ') or prefix.endswith("'"):
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    
    # Filter prefixes by minimum frequency
    common_prefixes = [
        prefix for prefix, count in prefix_counts.items()
        if count >= min_frequency
    ]
    
    # Remove overlapping prefixes (if a longer prefix exists, remove shorter ones that are substrings)
    filtered_prefixes = []
    for prefix in sorted(common_prefixes, key=len, reverse=True):  # Start with longest
        # Check if this prefix is already covered by a longer prefix
        is_covered = False
        for existing_prefix in filtered_prefixes:
            if existing_prefix.startswith(prefix):
                is_covered = True
                break
        
        if not is_covered:
            filtered_prefixes.append(prefix)
    
    # Sort by length (longest first) for specificity
    filtered_prefixes.sort(key=len, reverse=True)
    
    return filtered_prefixes

def truncate_common_prefixes(text, common_prefixes=None):
    """
    Remove common court prefixes to focus on the city/location name.
    
    Parameters:
    -----------
    text : str
        The court name to process
    common_prefixes : list of str, optional
        List of common prefixes to remove. If None, uses hardcoded defaults.
        
    Returns:
    --------
    str
        The text with common prefixes removed
    """
    text_lower = text.lower().strip()
    
    # Use provided prefixes or fall back to hardcoded ones
    if common_prefixes is None:
        prefixes_to_remove = [
            "tribunal supérieur d'appel de ",
            "cour d'appel d'",
            "cour d'appel de ",
            "tribunal supérieur d'appel d'",
        ]
    else:
        prefixes_to_remove = common_prefixes
    
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            return text_lower[len(prefix):].strip()
    
    return text_lower

def create_location_mapping(list1, list2, threshold=80, method='token_sort_ratio', 
                          auto_detect_prefixes=True, min_prefix_frequency=2):
    """
    Create a mapping between two sets of geographic locations using fuzzy matching.
    
    Parameters:
    -----------
    list1 : list of str
        First set of locations to map from
    list2 : list of str
        Second set of locations to map to
    threshold : int, default=80
        Minimum similarity score (0-100) to consider a match
    method : str, default='token_sort_ratio'
        Fuzzy matching method: 'ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio'
    auto_detect_prefixes : bool, default=True
        Whether to automatically detect common prefixes from the data
    min_prefix_frequency : int, default=2
        Minimum frequency for prefix detection (only used if auto_detect_prefixes=True)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: 'set1_original', 'set2_original', 'set1_normalized', 
        'set2_normalized', 'similarity_score'
    """
    # Convert to lists if sets)
    
    # Discover common prefixes if requested
    common_prefixes = None
    if auto_detect_prefixes:
        all_texts = list1 + list2
        common_prefixes = discover_common_prefixes(
            all_texts, 
            min_frequency=min_prefix_frequency,
            min_length=3,
            max_prefix_length=50
        )
        print(f"Discovered {len(common_prefixes)} common prefixes:")
        for prefix in common_prefixes[:10]:  # Show first 10
            print(f"  '{prefix}'")
        if len(common_prefixes) > 10:
            print(f"  ... and {len(common_prefixes) - 10} more")
    
    # Normalize both sets and truncate common prefixes
    normalized1 = {loc: truncate_common_prefixes(normalize_text(loc), common_prefixes) for loc in list1}
    normalized2 = {loc: truncate_common_prefixes(normalize_text(loc), common_prefixes) for loc in list2}
    
    # Choose scorer based on method
    scorer_map = {
        'ratio': fuzz.ratio,
        'partial_ratio': fuzz.partial_ratio,
        'token_sort_ratio': fuzz.token_sort_ratio,
        'token_set_ratio': fuzz.token_set_ratio
    }
    scorer = scorer_map.get(method, fuzz.token_sort_ratio)
    
    # Create mappings
    mappings = []
    
    for loc1_orig, loc1_norm in normalized1.items():
        # Find best match in set2
        result = process.extractOne(
            loc1_norm,
            normalized2.values(),
            scorer=scorer
        )
        
        if result:
            matched_norm = result[0]
            score = result[1]
            
            # Special handling for very short normalized names (likely truncated)
            # Try to find better matches using partial matching
            if len(loc1_norm) <= 10 and score < 70:
                # Try partial ratio for short names
                partial_result = process.extractOne(
                    loc1_norm,
                    normalized2.values(),
                    scorer=fuzz.partial_ratio
                )
                
                if partial_result and partial_result[1] > score:
                    matched_norm = partial_result[0]
                    score = partial_result[1]
            
            # Find original location from normalized
            loc2_orig = [k for k, v in normalized2.items() if v == matched_norm][0]
            if score > threshold:
                mappings.append({
                    'left_o': loc1_orig,
                    'right_o': loc2_orig,
                    'left_normalized': loc1_norm,
                    'right_normalized': matched_norm,
                    'similarity_score': round(score, 2)
                })
    
    # Create DataFrame
    df = pd.DataFrame(mappings)
    
    # Sort by similarity score (descending)
    if not df.empty:
        df = df.sort_values('similarity_score', ascending=False).reset_index(drop=True)
    
    return df


# Load court data
df = pd.read_csv("~/data/legal/cour_tj_correspondence.csv")
dfm = pd.read_csv("~/data/legal/justice.dataset.c/mentions.csv")

dfm["mention"] = dfm["mention"].apply(lambda x: x.lower())
# Get unique court names
ca_dfmr = sorted(dfm.mention.unique())  # Mention courts
ca_dfg = sorted(df[df.columns[1]].unique())  # Ground truth courts

# Create mapping with dynamic prefix detection
# This automatically discovers common prefixes like "Cour d'Appel de" and "tribunal supérieur d'appel de"
# and removes them before fuzzy matching to focus on city/location names
mapping_df = create_location_mapping(ca_dfmr, ca_dfg, threshold=60, auto_detect_prefixes=True)

# Save results sorted by similarity score (ascending to see worst matches first)
mapping_df.sort_values("similarity_score", ascending=True).to_csv("/home/alexander/tmp/cour.map.csv")

df_mr = pd.merge(mapping_df, df, left_on="right_o", right_on="Cour d'Appel compétente")

df_lmr = pd.merge(dfm, df_mr, right_on="left_o", left_on="mention")