import os
import re
import time
from ast import literal_eval
from importlib import import_module

import numpy as np
import pandas as pd
import requests

from .. import logger
from ..method import scan_method

method_meta_filename = f"{os.path.dirname(__file__)}/method_meta_dataframe.csv"
ref_filename = f"{os.path.dirname(__file__)}/method_meta_dataframe_2026.04.08.18.00.00.csv"
# TODO: remove from the package and put it in a use dir


def get_method_meta(regenerate=False):
    # Build metadata file on first use, or regenerate when explicitly requested.
    if regenerate or (not os.path.exists(method_meta_filename)):
        regenerate_method_meta()

    method_meta_df = pd.read_csv(method_meta_filename, index_col=0)
    return method_meta_df


def update_method_meta(ref_filename=ref_filename, method_meta_filename=method_meta_filename):
    # Implementation for updating method metadata
    ref_df = pd.read_csv(ref_filename, index_col=0)
    method_meta_df = pd.read_csv(method_meta_filename, index_col=0)

    def update_row(row):
        method_name = row.name
        if method_name not in ref_df.index:
            logger.debug(f"Method {method_name} not found in reference data, skipping update.")
            return row
        else:
            if pd.isna(row["citations"]) and pd.notna(ref_df.loc[method_name, "citations"]):
                row["citations"] = ref_df.loc[method_name, "citations"]
                logger.info(f"Updated citations for {method_name}: {row['citations']}")
            if pd.isna(row["stars"]) and pd.notna(ref_df.loc[method_name, "stars"]):
                row["stars"] = ref_df.loc[method_name, "stars"]
                logger.info(f"Updated stars for {method_name}: {row['stars']}")
            return row

    method_meta_df = method_meta_df.apply(update_row, axis=1)
    return method_meta_df


def regenerate_method_meta(method_meta_filename=method_meta_filename):
    # scan methods
    method_meta_df = scan_method(scan_dynverse_method=True)  # scan cafe + dynverse methods

    method_meta_df = add_prior_information_level(method_meta_df)  # prior level
    method_meta_df = add_citations_and_stars(method_meta_df)  # get google scholar citation and github star

    # fine tune
    if "palantir" in method_meta_df.index:
        method_meta_df.loc[
            "palantir", "wrapper_type"
        ] = "Lineage"  # palantir is linear for benchmark, although it can use 'Lineage' and 'Probability' wrapper

    method_meta_df.to_csv(method_meta_filename, index=True)
    logger.info(f"method metadata regenerated: {method_meta_filename}")
    return method_meta_df


def add_prior_information_level(method_meta_df):
    # get prior information level for every method
    # ref to: dev/hzy/25.12.19-scan_method_to_get_prior_information.ipynb
    prior_keys = ["cluster", "basis", "start_cell"]

    prior_information_df = pd.DataFrame("Not Required", index=method_meta_df.index, columns=prior_keys)

    def _safe_parameter_dict(value):
        if isinstance(value, dict):
            return value
        if pd.isna(value):
            return {}
        if isinstance(value, str):
            # Parameter column is expected to be dict; tolerate stringified dict.
            try:
                parsed = literal_eval(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    if "parameter" in method_meta_df.columns:
        for method_name, param_value in method_meta_df["parameter"].items():
            param_dict = _safe_parameter_dict(param_value)
            for key in prior_keys:
                if key not in param_dict:
                    prior_information_df.loc[method_name, key] = "Not Required"
                    continue

                key_info = param_dict[key]
                if isinstance(key_info, dict) and key_info.get("required", False):
                    prior_information_df.loc[method_name, key] = "Necessary"
                else:
                    prior_information_df.loc[method_name, key] = "Optional"

    prior_information_df = prior_information_df.fillna("Not Required")

    def classify_prior_level(row):
        if all(val == "Not Required" for val in row):
            return "free"
        if any(val == "Necessary" for val in row):
            return "strong restricted"
        if any(val == "Optional" for val in row):
            return "weakly restricted"
        return "unknown"

    method_meta_df[prior_keys] = prior_information_df.loc[method_meta_df.index, prior_keys]
    method_meta_df["prior_level"] = prior_information_df.apply(classify_prior_level, axis=1)
    return method_meta_df


def add_citations_and_stars(method_meta_df, scholarly_sleep=3, github_sleep=3):
    # get google scholar citation and github star for every method
    try:
        scholarly_module = import_module("scholarly")
        scholarly = scholarly_module.scholarly
    except Exception as e:
        logger.warning(f"scholarly is not available, skip citations: {e}")
        scholarly = None

    citations = []
    stars = []
    pub_years = []

    for _, row in method_meta_df.iterrows():
        # 1) google scholar citations by DOI
        doi = row.get("doi") if isinstance(row, pd.Series) else None
        citation_count = np.nan
        pub_year = np.nan

        if scholarly is not None and pd.notna(doi) and str(doi).strip() != "":
            try:
                search_result = scholarly.search_single_pub(str(doi).strip())
                citation_count = search_result.get("num_citations", np.nan)
                pub_year = search_result.get("bib", {}).get("pub_year", np.nan)
                time.sleep(scholarly_sleep)
            except Exception as e:
                logger.debug(f"could not fetch citation for DOI {doi}: {e}")

        citations.append(citation_count)
        pub_years.append(pub_year)

        # 2) github stars
        github_url = row.get("github_url") if isinstance(row, pd.Series) else None
        star_count = np.nan

        if pd.notna(github_url) and str(github_url).strip() != "":
            # Support both with and without protocol, and tolerate trailing slash.
            match = re.search(r"github\.com/([^/]+)/([^/#?]+)", str(github_url))
            if match:
                owner, repo = match.groups()
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                try:
                    response = requests.get(api_url, timeout=10)
                    if response.status_code == 200:
                        star_count = response.json().get("stargazers_count", np.nan)
                    else:
                        logger.debug(f"github api request failed for {github_url}: {response.status_code}")
                except Exception as e:
                    logger.debug(f"could not fetch stars for {github_url}: {e}")
                time.sleep(github_sleep)

        stars.append(star_count)

    method_meta_df["citations"] = pd.Series(citations, index=method_meta_df.index).astype("Int64")
    method_meta_df["stars"] = pd.Series(stars, index=method_meta_df.index).astype("Int64")
    method_meta_df["pub_year"] = pd.Series(pub_years, index=method_meta_df.index).astype("Int64")

    return method_meta_df
