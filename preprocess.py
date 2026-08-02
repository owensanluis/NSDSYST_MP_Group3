import pandas as pd

INPUT_PATH = "dataset_phishing.csv"
OUTPUT_PATH = "dataset_phishing_trimmed.csv"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    # direct copy of some features
    out["length_url"] = df["length_url"]
    out["length_hostname"] = df["length_hostname"]
    out["ip"] = df["ip"]
    out["https_token"] = df["https_token"]
    out["nb_subdomains"] = df["nb_subdomains"]
    out["prefix_suffix"] = df["prefix_suffix"]
    out["shortening_service"] = df["shortening_service"]
    out["ratio_digits_url"] = df["ratio_digits_url"]
    out["phish_hints"] = df["phish_hints"]
    out["nb_hyperlinks"] = df["nb_hyperlinks"]
    out["domain_age"] = df["domain_age"]
    out["web_traffic"] = df["web_traffic"]
    out["google_index"] = df["google_index"]

    # aggregated similar stuff
    specialchar_cols = [
        "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and", "nb_or",
        "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash",
        "nb_star", "nb_colon", "nb_comma", "nb_semicolumn", "nb_dollar",
        "nb_space",
    ]
    out["nb_specialchars"] = df[specialchar_cols].sum(axis=1)

    # aggregated brands
    brand_cols = ["domain_in_brand", "brand_in_subdomain", "brand_in_path"]
    out["brand_flags"] = df[brand_cols].max(axis=1)

    # aggregated tld/subdomain abnormality
    tld_subdomain_cols = ["tld_in_path", "tld_in_subdomain", "abnormal_subdomain"]
    out["tld_subdomain_abnormal"] = df[tld_subdomain_cols].max(axis=1)

    # aggregated redirection counts
    out["nb_redirection_total"] = (
        df["nb_redirection"] + df["nb_external_redirection"]
    )

    # aggregated suspicious HTML features
    html_score_cols = [
        "iframe", "popup_window", "onmouseover", "right_clic",
        "empty_title", "domain_in_title", "domain_with_copyright",
    ]
    out["suspicious_html_score"] = df[html_score_cols].sum(axis=1)

    # changed rows from categorical to numeric
    out["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

    return out


def main():
    df = load_data(INPUT_PATH)
    trimmed = build_features(df)

    print(f"Original shape: {df.shape}")
    print(f"Trimmed shape:  {trimmed.shape}")
    print(f"\nClass balance:\n{trimmed['status'].value_counts()}")
    print(f"\nColumns ({len(trimmed.columns)}):")
    for c in trimmed.columns:
        print(f"  - {c}")

    trimmed.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
