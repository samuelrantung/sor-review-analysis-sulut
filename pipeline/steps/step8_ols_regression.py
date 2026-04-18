import logging
import os
import sys

import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import log_step


# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE   = os.path.join(BASE_DIR, 'data', 'output', 'reviews_scored.csv')
OUTPUT_FILE  = os.path.join(BASE_DIR, 'data', 'output', 'ols_regression_results.txt')
OUTPUT_CSV   = os.path.join(BASE_DIR, 'data', 'output', 'ols_coefficients.csv')

# Variable columns from Step 7 output
X_COL = 'service_quality_X'       # Ordinal: -1 (negative service), 0 (absent), 1 (positive service)
M_COL = 'customer_satisfaction_M' # Ordinal: -1 (Negative), 0 (Neutral), 1 (Positive)
Y_COL = 'review_rating'           # Dependent variable (1–5 stars)

# Display labels used in the output report (thesis-aligned naming)
X_LABEL = 'service_quality_X (Kualitas Layanan)'
M_LABEL = 'customer_satisfaction_M (Kepuasan Pelanggan)'
Y_LABEL = 'review_rating_Y (Star Rating)'

# Significance threshold for hypothesis conclusions
ALPHA = 0.05


# ==========================================
# 2. HELPER
# ==========================================
def log_section(text):
    """Logs each line individually to the logger and returns the full string for the report."""
    text_str = str(text)
    for line in text_str.splitlines():
        logging.info(line)
    return text_str


def significance_label(p_value):
    """Returns a human-readable significance label based on p-value."""
    if p_value < 0.001:
        return "Significant (p < 0.001) ***"
    elif p_value < 0.01:
        return f"Significant (p = {p_value:.4f}) **"
    elif p_value < ALPHA:
        return f"Significant (p = {p_value:.4f}) *"
    else:
        return f"Not Significant (p = {p_value:.4f})"


# ==========================================
# 3. MAIN PIPELINE STEP
# ==========================================
def run_ols_regression(input_file, output_file, output_csv):
    """
    Performs OLS regression analysis on the scored review dataset.

    Variable encoding:
      Both X and M are treated as ordinal/cardinal variables and entered
      directly into OLS as integer scores (-1, 0, 1). This approach is
      justified because:
        1. The three categories have a natural and meaningful order.
        2. The intervals between categories are assumed to be equal-spaced.
        3. Each variable produces a single interpretable coefficient,
           consistent with the hypothesis framing in the thesis.

      - X (service_quality_X): -1 (negative service), 0 (absent), 1 (positive)
      - M (customer_satisfaction_M): -1 (Negative), 0 (Neutral), 1 (Positive)
      - Y (review_rating): integer 1–5, treated as continuous for OLS.

    Analysis steps:
      1. Correlation matrix (X, M, Y)
      2. OLS Model 1: X -> M  → tests H1
      3. OLS Model 2: X + M -> Y  → tests H2, H3, H4

    Hypothesis mapping:
      H1: X significantly predicts M (Model 1, path a)
      H2: M significantly predicts Y controlling for X (Model 2, path b)
      H3: X has a direct significant effect on Y controlling for M (Model 2, path c')
      H4: Mediation confirmed if H1 (path a) AND H2 (path b) are both significant
    """
    output_lines = []

    def record(text):
        line = log_section(text)
        output_lines.append(line)

    logging.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    logging.info(f"Loaded {total_rows} rows.")

    # Validate required columns
    required_columns = {X_COL, M_COL, Y_COL}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    # ---- Drop rows with missing values in analysis columns ----
    df = df.dropna(subset=[X_COL, M_COL, Y_COL])
    rows_after_dropna = len(df)
    dropped = total_rows - rows_after_dropna
    if dropped > 0:
        logging.warning(f"Dropped {dropped} rows with missing values. Analysing {rows_after_dropna} rows.")
    else:
        logging.info(f"No missing values found. Analysing {rows_after_dropna} rows.")

    # Log variable distributions
    x_counts = df[X_COL].value_counts().sort_index()
    m_counts = df[M_COL].value_counts().sort_index()
    y_counts = df[Y_COL].value_counts().sort_index()
    logging.info(f"X distribution — {x_counts.to_dict()}")
    logging.info(f"M distribution — {m_counts.to_dict()}")
    logging.info(f"Y distribution — {y_counts.to_dict()}")

    # ---- Stage 1: Correlation Matrix ----
    record("=" * 60)
    record("STAGE 1: CORRELATION MATRIX (X, M, Y)")
    record("=" * 60)
    record("Method: Pearson correlation (point-biserial for ordinal vs continuous)")
    corr_df = df[[X_COL, M_COL, Y_COL]].rename(columns={
        X_COL: X_LABEL, M_COL: M_LABEL, Y_COL: Y_LABEL
    })
    record(corr_df.corr())
    record("")

    coeff_rows = []

    # Tracking significance of key paths for H4
    path_a_significant = False  # X → M (Model 1)
    path_b_significant = False  # M → Y (Model 2)
    model1 = None
    model2 = None

    # ---- Stage 2: OLS Model 1 — X -> M (H1) ----
    record("=" * 60)
    record("STAGE 2: OLS MODEL 1 — X -> M")
    record("Does service quality predict customer sentiment?")
    record("=" * 60)
    try:
        X1 = sm.add_constant(df[X_COL])
        model1 = sm.OLS(df[M_COL], X1).fit()
        record(model1.summary())
        record("")

        # Collect full coefficient table
        conf_int = model1.conf_int()
        for name in model1.params.index:
            coeff_rows.append({
                'model':       'Model 1 (X→M)',
                'predictor':   name,
                'coefficient': round(model1.params[name], 4),
                'std_error':   round(model1.bse[name], 4),
                't_stat':      round(model1.tvalues[name], 4),
                'p_value':     round(model1.pvalues[name], 4),
                'ci_lower':    round(conf_int.loc[name, 0], 4),
                'ci_upper':    round(conf_int.loc[name, 1], 4),
            })

        p_x = model1.pvalues.get(X_COL, 1.0)
        path_a_significant = p_x < ALPHA

        record("-" * 60)
        record("H1 RESULT: Effect of Service Quality (X) on Customer Satisfaction (M)")
        record(f"  {X_LABEL} coefficient: {model1.params.get(X_COL, float('nan')):.4f}")
        record(f"  {significance_label(p_x)}")
        record(f"  Model R²: {model1.rsquared:.4f} | F-stat p: {model1.f_pvalue:.4f}")
        record(f"  H1 Conclusion: {'SUPPORTED — X significantly predicts M.' if path_a_significant else 'NOT SUPPORTED — X does not significantly predict M.'}")
        record("-" * 60)
        record("")

    except Exception as e:
        logging.error(f"Model 1 failed: {e}")
        record(f"[ERROR] Model 1 could not be fitted: {e}")
        record("")

    # ---- Stage 3: OLS Model 2 — X + M -> Y (H2, H3, H4) ----
    record("=" * 60)
    record("STAGE 3: OLS MODEL 2 — X + M -> Y")
    record("Do service quality and customer satisfaction explain review rating?")
    record("=" * 60)

    if model1 is None:
        record("[WARNING] Model 1 failed. H4 (mediation) cannot be fully evaluated.")
        record("")

    try:
        X2 = sm.add_constant(df[[X_COL, M_COL]])
        model2 = sm.OLS(df[Y_COL], X2).fit()
        record(model2.summary())
        record("")

        # Collect full coefficient table
        conf_int2 = model2.conf_int()
        for name in model2.params.index:
            coeff_rows.append({
                'model':       'Model 2 (X+M→Y)',
                'predictor':   name,
                'coefficient': round(model2.params[name], 4),
                'std_error':   round(model2.bse[name], 4),
                't_stat':      round(model2.tvalues[name], 4),
                'p_value':     round(model2.pvalues[name], 4),
                'ci_lower':    round(conf_int2.loc[name, 0], 4),
                'ci_upper':    round(conf_int2.loc[name, 1], 4),
            })

        p_m   = model2.pvalues.get(M_COL, 1.0)
        p_x2  = model2.pvalues.get(X_COL, 1.0)
        path_b_significant = p_m < ALPHA

        record("-" * 60)
        record("H2 RESULT: Effect of Customer Satisfaction (M) on Rating (Y), controlling for X")
        record(f"  {M_LABEL} coefficient: {model2.params.get(M_COL, float('nan')):.4f}")
        record(f"  {significance_label(p_m)}")
        record(f"  H2 Conclusion: {'SUPPORTED — M significantly predicts Y.' if path_b_significant else 'NOT SUPPORTED — M does not significantly predict Y.'}")
        record("")

        x_direct_sig = p_x2 < ALPHA
        record("H3 RESULT: Direct Effect of Service Quality (X) on Rating (Y), controlling for M")
        record(f"  {X_LABEL} coefficient: {model2.params.get(X_COL, float('nan')):.4f}")
        record(f"  {significance_label(p_x2)}")
        record(f"  H3 Conclusion: {'SUPPORTED — X has a significant direct effect on Y.' if x_direct_sig else 'NOT SUPPORTED — X has no significant direct effect on Y after controlling for M.'}")
        record("")

        record("H4 RESULT: Mediation — Does M mediate the effect of X on Y?")
        record(f"  Path a (X → M, Model 1): {'Significant' if path_a_significant else 'Not Significant'}")
        record(f"  Path b (M → Y, Model 2): {'Significant' if path_b_significant else 'Not Significant'}")
        if model1 is None:
            record("  H4 Conclusion: CANNOT BE EVALUATED — Model 1 failed to fit.")
        elif path_a_significant and path_b_significant:
            record("  H4 Conclusion: SUPPORTED — Both path a and path b are significant.")
            record("  Sentiment (M) mediates the effect of Service Quality (X) on Review Rating (Y).")
        elif path_a_significant and not path_b_significant:
            record("  H4 Conclusion: NOT SUPPORTED — Path b (M→Y) is not significant.")
        elif not path_a_significant and path_b_significant:
            record("  H4 Conclusion: NOT SUPPORTED — Path a (X→M) is not significant.")
        else:
            record("  H4 Conclusion: NOT SUPPORTED — Neither path a nor path b is significant.")
        record(f"  Model R²: {model2.rsquared:.4f} | F-stat p: {model2.f_pvalue:.4f}")
        record("-" * 60)
        record("")

    except Exception as e:
        logging.error(f"Model 2 failed: {e}")
        record(f"[ERROR] Model 2 could not be fitted: {e}")
        record("")

    # ---- Save text report ----
    # Expand any multiline strings (e.g. model.summary()) into individual lines
    # before joining, to avoid double newlines in the output file.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    expanded_lines = []
    for entry in output_lines:
        expanded_lines.extend(str(entry).splitlines())
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(expanded_lines))
    logging.info(f"Text report saved to: {output_file}")

    # ---- Save CSV coefficient table ----
    if coeff_rows:
        coeff_df = pd.DataFrame(coeff_rows, columns=[
            'model', 'predictor', 'coefficient', 'std_error',
            't_stat', 'p_value', 'ci_lower', 'ci_upper'
        ])
        coeff_df.to_csv(output_csv, index=False)
        logging.info(f"Coefficient table saved to: {output_csv}")

    r2_m1 = model1.rsquared if model1 is not None else float('nan')
    r2_m2 = model2.rsquared if model2 is not None else float('nan')

    if model1 is None or model2 is None:
        logging.warning("One or more models failed to fit. Check the log for details.")

    h4_status = (
        'CANNOT EVALUATE' if (model1 is None or model2 is None)
        else 'SUPPORTED' if path_a_significant and path_b_significant
        else 'NOT SUPPORTED'
    )

    log_step(
        'Step 8 - OLS Regression',
        f"{rows_after_dropna:,} rows analysed (dropped {dropped}) | "
        f"Model 1 R²: {r2_m1:.4f} | "
        f"Model 2 R²: {r2_m2:.4f} | "
        f"H4 Mediation: {h4_status}"
    )


# ==========================================
# 4. ENTRY POINT
# ==========================================
if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'data', 'output'), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(
                os.path.join(BASE_DIR, 'data', 'output', 'step8_ols_regression.log'),
                encoding='utf-8'
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_ols_regression(INPUT_FILE, OUTPUT_FILE, OUTPUT_CSV)
