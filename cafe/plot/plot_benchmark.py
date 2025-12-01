def style_benchmark(df, low_good_cols=["time", "memory", "cpu"]):
    styled = df.style
    styled = styled.format("{:.4f}")  # format all numbers to 4 decimal places
    styled = styled.background_gradient(subset=low_good_cols, cmap="RdYlGn_r")  # RdYlGn reverse, green for low, red for high.
    high_good_cols = df.columns.difference(low_good_cols)  # other cols are high good
    styled = styled.background_gradient(
        subset=high_good_cols,
        cmap="RdYlGn",  # RdYlGn reverse, green for low, red for high.
    )
    # TODO: save styled dataframe to file
    return styled
