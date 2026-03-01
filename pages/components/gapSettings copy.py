import streamlit as st

def get_gap_settings():
    gap_threshold = st.sidebar.slider(
        "Gap Threshold (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Sets the % gap threshold for UP or DOWN alerts."
    )

    gap_threshold_decimal = gap_threshold / 100.0
    return gap_threshold, gap_threshold_decimal
