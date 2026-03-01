# pages/components/exportCharts.py

import io
import streamlit as st
import plotly.io as pio

def export_chart_png(fig, filename: str) -> bytes:
    """
    Export a Plotly figure to PNG bytes.

    NOTE: Under the hood Plotly uses kaleido for static image export.
    If kaleido / Chrome are not set up correctly, this will raise.
    """
    try:
        png_bytes = pio.to_image(fig, format="png", width=1200, height=700, scale=2)
        return png_bytes
    except Exception as e:
        st.error(f"PNG export failed for {filename}: {e}")
        return b""


def export_batch(fig_map: dict) -> dict:
    """
    Batch export a dict of {name: fig} → {name: png_bytes}.
    """
    results = {}
    for name, fig in fig_map.items():
        png_bytes = export_chart_png(fig, f"{name}.png")
        if png_bytes:
            results[name] = png_bytes
    return results


def get_download_button(file_bytes: bytes, filename: str, label: str = "Download PNG"):
    """
    Render a Streamlit download button for a single PNG.
    """
    if not file_bytes:
        st.warning(f"No PNG generated for {filename}")
        return

    st.download_button(
        label=label,
        data=file_bytes,
        file_name=filename,
        mime="image/png",
    )
