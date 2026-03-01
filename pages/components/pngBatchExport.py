# pages/components/pngBatchExport.py

import io
import zipfile

import streamlit as st
import plotly.io as pio
from plotly.graph_objs import Figure


def render_png_batch_download(fig_map: dict[str, Figure]) -> None:
    """
    fig_map: { "NVDA": fig, "AMD": fig, ... }
    Renders a button that downloads a ZIP with one PNG per ticker.
    """
    if not fig_map:
        return

    st.markdown("### 📦 Batch PNG export")

    if st.button("Download PNG batch (.zip)"):
        buf = io.BytesIO()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, fig in fig_map.items():
                try:
                    png_bytes = pio.to_image(fig, format="png", scale=3)
                    zf.writestr(f"{name}.png", png_bytes)
                except Exception as e:
                    # Skip any figure that fails
                    print(f"PNG export failed for {name}: {e}")

        buf.seek(0)

        st.download_button(
            label="⬇️ Save PNG batch",
            data=buf,
            file_name="mike_charts_batch.zip",
            mime="application/zip",
        )
