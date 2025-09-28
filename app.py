import streamlit as st
from streamlit_extensions import volatility_inference_tab, regime_detection_tab, real_data_tab


def main():
    st.title("Inverse PINNs for Volatility & Regimes")

    tab1, tab2, tab3 = st.tabs(["Volatility Inference", "Regime Detection", "Real Market Data"])

    with tab1:
        volatility_inference_tab()

    with tab2:
        regime_detection_tab()

    with tab3:
        real_data_tab()

if __name__ == "__main__":
    main()
