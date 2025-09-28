import streamlit as st
import os
import numpy as np
import torch
from inverse_pinn import InversePINN, Trainer
from detect_regimes import compute_features_from_sigma, run_kmeans
from load_real_data import build_grid_from_yfinance
import simulate_data
import matplotlib.pyplot as plt
import pandas as pd

def volatility_inference_tab():
    st.header("Inverse PINN: Volatility Surface Inference")
    st.write("Train/Infer sigma(S,t) from observed prices (synthetic or uploaded).")

    mode = st.radio("Mode", ["Use Synthetic Data", "Upload CSV", "Use Saved Model"])

    if mode == "Use Synthetic Data":
        dataset = st.selectbox("Scenario", ["regime_switch", "smooth", "heston_like"])
        if st.button("Generate & Train (quick)"):
            st.info("Generating dataset and training (this may take a while).")
            os.makedirs("data", exist_ok=True)
            fname = simulate_data.generate_dataset(
                nS=64, nT=40, scenario=dataset, noise_std=0.005, out_path='data'
            )
            st.write("Dataset saved to", fname)

            model = InversePINN(hidden_V=(128,128), hidden_sigma=(128,128))
            trainer = Trainer(model, data_npz=fname, device='cpu',
                              lambda_data=1.0, lambda_pde=1.0, lambda_reg=1e-5)
            trainer.data_npz = fname  # ensure predict() works

            # Blocking training (consider background for large epochs)
            history = trainer.train(num_epochs=1500, lr=1e-3, save_path='models', print_every=300)

            # Load grids from dataset for saving prediction
            data = np.load(fname)
            S_grid = data['S_grid']
            t_grid = data['t_grid']

            # Predict sigma
            sigmap = trainer.predict()  # returns sigma_hat(S,t)
            os.makedirs("results", exist_ok=True)
            np.savez("results/prediction.npz",
                     S_grid=S_grid,
                     t_grid=t_grid,
                     sigmap=sigmap)
            st.success("Training finished. Saved models/inference in models/ and results/")

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV with columns S,t,price", type=['csv'])
        if uploaded is not None and st.button("Run Inference"):
            df = np.genfromtxt(uploaded, delimiter=',', names=True)
            st.write("Uploading not yet fully implemented in UI. Use CLI for now.")

    else:
        model_file = st.file_uploader("Upload saved model state dict (.pt)", type=['pt'])
        if model_file is not None:
            state = torch.load(model_file)
            model = InversePINN(hidden_V=(128,128), hidden_sigma=(128,128))
            model.load_state_dict(state)
            st.success("Model loaded. Now run prediction on chosen grid.")
            st.write("Grid selection and prediction not implemented in this UI mode.")

def regime_detection_tab():
    st.header("Regime Detection")
    st.write("Detect regimes from recovered volatility surface.")

    if os.path.exists('results/prediction.npz'):
        data = np.load('results/prediction.npz')
        S_grid = data['S_grid']
        t_grid = data['t_grid']
        sigmap = data['sigmap']

        st.write("Loaded results from results/prediction.npz")
        st.write("Sigma shape:", sigmap.shape)

        features = compute_features_from_sigma(sigmap, S_grid)
        k = st.slider("Number of clusters (KMeans)", 2, 5, 2)
        if st.button("Run KMeans"):
            labels, km = run_kmeans(features, k=k)

            st.line_chart({'mean_sigma': features[:,0], 'std_sigma': features[:,1]})
            st.write("Regime labels per time index:", labels.tolist())
            st.bar_chart(labels)

    else:
        st.warning("No prediction results found. Please run Volatility Inference first.")

def real_data_tab():
    st.header("📊 Real Market Data Loader")
    st.write("Fetch real option chain data (SPY by default) from Yahoo Finance and display the price surface.")

    ticker = st.text_input("Ticker Symbol", "SPY")
    expiries = st.slider("Number of Expiries to Fetch", 1, 5, 3)
    strikes = st.slider("Number of Strikes per Expiry", 5, 50, 25)

    if st.button("Fetch Data"):
        try:
            with st.spinner("Fetching data from Yahoo Finance..."):
                S_grid, t_grid, V_grid, S_now = build_grid_from_yfinance(
                    ticker=ticker,
                    expiries=expiries,
                    strikes_per_expiry=strikes
                )
            st.success(f"Fetched data for {ticker}. Current spot price: {S_now:.2f}")

            st.write(f"S grid: {S_grid.shape}, t grid: {t_grid.shape}, V grid: {V_grid.shape}")

            df = pd.DataFrame(V_grid, index=[f"t={t:.3f}" for t in t_grid],
                              columns=[f"S={s:.2f}" for s in S_grid])
            st.dataframe(df)

            fig, ax = plt.subplots(figsize=(8,5))
            c = ax.imshow(V_grid, origin='lower', aspect='auto',
                          extent=[S_grid.min(), S_grid.max(), t_grid.min(), t_grid.max()])
            ax.set_xlabel("Underlying Price (S)")
            ax.set_ylabel("Time to Maturity (t)")
            ax.set_title("Option Price Surface")
            fig.colorbar(c, ax=ax, label="Call Price")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error fetching real market data: {e}")

