import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import networkx as nx
from scipy.spatial.distance import euclidean

# DOMAIN SELECTION
domain_options = {
    "Hospital A": "rtls_hospital_a.csv",
    "Warehouse X": "rtls_warehouse_x.csv",
    "Airport Y": "rtls_airport_y.csv"
}

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

#INITIAL PAGE SETUP
if "page" not in st.session_state:
    st.session_state["page"] = "home"

st.set_page_config(layout="wide")

#PAGE 1:HOME
if st.session_state["page"] == "home":
    st.title("📍 RTLS Smart Asset Movement Tracker")
    st.markdown("### Welcome to My Smart Asset Analytics Platform")

    selected_domain = st.selectbox("Select your facility or RTLS domain:", list(domain_options.keys()))
    if st.button("Load Data"):
        st.session_state['selected_domain'] = selected_domain
        st.session_state['file_path'] = domain_options[selected_domain]
        st.session_state["selected_k"] = 6  # default K
        st.session_state["page"] = "movement"

#PAGE 2: MOVEMENT
elif st.session_state["page"] == "movement":
    st.subheader(f"Raw Asset Movements – {st.session_state['selected_domain']}")
    df = load_data(st.session_state['file_path'])

    #GLOBAL SHARED PREPROCESSING
    df_clean = df[(df["X"] < 150) & (df["Y"] < 150)].copy()
    scaler = StandardScaler()
    df_clean[["X_scaled", "Y_scaled"]] = scaler.fit_transform(df_clean[["X", "Y"]])

    selected_k = st.sidebar.slider("Select Number of Zones (K)", min_value=2, max_value=10, value=6)

    kmeans = KMeans(n_clusters=selected_k, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(df_clean[["X_scaled", "Y_scaled"]])

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["X_scaled", "Y_scaled"])
    centers[["X", "Y"]] = scaler.inverse_transform(centers[["X_scaled", "Y_scaled"]])
    centers["Zone ID"] = centers.index

    #PLOT RAW MOVEMENTS
    fig, ax = plt.subplots(figsize=(10, 6))
    for asset_id in df["Asset ID"].unique():
        asset_data = df[df["Asset ID"] == asset_id]
        ax.plot(asset_data["X"], asset_data["Y"], label=asset_id, alpha=0.6)

    ax.set_title("Raw Asset Movement Paths")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    if st.button("🔙 Back to Home"):
        st.session_state["page"] = "home"

    # ANALYSIS OPTIONS
    st.sidebar.header("🧭 Analysis Options")
    analysis_option = st.sidebar.radio("Select Analysis Step",
        ["None","Detect Zones using Clustering","Zone Visit Heatmap","Path Efficiency Score"]
    )

    # ZONE CLUSTERING
    if analysis_option == "Detect Zones using Clustering":
        st.subheader("🔍 Group Movement into Zones using Clustering")
        st.markdown("Each color represents a zone frequently visited by assets.")

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_clean, x="X", y="Y", hue="Cluster", palette="tab10", s=50, ax=ax2)
        ax2.set_title(f"K-Means Zones (K={selected_k})")
        ax2.set_xlabel("X Coordinate")
        ax2.set_ylabel("Y Coordinate")
        ax2.legend(title="Zone ID")
        ax2.grid(True)
        st.pyplot(fig2)

    #HEATMAP
    elif analysis_option == "Zone Visit Heatmap":
        st.subheader("🔥 Zone-wise Heatmap of Visits")
        heatmap_type = st.sidebar.radio("Heatmap Type", ["Combined (All Assets)", "Specific Asset"])

        zone_counts = df_clean["Cluster"].value_counts().sort_index()
        centers["Visits"] = zone_counts.values

        if heatmap_type == "Combined (All Assets)":
            st.markdown("This heatmap shows how frequently zones are visited by all assets.")
            fig3 = px.scatter(
                centers, x="X", y="Y", size="Visits", color="Visits",
                color_continuous_scale="RdYlBu_r", hover_data=["Zone ID", "Visits"],
                title="Combined Heatmap – All Assets"
            )
            fig3.update_traces(marker=dict(line=dict(width=1, color='black')))
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(centers[["Zone ID", "X", "Y", "Visits"]].sort_values("Visits", ascending=False))

        else:
            selected_asset = st.sidebar.selectbox("Select Asset", df_clean["Asset ID"].unique())
            st.markdown(f"Heatmap for **Asset {selected_asset}**")

            asset_data = df_clean[df_clean["Asset ID"] == selected_asset]
            asset_zone_counts = asset_data["Cluster"].value_counts().sort_index()

            centers_asset = centers.copy()
            centers_asset["Visits"] = centers_asset["Zone ID"].map(asset_zone_counts).fillna(0)

            fig4 = px.scatter(
                centers_asset, x="X", y="Y", size="Visits", color="Visits",
                color_continuous_scale="RdYlBu_r", hover_data=["Zone ID", "Visits"],
                title=f"Heatmap – Asset {selected_asset}"
            )
            fig4.update_traces(marker=dict(line=dict(width=1, color='black')))
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(centers_asset[["Zone ID", "X", "Y", "Visits"]].sort_values("Visits", ascending=False))

    # PATH EFFICIENCY
    elif analysis_option == "Path Efficiency Score":
        st.subheader("🚀 Path Efficiency Score of an Asset")
        selected_asset = st.sidebar.selectbox("Select Asset", df_clean["Asset ID"].unique())

        zone_coords = {int(row['Zone ID']): (row['X'], row['Y']) for _, row in centers.iterrows()}
        G = nx.Graph()
        for i in zone_coords:
            for j in zone_coords:
                if i != j:
                    dist = euclidean(zone_coords[i], zone_coords[j])
                    G.add_edge(i, j, weight=dist)

        asset_data = df_clean[df_clean["Asset ID"] == selected_asset].sort_values("Timestamp")
        actual_path = asset_data["Cluster"].tolist()
        actual_path = [z for i, z in enumerate(actual_path) if i == 0 or z != actual_path[i - 1]]

        def heuristic(a, b): return euclidean(zone_coords[a], zone_coords[b])
        def get_optimal_path(start, end):
            try:
                return nx.astar_path(G, start, end, heuristic=heuristic, weight='weight')
            except:
                return []

        def compute_length(path):
            return sum(euclidean(zone_coords[path[i]], zone_coords[path[i+1]]) for i in range(len(path)-1)) if len(path) > 1 else 0

        if len(actual_path) >= 2:
            start_zone, end_zone = actual_path[0], actual_path[-1]
            optimal_path = get_optimal_path(start_zone, end_zone)

            actual_len = compute_length(actual_path)
            optimal_len = compute_length(optimal_path)
            score = round(actual_len / optimal_len, 2) if optimal_len > 0 else 0

            fig5, ax5 = plt.subplots(figsize=(10, 6))
            for z, (x, y) in zone_coords.items():
                ax5.scatter(x, y, c='lightgray', s=300, edgecolors='black')
                ax5.text(x, y, f"Zone {z}", ha='center', va='center', fontsize=9)

            actual_coords = [zone_coords[z] for z in actual_path]
            ax5.plot(*zip(*actual_coords), c='blue', marker='o', label='Actual Path')
            for idx, zone_id in enumerate(actual_path):
                x, y = zone_coords[zone_id]
                ax5.annotate(f"{idx+1}", (x, y), textcoords="offset points", xytext=(0, 5),
                             ha='center', fontsize=8, color='blue')

            if optimal_path:
                optimal_coords = [zone_coords[z] for z in optimal_path]
                ax5.plot(*zip(*optimal_coords), c='green', linestyle='--', marker='x', label='Optimal Path')

            ax5.set_title(f"Path Efficiency – Asset {selected_asset} (Score: {score})")
            ax5.legend()
            ax5.set_xlabel("X Coordinate")
            ax5.set_ylabel("Y Coordinate")
            ax5.grid(True)
            st.pyplot(fig5)

            st.markdown(f"""
            ### Efficiency Score Summary  
            - Start Zone: `{start_zone}`  
            - End Zone: `{end_zone}`  
            - Actual Zone Path: `{actual_path}`
            - Actual Path Length: `{round(actual_len, 2)}`  
            - Optimal Path Length: `{round(optimal_len, 2)}`  
            - **Efficiency Score: `{score}`**
            """)

            if score < 0.7:
                st.warning("⚠️ Low efficiency detected.")
            elif score > 2.0:
                st.error("🔴 High inefficiency – asset took a long detour.")

            if st.button("🔍 View All Asset Efficiency Scores"):
                st.session_state["page"] = "efficiency_summary"
        else:
            st.info("Not enough data to compute efficiency.")

# PAGE 3: LEADERBOARD
elif st.session_state["page"] == "efficiency_summary":
    st.subheader("CHECK YOUR ASSET'S EFFICIENCY")
    df = load_data(st.session_state['file_path'])

    df_clean = df[(df["X"] < 150) & (df["Y"] < 150)].copy()
    scaler = StandardScaler()
    df_clean[["X_scaled", "Y_scaled"]] = scaler.fit_transform(df_clean[["X", "Y"]])
    selected_k = st.session_state["selected_k"]
    kmeans = KMeans(n_clusters=selected_k, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(df_clean[["X_scaled", "Y_scaled"]])

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["X_scaled", "Y_scaled"])
    centers[["X", "Y"]] = scaler.inverse_transform(centers[["X_scaled", "Y_scaled"]])
    centers["Zone ID"] = centers.index

    zone_coords = {int(row['Zone ID']): (row['X'], row['Y']) for _, row in centers.iterrows()}
    G = nx.Graph()
    for i in zone_coords:
        for j in zone_coords:
            if i != j:
                dist = euclidean(zone_coords[i], zone_coords[j])
                G.add_edge(i, j, weight=dist)

    def compute_efficiency(asset_id):
        asset_data = df_clean[df_clean["Asset ID"] == asset_id].sort_values("Timestamp")
        path = asset_data["Cluster"].tolist()
        path = [z for i, z in enumerate(path) if i == 0 or z != path[i - 1]]
        if len(path) < 2: return None

        try:
            optimal = nx.astar_path(G, path[0], path[-1], heuristic=lambda a, b: euclidean(zone_coords[a], zone_coords[b]), weight='weight')
        except:
            return None

        actual_len = sum(euclidean(zone_coords[path[i]], zone_coords[path[i + 1]]) for i in range(len(path) - 1))
        optimal_len = sum(euclidean(zone_coords[optimal[i]], zone_coords[optimal[i + 1]]) for i in range(len(optimal) - 1))
        score = round(actual_len / optimal_len, 2) if optimal_len > 0 else None

        return {
            "Asset ID": asset_id,
            "Start Zone": path[0],
            "End Zone": path[-1],
            "Actual Path Length": round(actual_len, 2),
            "Optimal Path Length": round(optimal_len, 2),
            "Efficiency Score": score
        }

    all_assets = df_clean["Asset ID"].unique()
    summary = [compute_efficiency(a) for a in all_assets]
    summary = [s for s in summary if s]
    df_summary = pd.DataFrame(summary).sort_values("Efficiency Score")

    st.markdown("### Leaderboard")
    st.dataframe(df_summary.reset_index(drop=True), use_container_width=True)

    # 🚨 Alerts for inefficient and extreme cases
    st.markdown("### ⚠️ Low-Efficiency Alerts (Score < 0.7)")
    low_eff = df_summary[df_summary["Efficiency Score"] < 0.7]
    if not low_eff.empty:
        st.dataframe(low_eff.reset_index(drop=True), use_container_width=True)
    else:
        st.success("✅ No inefficient assets found.")

    st.markdown("### 🔴 Extreme Inefficiency Alerts (Score > 2.5)")
    high_eff = df_summary[df_summary["Efficiency Score"] > 2.5]
    if not high_eff.empty:
        st.dataframe(high_eff.reset_index(drop=True), use_container_width=True)
    else:
        st.success("✅ No extreme inefficiencies detected.")

    st.download_button("Download Efficiency Report", df_summary.to_csv(index=False), "efficiency_scores.csv", "text/csv")

    if st.button("🔙 Back to Analysis Page"):
        st.session_state["page"] = "movement"
