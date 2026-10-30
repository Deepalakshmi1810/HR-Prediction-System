
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="HR Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HR Prediction Dashboard")
st.write("Explore candidate placement outcomes and recruitment insights.")

# Load cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv("HR_Prediction_Cleaned.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("HR_Prediction_Cleaned.csv not found in the project folder.")
    st.stop()

# Validate required columns
required = ["status", "company_tier", "skills_match_percentage"]

missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Calculate KPIs
total = len(df)
placed = (df["status"] == "Placed").sum()
not_placed = (df["status"] == "Not Placed").sum()
placement_rate = placed / total * 100 if total else 0

st.subheader("Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Candidates", f"{total:,}")
c2.metric("Placed", f"{placed:,}")
c3.metric("Not Placed", f"{not_placed:,}")
c4.metric("Placement Rate", f"{placement_rate:.2f}%")

st.divider()

# Charts
left, right = st.columns(2)
# LEFT SIDE: Placement Status
with left:
    st.subheader("Placement Status Distribution")

    status_counts = (
        df["status"]
        .value_counts()
        .reindex(["Placed", "Not Placed"], fill_value=0)
    )

    st.bar_chart(status_counts)


# RIGHT SIDE: Placement Rate by Company Tier
with right:
    st.subheader("Placement Rate by Company Tier")

    # Remove extra spaces
    df["company_tier"] = (
        df["company_tier"].astype(str).str.strip()
    )

    # Calculate placement rate
    tier_summary = df.groupby("company_tier").agg(
        total_candidates=("status", "size"),
        placed_candidates=(
            "status",
            lambda x: (x.str.strip() == "Placed").sum()
        )
    )

    tier_summary["placement_rate"] = (
        tier_summary["placed_candidates"]
        / tier_summary["total_candidates"] * 100
    ).round(2)

    tier_summary = tier_summary.reindex(
        ["Tier 1", "Tier 2", "Tier 3"]
    )

    st.bar_chart(tier_summary["placement_rate"])


# Display summary table
st.dataframe(tier_summary)



st.subheader("Skills Match vs Placement Status")

skills_summary = (
    df.groupby("status")["skills_match_percentage"]
    .mean()
    .round(2)
)
st.bar_chart(skills_summary)

st.subheader("Candidate Data Explorer")

status_options = ["All"] + sorted(df["status"].dropna().unique().tolist())
selected_status = st.selectbox("Filter by status", status_options)

filtered = df if selected_status == "All" else df[df["status"] == selected_status]

st.write(f"Showing {len(filtered):,} candidates")
st.dataframe(filtered, use_container_width=True)

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered candidate data",
    data=csv_data,
    file_name="HR_Prediction_Filtered_Data.csv",
    mime="text/csv"
)

st.caption(
    "Note: This dashboard describes Placed / Not Placed outcomes. "
    "These labels do not independently confirm whether an offer was accepted."
)
