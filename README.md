# EV Explorer: Mapping Electric Vehicle Travel and Infrastructure Needs

## Overview and Purpose

This dashboard helps users explore electric vehicle (EV) travel patterns across Washington State. It visualizes how far EVs can travel on a single charge, highlights high-traffic areas where many EVs gather, and identifies "dead zones" where charging infrastructure might be missing. This tool was created initially for personal use to understand EV practicality before moving to downtown Chicago but was expanded to help others considering electric vehicles.

---

## How the Dashboard Works

### 1. Loading and Cleaning Data
- First, the dashboard loads EV data from a file. If no file is provided, it uses sample data.
- It cleans the data by removing missing location entries and prepares the locations for mapping.

### 2. Simulating EV Travel Range
- The dashboard calculates the maximum distance each EV can drive on a single charge.
- This distance is shown on the map so you can see how far vehicles can reach.

### 3. Grouping EV Destinations
- Vehicles that end up in similar locations after driving their maximum range are grouped together using a clustering method.
- Groups of EVs are shown on the map, making it easy to spot busy areas.
- EVs that can't reach any group or charging stations are marked as "dead zones," meaning these spots might need more charging stations.

### 4. Interactive Features
- **Map:**
  - **Clusters:** Shows groups of EVs with details like how many EVs are in each cluster and common vehicle types.
  - **Individual EVs:** Colored dots show exactly where EVs stop.
  - **Dead Zones:** Red markers with warnings highlight areas without charging stations, indicating potential problems.

- **Charts:**
  - **Bar Chart:** Quickly see which areas have the most EVs.
  - **Pie Chart:** Shows what types of EVs (fully electric vs. plug-in hybrid) are most common in different areas.

- **Table and Risk Summary:**
  - Lists EVs stuck in "dead zones" and quickly summarizes how many EVs might face charging problems.

### 5. How to Use the Dashboard
- Select EV types, travel ranges, or specific counties.
- Click "Run Simulation" to see updated results instantly (give it a couple seconds to run).
- Easily test different scenarios to understand where more chargers might help most.

---

## Installation Instructions

To run this dashboard on your computer, follow these steps:

1. **Clone or download the repository:**
```
git clone [repository-link]
cd [repository-folder]
```

2. **Install dependencies:**
```
pip install -r requirements.txt
```

3. **Run the dashboard:**
```
streamlit run ev_dashboard.py or python3 ev_dashboard.py
```

After running the above command, Streamlit will launch the dashboard in your default web browser.

---

## Usage Guide

- **Loading Data:**
  - The dashboard automatically loads EV data from a provided CSV file. If no file is available, it uses sample data.

- **Exploring the Dashboard:**
  - Use the sidebar filters to choose specific EV models, vehicle types, maximum travel ranges, or counties.
  - Click the **"Run Simulation"** button to refresh the visuals with your selected filters.

- **Understanding Visualizations:**
  - **Interactive Map:** View clusters of EVs, individual vehicle locations, and highlighted dead zones.
  - **Bar Chart:** Compare the number of EVs in each area.
  - **Pie Chart:** See the distribution between different types of EVs.
  - **Data Table:** Review a detailed list of EVs located in dead zones and a risk summary indicating areas needing charging stations.

This dashboard provides an interactive, user-friendly tool to assess EV usage patterns and charging infrastructure needs effectively.

---

### Technical Highlights
- Simple and clear visuals to easily understand EV patterns.
- Interactive features like filtering and detailed tooltips help you explore the data easily.
- Clean layout and consistent colors help you quickly identify different EV clusters and problem areas.

