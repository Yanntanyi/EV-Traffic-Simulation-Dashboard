import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import json
import os

# I made a Helper function to convert a hex color to an rgba string
def hex_to_rgba(hex_color, alpha):
    """Convert a hex color (format: '#RRGGBB') to an rgba string with the given alpha."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

##############################
# Data Loading and Processing
##############################
csv_filename = "Electric_Vehicle_Population_Data 2.csv"
if os.path.exists(csv_filename):
    ev_pop = pd.read_csv(csv_filename)
else:
    # Dummy data if CSV file is missing (this is just precautionary)
    data = {
        "Vehicle Location": [
            "POINT(-122.34301 47.659185)",
            "POINT(-122.20578 47.762405)",
            "POINT(-120.6027202 46.5965625)"
        ],
        "Make": ["TESLA", "JEEP", "JEEP"],
        "Electric Vehicle Type": [
            "Battery Electric Vehicle (BEV)",
            "Plug-in Hybrid Electric Vehicle (PHEV)",
            "Plug-in Hybrid Electric Vehicle (PHEV)"
        ],
        "Electric Range": [215, 25, 25],
        "County": ["King", "King", "Yakima"],
        "City": ["Seattle", "Bothell", "Yakima"]
    }
    ev_pop = pd.DataFrame(data)

# Cleaning and processing the data
ev_pop = ev_pop.dropna(subset=["Vehicle Location"])
ev_pop["Vehicle Location"] = ev_pop["Vehicle Location"].astype(str)
ev_pop["geometry"] = ev_pop["Vehicle Location"].apply(
    lambda x: wkt.loads(x) if x.startswith("POINT") else None)
ev_pop = ev_pop.dropna(subset=["geometry"])

# Creating the GeoDataFrame
ev_pop_gdf = gpd.GeoDataFrame(ev_pop, geometry="geometry", crs="EPSG:4326")
ev_pop_gdf["Latitude"] = ev_pop_gdf.geometry.y
ev_pop_gdf["Longitude"] = ev_pop_gdf.geometry.x

# Travel range simulation function (updated)
def simulate_ev_range(lat, lon, ev_range_miles):
    """Calculate max travel coordinates using geodesic with distance in meters."""
    if ev_range_miles <= 0:
        return lat, lon
    distance_meters = ev_range_miles * 1609.34
    destination = geodesic(meters=distance_meters).destination((lat, lon), 90)
    return destination.latitude, destination.longitude

# Applying the simulation
ev_pop_gdf["Max_Travel_Lat"], ev_pop_gdf["Max_Travel_Lon"] = zip(
    *ev_pop_gdf.apply(
        lambda row: simulate_ev_range(row["Latitude"], row["Longitude"], row["Electric Range"]),
        axis=1
    )
)

# Creating the travel GeoDataFrame using the simulated max travel points
ev_travel_gdf = gpd.GeoDataFrame(
    ev_pop_gdf,
    geometry=gpd.points_from_xy(ev_pop_gdf["Max_Travel_Lon"], ev_pop_gdf["Max_Travel_Lat"]),
    crs="EPSG:4326"
)

##############################
# Dashboard Setup
##############################
washington_counties_list = [
    "Adams", "Asotin", "Benton", "Chelan", "Clallam", "Clark", "Columbia", "Cowlitz",
    "Douglas", "Ferry", "Franklin", "Garfield", "Grant", "Grays Harbor", "Island", "Jefferson",
    "King", "Kitsap", "Kittitas", "Klickitat", "Lewis", "Lincoln", "Mason", "Okanogan", "Pacific",
    "Pend Oreille", "Pierce", "San Juan", "Skagit", "Skamania", "Snohomish", "Spokane", "Stevens",
    "Thurston", "Wahkiakum", "Walla Walla", "Whatcom", "Whitman", "Yakima"
]
# Restricting county options to those present in the data and in Washington
county_options = sorted(list(set(ev_travel_gdf["County"]).intersection(set(washington_counties_list))))

# Initializing Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server

# Dashboard Layout
sidebar = dbc.Col([
    html.H4("Simulation Controls", className="mb-4"),
    html.Hr(),
    html.Label("Select EV Models:", className="mb-2"),
    dcc.Dropdown(
        id='model-selector',
        options=[{'label': m, 'value': m} for m in ev_travel_gdf["Make"].unique()],
        multi=True,
        placeholder="All Models"
    ),
    html.Label("Vehicle Type:", className="mt-3 mb-2"),
    dcc.Dropdown(
        id='ev-type-selector',
        options=[
            {'label': 'Battery (BEV)', 'value': 'Battery Electric Vehicle (BEV)'},
            {'label': 'Plug-in Hybrid (PHEV)', 'value': 'Plug-in Hybrid Electric Vehicle (PHEV)'}
        ],
        multi=True,
        placeholder="All Types"
    ),
    html.Label("Max Travel Range (miles):", className="mt-3 mb-2"),
    dcc.Slider(
        id='range-slider',
        min=0,
        max=500,
        step=50,
        value=500,
        marks={i: f"{i}" for i in range(0, 501, 50)}
    ),
    html.Label("County Filter:", className="mt-3 mb-2"),
    dcc.Dropdown(
        id='county-selector',
        options=[{'label': c, 'value': c} for c in county_options],
        multi=True,
        placeholder="All Counties"
    ),
    dbc.Button("Run Simulation", id='run-button', color="primary", className="mt-4 w-100")
], md=3, style={'background-color': '#f8f9fa', 'padding': '20px'})

main_content = dbc.Col([
    html.H2("EV Traffic Simulation Dashboard", className="text-center mb-4"),
    dcc.Graph(id='cluster-map', style={'height': '75vh'}),
    dbc.Row([
        dbc.Col(dcc.Graph(id='cluster-bar')),
        dbc.Col(dcc.Graph(id='type-pie'))
    ], className="mt-4"),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id='deadzone-table',
            columns=[
                {'name': 'Model', 'id': 'Make'},
                {'name': 'Location', 'id': 'City'},
                {'name': 'Range', 'id': 'Electric Range'},
                {'name': 'County', 'id': 'County'}
            ],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={'fontWeight': 'bold'}
        ), md=8)
    ], className="mt-4"),
    html.Div(id='risk-summary', className="mt-4 p-3",
             style={'border': '1px solid #dee2e6', 'border-radius': '5px'})
], md=9)

app.layout = dbc.Container([
    dbc.Row([sidebar, main_content])
], fluid=True)

# Callback to update the dashboard
@app.callback(
    [Output('cluster-map', 'figure'),
     Output('cluster-bar', 'figure'),
     Output('type-pie', 'figure'),
     Output('deadzone-table', 'data'),
     Output('risk-summary', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('model-selector', 'value'),
     State('ev-type-selector', 'value'),
     State('range-slider', 'value'),
     State('county-selector', 'value')]
)
def update_dashboard(n_clicks, models, ev_types, max_range, counties):
    filtered = ev_travel_gdf.copy()
    if models:
        filtered = filtered[filtered["Make"].isin(models)]
    if ev_types:
        filtered = filtered[filtered["Electric Vehicle Type"].isin(ev_types)]
    if counties:
        filtered = filtered[filtered["County"].isin(counties)]
    filtered = filtered[filtered["Electric Range"] <= max_range]
    
    # Performing clustering on simulated max travel points
    coords = np.radians(filtered[["Max_Travel_Lat", "Max_Travel_Lon"]])
    if len(coords) > 0:
        db = DBSCAN(eps=0.001, min_samples=10, metric='haversine').fit(coords)
        filtered["Cluster"] = db.labels_
    else:
        filtered["Cluster"] = -1
    
    dead = filtered[filtered["Cluster"] == -1]
    non_dead = filtered[filtered["Cluster"] != -1]
    
    fig_map = go.Figure()
    
    # adding county boundaries if it is available
    try:
        with open('wa_counties.geojson') as f:
            counties_geo = json.load(f)
        fig_map.update_layout(mapbox_layers=[{
            'source': counties_geo,
            'type': 'line',
            'color': 'gray',
            'opacity': 0.3
        }])
    except Exception as e:
        print("County boundaries load error:", e)
    
    # Creating a discrete color mapping for clusters
    unique_clusters = sorted(non_dead["Cluster"].unique())
    palette = px.colors.qualitative.Plotly
    cluster_color_mapping = {cluster: palette[i % len(palette)] for i, cluster in enumerate(unique_clusters)}
    
    # For each cluster, I had to compute convex hull and add a shaded polygon
    for cluster in unique_clusters:
        cluster_data = non_dead[non_dead["Cluster"] == cluster]
        if len(cluster_data) >= 3:
            points = cluster_data[["Max_Travel_Lat", "Max_Travel_Lon"]].to_numpy()
            try:
                hull = ConvexHull(points)
            except Exception as e:
                print(f"Convex hull error for cluster {cluster}: {e}")
                continue
            hull_points = points[hull.vertices]
            hull_lat = list(hull_points[:, 0]) + [hull_points[0, 0]]
            hull_lon = list(hull_points[:, 1]) + [hull_points[0, 1]]

            count = len(cluster_data)
            top_model = cluster_data["Make"].mode().iloc[0] if not cluster_data["Make"].mode().empty else "N/A"
            avg_range = cluster_data["Electric Range"].mean()
            hover_text = f"Cluster {cluster}<br>Count: {count}<br>Top Model: {top_model}<br>Avg Range: {avg_range:.1f} mi"
            base_color = cluster_color_mapping[cluster]
            fill_color = hex_to_rgba(base_color, 0.2)
            fig_map.add_trace(go.Scattermapbox(
                lat=hull_lat,
                lon=hull_lon,
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=base_color, width=2),
                name=f"Cluster {cluster} Area",
                hoverinfo="text",
                hovertext=hover_text
            ))
    
    # Adding non-dead EV locations (scatter)
    if not non_dead.empty:
        fig_map.add_trace(go.Scattermapbox(
            lat=non_dead["Latitude"],
            lon=non_dead["Longitude"],
            mode="markers",
            marker=dict(
                size=8,
                color=[cluster_color_mapping.get(x, "#000000") for x in non_dead["Cluster"]],
                opacity=0.7
            ),
            hoverinfo="text",
            hovertext=non_dead.apply(lambda x: f"{x['Make']} {x['Model']}<br>Range: {x['Electric Range']} mi", axis=1),
            name="EV Locations"
        ))
    
    # Adding dead zones with a bright red "X"
    if not dead.empty:
        fig_map.add_trace(go.Scattermapbox(
            lat=dead["Latitude"],
            lon=dead["Longitude"],
            mode="markers+text",
            marker=dict(
                size=12,
                color="red",
                symbol="x"
            ),
            text="⚠️",
            textposition="middle center",
            hovertext="No charging station nearby – EV may be stranded here!",
            name="Dead Zones"
        ))
    
    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=dict(lat=47.5, lon=-120.5), zoom=6),
        margin={"r":0, "t":40, "l":0, "b":0},
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title="EV Locations, Cluster Boundaries, and Dead Zones"
    )
    
    # Cluster bar chart (bottom of dashboard)
    cluster_bar = px.bar(
        filtered.groupby("Cluster").size().reset_index(name="Count"),
        x="Cluster",
        y="Count",
        title="EV Clusters by Travel Region"
    )
    
    # Vehicle type pie chart (bottom of dashboard)
    type_pie = px.pie(
        filtered,
        names="Electric Vehicle Type",
        title="BEV vs PHEV Distribution"
    )
    
    # Dead zone table (bottom of dashboard)
    deadzone_table = dead[["Make", "City", "Electric Range", "County"]].to_dict("records")
    
    # Risk summary (dead zone counter and additional info)
    dead_zone_count = len(dead)
    risk_summary = html.Div([
        html.H4("Risk Summary", style={"color": "red"}),
        html.Hr(),
        html.P(f"🚨 EVs in Danger Zones: {dead_zone_count}"),
    ])
    
    return fig_map, cluster_bar, type_pie, deadzone_table, risk_summary

if __name__ == '__main__':
    app.run_server(debug=True, port=8055)